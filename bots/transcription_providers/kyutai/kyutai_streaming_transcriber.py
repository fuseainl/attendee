import audioop
import logging
import socket
import sys
import threading
import time

import msgpack
import numpy as np
import websocket

logger = logging.getLogger(__name__)

# Kyutai server expects audio at exactly 24000 Hz
KYUTAI_SAMPLE_RATE = 24000
SAMPLE_WIDTH = 2  # 16-bit PCM
CHANNELS = 1  # mono

# Kyutai's semantic VAD has multiple prediction heads for different pause lengths
# Index 0: 0.5s, Index 1: 1.0s, Index 2: 2.0s, Index 3: 3.0s
# We use 0.5 seconds as a good balance for natural speech segmentation
PAUSE_PREDICTION_HEAD_INDEX = 0
PAUSE_THRESHOLD = 0.5  # Confidence threshold for detecting pauses


class KyutaiStreamingTranscriber:
    """
    Streaming transcriber for Kyutai STT service.

    This class handles real-time speech-to-text transcription using
    the Kyutai service, similar to the DeepgramStreamingTranscriber
    but adapted for Kyutai's API.

    Based on:
    https://github.com/kyutai-labs/delayed-streams-modeling/
    """

    def __init__(
        self,
        *,
        server_url,
        sample_rate,
        metadata=None,
        interim_results=True,
        model=None,
        api_key=None,
        callback=None,
        max_retry_time=300,
    ):
        """
        Initialize the Kyutai streaming transcriber.

        Args:
            server_url: URL of the Kyutai server
                (e.g., "ws://localhost:8080")
            sample_rate: Audio sample rate (Kyutai uses 24000 Hz)
            metadata: Optional metadata to send with the connection
            interim_results: Whether to receive interim results
            model: Model name to use (optional)
            api_key: API key for authentication (optional)
            callback: Callback function for utterances
                      (receives transcript text)
            max_retry_time: Maximum time in seconds to keep retrying
                connection (default: 300s / 5 minutes)
        """
        self.server_url = server_url
        self.sample_rate = sample_rate
        self.metadata = metadata or {}
        self.interim_results = interim_results
        self.model = model
        self.api_key = api_key
        self.callback = callback
        self.last_send_time = time.time()
        self.max_retry_time = max_retry_time

        # Track current transcript
        self.current_transcript = []
        # Audio stream anchor: wall-clock time when server sent "Ready"
        # This is the stable reference point for all timestamp calculations
        self.audio_stream_anchor_time = None
        # Track when last word was received (wall clock, for silence detection)
        self.last_word_received_time = None
        # Track audio stream positions for current utterance
        self.current_utterance_first_word_start_time = None  # From "Word"
        self.current_utterance_last_word_stop_time = None  # From "EndWord"

        # Semantic VAD tracking (from Step messages)
        self.semantic_vad_detected_pause = False
        self.speech_started = False  # Track if we've received any words

        # WebSocket connection
        self.ws = None
        self.connected = False
        self.should_stop = False
        self.finished = False  # Track if finish() has been called

        # Threading for async operations
        self.receive_thread = None
        self.send_thread = None

        # Initialize connection
        self._connect()

    def _connect(self):
        """
        Establish WebSocket connection to Kyutai server with retry logic.

        Uses exponential backoff (1s, 2s, 4s, 8s, 16s) followed by
        fixed 10-second intervals until connection succeeds or max_retry_time
        is reached.
        """
        # Exponential backoff delays (in seconds)
        exponential_delays = [1, 2, 4, 8, 16]
        # Fixed delay after exponential backoff exhausted
        fixed_delay = 10

        attempt = 0
        start_time = time.time()

        while True:
            elapsed_time = time.time() - start_time

            # Check if we've exceeded max retry time
            if elapsed_time >= self.max_retry_time:
                logger.error(f"Failed to connect to Kyutai server after " f"{self.max_retry_time}s. Giving up.")
                return

            try:
                attempt += 1
                logger.info(f"Attempting to connect to Kyutai server " f"(attempt {attempt}, elapsed: {elapsed_time:.1f}s)")

                # Clean up any previous failed connection attempt
                if self.ws is not None:
                    try:
                        self.ws.close()
                    except Exception:
                        pass  # Ignore errors closing old connection
                    self.ws = None

                # Reset connection state
                self.connected = False

                # Add authentication header if API key is provided
                headers = {}
                if self.api_key:
                    headers["kyutai-api-key"] = self.api_key

                # Create WebSocket connection with low-latency options
                self.ws = websocket.WebSocketApp(
                    self.server_url,
                    header=headers,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                    on_open=self._on_open,
                )

                # Start WebSocket in a separate thread
                self.receive_thread = threading.Thread(target=self._run_websocket, daemon=True)
                self.receive_thread.start()

                # Wait for connection to be established (with timeout)
                connection_timeout = 5
                connection_start = time.time()
                while not self.connected and time.time() - connection_start < connection_timeout:
                    time.sleep(0.1)

                if self.connected:
                    logger.info(f"Successfully connected to Kyutai server " f"after {attempt} attempt(s)")
                    return

                # Connection failed, clean up before retrying
                if self.ws is not None:
                    try:
                        self.ws.close()
                    except Exception:
                        pass
                    self.ws = None

                # Determine retry delay
                if attempt <= len(exponential_delays):
                    # Use exponential backoff
                    delay = exponential_delays[attempt - 1]
                    logger.warning(f"Failed to connect to Kyutai server " f"(attempt {attempt}). Retrying in {delay}s " f"(exponential backoff)...")
                else:
                    # Use fixed delay
                    delay = fixed_delay
                    logger.warning(f"Failed to connect to Kyutai server " f"(attempt {attempt}). Retrying in {delay}s...")

                # Check if delay would exceed max retry time
                if elapsed_time + delay > self.max_retry_time:
                    remaining_time = self.max_retry_time - elapsed_time
                    if remaining_time > 0:
                        logger.info(f"Only {remaining_time:.1f}s remaining before " f"timeout, waiting that long...")
                        time.sleep(remaining_time)
                    break
                else:
                    time.sleep(delay)

            except Exception as e:
                logger.error(f"Error connecting to Kyutai server " f"(attempt {attempt}): {e}")

                # Clean up on exception
                if self.ws is not None:
                    try:
                        self.ws.close()
                    except Exception:
                        pass
                    self.ws = None

                # Determine retry delay same way as above
                if attempt <= len(exponential_delays):
                    delay = exponential_delays[attempt - 1]
                else:
                    delay = fixed_delay

                # Check if delay would exceed max retry time
                if elapsed_time + delay > self.max_retry_time:
                    remaining_time = self.max_retry_time - elapsed_time
                    if remaining_time > 0:
                        time.sleep(remaining_time)
                    break
                else:
                    time.sleep(delay)

    def _run_websocket(self):
        """Run WebSocket connection in separate thread."""
        try:
            # Run forever with ping/pong keepalive
            # The server may close if it doesn't receive data for a while
            self.ws.run_forever(
                ping_interval=20,  # Send ping every 20 seconds
                ping_timeout=10,  # Timeout if no pong within 10 seconds
                skip_utf8_validation=True,  # Binary frames, skip validation
                sockopt=(
                    (socket.IPPROTO_TCP, socket.TCP_NODELAY, 1),  # Disable Nagle
                ),
            )
        except Exception as e:
            logger.error(f"WebSocket run error: {e}", exc_info=True)

    def _on_open(self, ws):
        """Handle WebSocket connection opened."""
        # Only mark as connected if this is the current websocket instance
        # This prevents race conditions with multiple retry attempts
        if ws == self.ws:
            self.connected = True
            logger.info("🔌 Kyutai WebSocket connection opened")
        else:
            # This is an old connection attempt that succeeded late
            logger.warning("🔌 Kyutai: Closing stale WebSocket connection " "(newer attempt succeeded)")
            try:
                ws.close()
            except Exception:
                pass

    def _on_message(self, ws, message):
        """
        Handle incoming transcription messages from Kyutai server.

        Expected message format (MessagePack):
        - {"type": "Word", "text": "word", "start_time": 0.0}
        - {"type": "EndWord", "stop_time": 0.5}
        - {"type": "Step", ...}
        - {"type": "Marker"}
        """
        # Ignore messages from stale connections
        if ws != self.ws:
            return

        try:
            # Decode MessagePack message
            data = msgpack.unpackb(message, raw=False)

            msg_type = data.get("type")

            if msg_type == "Word":
                # Received a new word
                text = data.get("text", "")
                start_time = data.get("start_time", 0.0)
                stop_time = data.get("stop_time", None)  # May be None until EndWord

                # Log detailed timing information for analysis
                wall_clock_now = time.time()
                audio_offset_from_anchor = None
                if self.audio_stream_anchor_time is not None:
                    audio_offset_from_anchor = wall_clock_now - self.audio_stream_anchor_time

                logger.info(f"Kyutai Word RECEIVED: text='{text}', " f"start_time={start_time:.4f}s, " f"stop_time={stop_time}, " f"wall_clock={wall_clock_now:.4f}, " f"anchor_offset={audio_offset_from_anchor:.4f}s, " f"transcript_len={len(self.current_transcript)}, " f"has_pending_stop={self.current_utterance_last_word_stop_time is not None}")

                if text:
                    # Check if there's a significant gap in audio timestamps
                    # If the new word starts >1s after the last word ended,
                    # emit the previous utterance first
                    if self.current_transcript and self.current_utterance_last_word_stop_time is not None and start_time - self.current_utterance_last_word_stop_time > 1.0:
                        logger.info(f"Kyutai: Detected {start_time - self.current_utterance_last_word_stop_time:.4f}s " f"silence, emitting previous utterance")
                        self._emit_current_utterance()

                    # Track first word's start_time for this utterance
                    if not self.current_transcript:
                        self.current_utterance_first_word_start_time = start_time
                        logger.info(f"Kyutai Word: Starting new utterance - " f"text='{text}', start_time={start_time:.4f}s, " f"anchor_time={self.audio_stream_anchor_time:.4f}")
                    else:
                        logger.info(f"Kyutai Word: Adding to utterance - " f"text='{text}', start_time={start_time:.4f}s, " f"has_stop_time={self.current_utterance_last_word_stop_time is not None}, " f"current_utterance_start={self.current_utterance_first_word_start_time:.4f}s")

                    # Track when this word was received (wall clock, for silence detection)
                    self.last_word_received_time = time.time()

                    # Mark that speech has started (for semantic VAD)
                    self.speech_started = True

                    # Add to current transcript
                    self.current_transcript.append({"text": text, "timestamp": [start_time, start_time]})

            elif msg_type == "EndWord":
                # Update the end time of the last word
                stop_time = data.get("stop_time", 0.0)
                if self.current_transcript:
                    # Calculate timing analysis
                    wall_clock_now = time.time()
                    word_start_time = self.current_transcript[-1]["timestamp"][0]
                    time_since_word_received = wall_clock_now - self.last_word_received_time if self.last_word_received_time else 0
                    audio_duration = stop_time - word_start_time

                    logger.info(f"Kyutai EndWord RECEIVED: stop_time={stop_time:.4f}s, " f"word_start={word_start_time:.4f}s, " f"audio_duration={audio_duration:.4f}s, " f"time_since_word={time_since_word_received:.4f}s, " f"wall_clock={wall_clock_now:.4f}, " f"transcript_len={len(self.current_transcript)}")

                    self.current_transcript[-1]["timestamp"][1] = stop_time

                    # Track the last word's stop time for utterance
                    self.current_utterance_last_word_stop_time = stop_time

                    # Log final word with timing
                    word_data = self.current_transcript[-1]
                    logger.info(f"Kyutai word finalized: '{word_data['text']}' " f"[{word_data['timestamp'][0]:.4f}s - " f"{word_data['timestamp'][1]:.2f}s]")

            elif msg_type == "Step":
                # Step messages contain semantic VAD predictions
                # The "prs" field contains pause predictions for different lengths
                if "prs" in data and len(data["prs"]) > PAUSE_PREDICTION_HEAD_INDEX:
                    pause_prediction = data["prs"][PAUSE_PREDICTION_HEAD_INDEX]

                    # Detect pause: high confidence prediction + speech has started
                    if pause_prediction > PAUSE_THRESHOLD and self.speech_started:
                        logger.info(f"Kyutai: Semantic VAD detected pause " f"(confidence={pause_prediction:.2f})")
                        self.semantic_vad_detected_pause = True
                        # Emit utterance on natural pause
                        self._check_and_emit_utterance()

                # Also check for time-based silence detection as fallback
                self._check_and_emit_utterance()

            elif msg_type == "Marker":
                # End of stream marker received
                logger.info("Kyutai: End of stream marker received")
                # Emit any remaining transcript
                self._emit_current_utterance()

            elif msg_type == "Ready":
                # Server is ready - set our time anchor for timestamp calculations
                # All audio timestamps will be relative to this moment
                self.audio_stream_anchor_time = time.time()
                logger.info("🎯 Kyutai: Audio stream anchor set (Ready signal)")

            else:
                logger.warning(f"Unknown Kyutai message type: {msg_type}")

        except Exception as e:
            logger.error(f"Error processing Kyutai message: {e}")
            logger.debug(f"Raw message: {message}")

    def _on_error(self, ws, error):
        """Handle WebSocket errors."""
        try:
            # Avoid recursion in error logging
            error_msg = str(error)
            logger.error(f"❌ Kyutai WebSocket error: {error_msg}")
        except Exception:
            # If logging fails, just print to avoid recursion
            print(f"Kyutai WebSocket error: {error}", file=sys.stderr)

    def _on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket connection closed."""
        self.connected = False
        logger.warning(f"🔌 Kyutai WebSocket closed: code={close_status_code}, " f"msg={close_msg}, should_stop={self.should_stop}")
        if not self.should_stop:
            logger.error("⚠️  WebSocket closed unexpectedly! " "This may indicate a server issue or protocol mismatch.")

    def send(self, audio_data):
        """
        Send audio data to the Kyutai server.

        Args:
            audio_data: Audio data as bytes (int16 PCM)
        """
        if not self.connected or self.should_stop or not self.receive_thread.is_alive():
            # Silently drop audio during shutdown - expected behavior
            # Audio may still be queued from meeting adapter
            return

        try:
            # Resample from input sample rate to Kyutai's expected 24000 Hz
            if self.sample_rate != KYUTAI_SAMPLE_RATE:
                audio_data, _ = audioop.ratecv(
                    audio_data,
                    SAMPLE_WIDTH,
                    CHANNELS,
                    self.sample_rate,
                    KYUTAI_SAMPLE_RATE,
                    None,
                )

            # Convert int16 bytes to numpy array (zero-copy view)
            audio_samples = np.frombuffer(audio_data, dtype=np.int16)

            # Convert int16 to float32 (normalize to [-1.0, 1.0])
            audio_float = audio_samples.astype(np.float32) / 32768.0

            # Convert to Python list (use .tolist() for performance)
            pcm_list = audio_float.tolist()

            # Pack with MessagePack
            message = msgpack.packb(
                {"type": "Audio", "pcm": pcm_list},
                use_bin_type=True,
                use_single_float=True,
            )

            # Send as BINARY WebSocket frame
            self.ws.send(message, opcode=websocket.ABNF.OPCODE_BINARY)
            self.last_send_time = time.time()

            # Log occasionally to confirm audio is flowing
            if int(time.time() * 10) % 50 == 0:  # Log every ~5 seconds
                logger.info(f"Kyutai: Sent audio chunk " f"({len(audio_samples)} samples at " f"{KYUTAI_SAMPLE_RATE}Hz, {len(message)} bytes)")

        except Exception as e:
            logger.error(f"Error sending audio to Kyutai: {e}", exc_info=True)

    def _check_and_emit_utterance(self):
        """
        Check if there's a natural pause in speech to emit utterance.
        Uses semantic VAD from Kyutai when available, falls back to timing.
        """
        if not self.current_transcript:
            return

        # Check if we've received any words yet
        if self.last_word_received_time is None:
            return

        # Priority 1: Semantic VAD detected a natural pause
        if self.semantic_vad_detected_pause:
            logger.info("Kyutai: Emitting utterance on semantic VAD pause")
            self._emit_current_utterance()
            self.semantic_vad_detected_pause = False  # Reset flag
            return

        # Priority 2: Time-based silence detection (fallback)
        current_time = time.time()
        silence_duration = current_time - self.last_word_received_time

        # For single-word utterances, be more patient waiting for EndWord
        if len(self.current_transcript) == 1:
            # Wait up to 1.0s for EndWord on single-word utterances
            if self.current_utterance_last_word_stop_time is None:
                if silence_duration > 1.0:
                    logger.warning(f"Kyutai: Single-word utterance, no EndWord after " f"{silence_duration:.2f}s - emitting anyway")
                    self._emit_current_utterance()
            else:
                # Have EndWord, can emit after normal 0.25s silence
                if silence_duration > 0.25:
                    self._emit_current_utterance()
        else:
            # Multi-word utterance: emit after 0.5s silence
            if silence_duration > 0.5:
                self._emit_current_utterance()

    def _emit_current_utterance(self):
        """Emit the current transcript as an utterance and clear it."""
        if self.current_transcript and self.callback:
            # Convert list of word objects to text
            transcript_text = " ".join([w["text"] for w in self.current_transcript])

            # Log emission analysis
            wall_clock_now = time.time()
            time_since_first_word = wall_clock_now - self.last_word_received_time if self.last_word_received_time else 0

            logger.info(f"Kyutai EMIT ANALYSIS: transcript='{transcript_text}', " f"word_count={len(self.current_transcript)}, " f"time_since_last_word={time_since_first_word:.4f}s, " f"has_endword={self.current_utterance_last_word_stop_time is not None}, " f"first_word_audio={self.current_utterance_first_word_start_time:.4f}s, " f"last_word_audio={self.current_utterance_last_word_stop_time}")

            # Calculate timestamp and duration using audio stream positions
            # timestamp_ms: anchor_time + first_word_start_time
            # duration_ms: (last_word_stop_time - first_word_start_time)

            if self.audio_stream_anchor_time is not None and self.current_utterance_first_word_start_time is not None:
                # Timestamp: When utterance started in wall-clock time
                # anchor + audio position of first word
                timestamp_ms = int((self.audio_stream_anchor_time + self.current_utterance_first_word_start_time) * 1000)

                # Duration: Speaking duration from first to last word
                if self.current_utterance_last_word_stop_time is not None:
                    # Have EndWord timing - use it
                    duration_seconds = self.current_utterance_last_word_stop_time - self.current_utterance_first_word_start_time
                    duration_ms = int(duration_seconds * 1000)
                    logger.info(f"Kyutai: Using EndWord timing - " f"first={self.current_utterance_first_word_start_time:.2f}s, " f"last={self.current_utterance_last_word_stop_time:.2f}s, " f"duration={duration_ms}ms")
                else:
                    # EndWord not received yet (emitting due to silence timeout)
                    # This should be rare now that we know EndWord timing is reliable
                    # Removed word-length estimation - always use actual timing when available
                    if self.current_transcript:
                        last_word_start = self.current_transcript[-1]["timestamp"][0]
                        duration_seconds = last_word_start - self.current_utterance_first_word_start_time

                        duration_ms = int(duration_seconds * 1000)
                        logger.warning(f"Kyutai: EndWord not received yet - " f"using last word start time as approximation. " f"first={self.current_utterance_first_word_start_time:.2f}s, " f"last_word_start={last_word_start:.2f}s, " f"duration_estimate={duration_ms}ms, " f"transcript_len={len(self.current_transcript)}")
                    else:
                        duration_ms = 0
                        logger.warning("Kyutai: No words in transcript!")

            else:
                # Fallback if we don't have proper anchoring
                logger.warning("Kyutai: Missing timing data - " f"anchor={self.audio_stream_anchor_time is not None}, " f"first_word={self.current_utterance_first_word_start_time}, " f"last_word={self.current_utterance_last_word_stop_time}")
                timestamp_ms = int(time.time() * 1000)
                duration_ms = 0

            logger.info(f"Kyutai: Emitting utterance " f"(timestamp={timestamp_ms}, duration={duration_ms}ms): " f"{transcript_text}")

            # Call callback with duration and timestamp in metadata
            metadata = {
                "duration_ms": duration_ms,
                "timestamp_ms": timestamp_ms,
            }
            self.callback(transcript_text, metadata)

            # Clear transcript for next utterance
            self.current_transcript = []
            # Reset timing for next utterance
            self.current_utterance_first_word_start_time = None
            self.current_utterance_last_word_stop_time = None
            self.last_word_received_time = None
            # Reset semantic VAD state
            self.semantic_vad_detected_pause = False
            self.speech_started = False

    def finish(self):
        """
        Close the connection and clean up resources.
        """
        if self.finished:
            return  # Already finished

        self.finished = True
        logger.info("Finishing Kyutai streaming transcriber")

        # Emit any remaining transcript before closing
        self._emit_current_utterance()
        self.should_stop = True

        try:
            # Send Marker message to indicate end of stream
            if self.connected and self.ws:
                # Include an ID in the marker (server echoes it back)
                marker_msg = msgpack.packb({"type": "Marker", "id": 0}, use_bin_type=True)
                self.ws.send(marker_msg, opcode=websocket.ABNF.OPCODE_BINARY)

                # Wait for server to finish processing and send final words
                # before closing the connection
                time.sleep(0.5)  # Increased from 0.5s to give server more time

                # Close WebSocket
                self.ws.close()

            # Wait for threads to finish
            if self.receive_thread and self.receive_thread.is_alive():
                self.receive_thread.join(timeout=2)

        except Exception as e:
            logger.error(f"Error finishing Kyutai transcriber: {e}")
        finally:
            self.connected = False
