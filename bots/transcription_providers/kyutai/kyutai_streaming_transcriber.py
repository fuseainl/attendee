import logging
import queue
import sys
import threading
import time
import traceback

import msgpack
import numpy as np
import websocket

logger = logging.getLogger(__name__)


class KyutaiStreamingTranscriber:
    """
    Streaming transcriber for Kyutai STT service.

    This class handles real-time speech-to-text transcription using
    the Kyutai service, similar to the DeepgramStreamingTranscriber
    but adapted for Kyutai's API.

    Based on:
    https://github.com/kyutai-labs/delayed-streams-modeling/
    blob/main/scripts/stt_from_mic_mlx.py
    """

    def __init__(self, *, server_url, sample_rate, metadata=None, interim_results=True, model=None, api_key=None, callback=None):
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
        """
        self.server_url = server_url
        self.sample_rate = sample_rate
        self.metadata = metadata or {}
        self.interim_results = interim_results
        self.model = model
        self.api_key = api_key
        self.callback = callback
        self.last_send_time = time.time()

        # Track current transcript
        self.current_transcript = []
        # Track last utterance emission time
        self.last_utterance_time = time.time()

        # WebSocket connection
        self.ws = None
        self.connected = False
        self.should_stop = False

        # Audio buffer for chunking
        self.audio_buffer = queue.Queue()

        # Threading for async operations
        self.receive_thread = None
        self.send_thread = None

        # Initialize connection
        self._connect()

    def _connect(self):
        """Establish WebSocket connection to Kyutai server."""
        try:
            # Build WebSocket URL with parameters
            ws_url = self._build_connection_url()

            # Add authentication header if API key is provided
            headers = {}
            if self.api_key:
                headers["kyutai-api-key"] = self.api_key

            # Create WebSocket connection
            self.ws = websocket.WebSocketApp(ws_url, header=headers, on_message=self._on_message, on_error=self._on_error, on_close=self._on_close, on_open=self._on_open)

            # Start WebSocket in a separate thread
            self.receive_thread = threading.Thread(target=self._run_websocket, daemon=True)
            self.receive_thread.start()

            # Wait for connection to be established (with timeout)
            start_time = time.time()
            while not self.connected and time.time() - start_time < 5:
                time.sleep(0.1)

            if not self.connected:
                logger.error("Failed to connect to Kyutai server within timeout")
            else:
                logger.info(f"Connected to Kyutai server at {self.server_url}")

        except Exception as e:
            logger.error(f"Error connecting to Kyutai server: {e}")

    def _build_connection_url(self):
        """Build WebSocket URL with query parameters."""
        # MLX server expects /api/asr-streaming path
        # If URL doesn't already have a path, add it
        if "/api/asr-streaming" not in self.server_url:
            base_url = self.server_url.rstrip("/")
            return f"{base_url}/api/asr-streaming"
        return self.server_url

    def _run_websocket(self):
        """Run WebSocket connection in separate thread."""
        try:
            # Enable trace for debugging
            # websocket.enableTrace(True)

            # Run forever with ping/pong keepalive
            # The server may close if it doesn't receive data for a while
            self.ws.run_forever(
                ping_interval=20,  # Send ping every 20 seconds
                ping_timeout=10,  # Timeout if no pong within 10 seconds
            )
        except Exception as e:
            logger.error(f"WebSocket run error: {e}")

            logger.error(traceback.format_exc())

    def _on_open(self, ws):
        """Handle WebSocket connection opened."""
        self.connected = True
        logger.info("🔌 Kyutai WebSocket connection opened")

        # Log thread status
        if self.receive_thread:
            logger.info(f"   Receive thread alive: {self.receive_thread.is_alive()}")

    def _on_message(self, ws, message):
        """
        Handle incoming transcription messages from Kyutai server.

        Expected message format (MessagePack):
        - {"type": "Word", "text": "word", "start_time": 0.0}
        - {"type": "EndWord", "stop_time": 0.5}
        - {"type": "Step", ...}
        - {"type": "Marker"}
        """
        try:
            # Decode MessagePack message
            data = msgpack.unpackb(message, raw=False)

            msg_type = data.get("type")

            if msg_type == "Word":
                # Received a new word
                text = data.get("text", "")
                start_time = data.get("start_time", 0.0)

                # Update last utterance time - we just received a word
                self.last_utterance_time = time.time()

                if text:
                    # Add to current transcript
                    self.current_transcript.append({"text": text, "timestamp": [start_time, start_time]})

                    # DEBUG: Log each word as it arrives
                    logger.info(f"🎤 Kyutai word received: '{text}' " f"at {start_time:.2f}s")

            elif msg_type == "EndWord":
                # Update the end time of the last word
                stop_time = data.get("stop_time", 0.0)
                if self.current_transcript:
                    self.current_transcript[-1]["timestamp"][1] = stop_time

                    # Log final word with timing
                    word_data = self.current_transcript[-1]
                    logger.info(f"✅ Kyutai word finalized: '{word_data['text']}' " f"[{word_data['timestamp'][0]:.2f}s - " f"{word_data['timestamp'][1]:.2f}s]")

                    # DEBUG: Show accumulated transcript every 5 words
                    if len(self.current_transcript) % 5 == 0:
                        full_text = " ".join([w["text"] for w in self.current_transcript])
                        word_count = len(self.current_transcript)
                        logger.info(f"📝 Kyutai transcript so far " f"({word_count} words): {full_text}")

            elif msg_type == "Step":
                # Server processed a step
                # Check for silence-based utterance boundary
                self._check_and_emit_utterance()

            elif msg_type == "Marker":
                # End of stream marker received
                logger.info("Kyutai: End of stream marker received")
                # Emit any remaining transcript
                self._emit_current_utterance()

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
        Send audio data to Kyutai server for transcription.

        Args:
            audio_data: Raw audio bytes (int16 PCM format) or numpy array
        """
        # Check thread status for logging only - don't reconnect!
        if self.receive_thread and not self.receive_thread.is_alive():
            logger.error("⚠️  Receive thread died!")

        if not self.connected or self.should_stop:
            thread_status = self.receive_thread.is_alive() if self.receive_thread else "None"
            logger.warning(f"Cannot send audio: connected={self.connected}, " f"should_stop={self.should_stop}, " f"thread_alive={thread_status}")
            return

        try:
            # Convert to numpy array if needed
            if isinstance(audio_data, bytes):
                # Convert int16 bytes to numpy array
                audio_samples = np.frombuffer(audio_data, dtype=np.int16)
            elif isinstance(audio_data, np.ndarray):
                audio_samples = audio_data
            else:
                audio_type = type(audio_data)
                logger.error(f"Unsupported audio data type: {audio_type}")
                return

            # Convert int16 to float32 (normalize to [-1.0, 1.0])
            if audio_samples.dtype == np.int16:
                audio_float = audio_samples.astype(np.float32) / 32768.0
            else:
                audio_float = audio_samples.astype(np.float32)

            # Ensure we have valid audio data
            if len(audio_float) == 0:
                logger.warning("Empty audio chunk, skipping")
                return

            # Pack audio message using MessagePack
            # The MLX server expects: {"type": "Audio", "pcm": [f1, f2, ...]}
            # Convert to Python list of native Python floats (not numpy types!)
            # This matches the working microphone client pattern
            pcm_list = [float(x) for x in audio_float]

            if not isinstance(pcm_list, list) or len(pcm_list) == 0:
                pcm_len = len(pcm_list) if isinstance(pcm_list, list) else "N/A"
                logger.warning(f"Invalid PCM data: type={type(pcm_list)}, len={pcm_len}")
                return

            # Verify all elements are floats
            if not all(isinstance(x, (float, int)) for x in pcm_list[:5]):
                logger.error(f"PCM list contains non-numeric data: " f"{[type(x) for x in pcm_list[:5]]}")
                return

            # Log audio amplitude to detect speech vs silence (commented out)
            # max_amplitude = max(abs(x) for x in pcm_list)
            # if max_amplitude > 0.01:  # Speech detected (> 1% amplitude)
            #     logger.info(
            #         f"🎤 SPEECH detected: max_amplitude={max_amplitude:.4f}"
            #     )
            # elif max_amplitude > 0.001:  # Low volume
            #     logger.debug(
            #         f"Low audio: max_amplitude={max_amplitude:.6f}"
            #     )
            # Silent audio (< 0.001) is not logged to reduce noise

            # Pack with MessagePack settings that match the working mic client
            # MUST use: use_bin_type=True, use_single_float=True
            message = msgpack.packb({"type": "Audio", "pcm": pcm_list}, use_bin_type=True, use_single_float=True)

            # Send as BINARY WebSocket frame (CRITICAL!)
            # The server expects binary frames for MessagePack data
            self.ws.send(message, opcode=websocket.ABNF.OPCODE_BINARY)
            self.last_send_time = time.time()

            logger.debug("Message sent successfully")

        except Exception as e:
            logger.error(f"Error sending audio to Kyutai: {e}")
            import traceback

            logger.error(traceback.format_exc())

    def get_transcript_text(self):
        """
        Get the current transcript as a single string.

        Returns:
            str: Complete transcript text
        """
        if not self.current_transcript:
            return ""
        return " ".join([word["text"] for word in self.current_transcript])

    def get_transcript_with_timestamps(self):
        """
        Get the transcript with word-level timestamps.

        Returns:
            list: List of dicts with 'text' and 'timestamp' keys
        """
        return self.current_transcript.copy()

    def _check_and_emit_utterance(self):
        """Check if enough time has passed to emit current utterance."""
        current_time = time.time()
        # If we have transcript and it's been > 2 seconds since last word
        # emit as an utterance
        if self.current_transcript and (current_time - self.last_utterance_time) > 2.0:
            self._emit_current_utterance()

    def _emit_current_utterance(self):
        """Emit the current transcript as an utterance and clear it."""
        if self.current_transcript and self.callback:
            # Convert list of word objects to text
            transcript_text = " ".join([w["text"] for w in self.current_transcript])
            logger.info(f"Kyutai: Emitting utterance: {transcript_text}")
            self.callback(transcript_text)
            # Clear transcript for next utterance
            self.current_transcript = []
            self.last_utterance_time = time.time()

    def finish(self):
        """
        Close the connection and clean up resources.
        """
        logger.info("Finishing Kyutai streaming transcriber")

        # Emit any remaining transcript before closing
        self._emit_current_utterance()

        # DEBUG: Show final complete transcript
        if self.current_transcript:
            full_text = " ".join([w["text"] for w in self.current_transcript])
            word_count = len(self.current_transcript)
            total_duration = self.current_transcript[-1]["timestamp"][1] if self.current_transcript else 0.0
            logger.info("=" * 80)
            logger.info("🎬 KYUTAI FINAL TRANSCRIPT")
            logger.info(f"   Words: {word_count}")
            logger.info(f"   Duration: {total_duration:.2f}s")
            logger.info(f"   Text: {full_text}")
            logger.info("=" * 80)
        else:
            logger.warning("⚠️  No transcript collected from Kyutai")

        self.should_stop = True

        try:
            # Send Marker message to indicate end of stream
            if self.connected and self.ws:
                # Include an ID in the marker (server echoes it back)
                marker_msg = msgpack.packb({"type": "Marker", "id": 0}, use_bin_type=True)
                self.ws.send(marker_msg, opcode=websocket.ABNF.OPCODE_BINARY)

                # Wait for server to finish processing and send final words
                # before closing the connection
                time.sleep(1.0)  # Increased from 0.5s to give server more time

                # Close WebSocket
                self.ws.close()

            # Wait for threads to finish
            if self.receive_thread and self.receive_thread.is_alive():
                self.receive_thread.join(timeout=2)

        except Exception as e:
            logger.error(f"Error finishing Kyutai transcriber: {e}")
        finally:
            self.connected = False
