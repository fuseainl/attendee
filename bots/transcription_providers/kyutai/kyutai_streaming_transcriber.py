import logging
import queue
import sys
import threading
import time

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
        # Track last utterance emission time (wall clock)
        self.last_utterance_time = time.time()
        # Track last word end time (audio timestamp in seconds)
        self.last_word_end_time = 0.0

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
            # Add authentication header if API key is provided
            headers = {}
            if self.api_key:
                headers["kyutai-api-key"] = self.api_key

            # Create WebSocket connection
            self.ws = websocket.WebSocketApp(self.server_url, header=headers, on_message=self._on_message, on_error=self._on_error, on_close=self._on_close, on_open=self._on_open)

            # Start WebSocket in a separate thread
            self.receive_thread = threading.Thread(target=self._run_websocket, daemon=True)
            self.receive_thread.start()

            # Wait for connection to be established (with timeout)
            start_time = time.time()
            while not self.connected and time.time() - start_time < 5:
                time.sleep(0.1)

            if not self.connected:
                logger.error("Failed to connect to Kyutai server within timeout")

        except Exception as e:
            logger.error(f"Error connecting to Kyutai server: {e}")

    def _run_websocket(self):
        """Run WebSocket connection in separate thread."""
        try:
            # Run forever with ping/pong keepalive
            # The server may close if it doesn't receive data for a while
            self.ws.run_forever(
                ping_interval=20,  # Send ping every 20 seconds
                ping_timeout=10,  # Timeout if no pong within 10 seconds
                skip_utf8_validation=True,  # Binary frames only, skip validation
            )
        except Exception as e:
            logger.error(f"WebSocket run error: {e}", exc_info=True)

    def _on_open(self, ws):
        """Handle WebSocket connection opened."""
        self.connected = True
        logger.info("🔌 Kyutai WebSocket connection opened")

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

                # Check if there's a significant gap in audio timestamps
                # If the new word starts >1s after the last word ended,
                # emit the previous utterance first
                if self.current_transcript and self.last_word_end_time > 0 and start_time - self.last_word_end_time > 1.0:
                    logger.info(f"Kyutai: Detected {start_time - self.last_word_end_time:.2f}s " f"silence, emitting previous utterance")
                    self._emit_current_utterance()

                # Update last utterance time - we just received a word
                self.last_utterance_time = time.time()

                if text:
                    # Add to current transcript
                    self.current_transcript.append({"text": text, "timestamp": [start_time, start_time]})

            elif msg_type == "EndWord":
                # Update the end time of the last word
                stop_time = data.get("stop_time", 0.0)
                if self.current_transcript:
                    self.current_transcript[-1]["timestamp"][1] = stop_time

                    # Log final word with timing
                    word_data = self.current_transcript[-1]
                    logger.info(f"Kyutai word finalized: '{word_data['text']}' " f"[{word_data['timestamp'][0]:.2f}s - " f"{word_data['timestamp'][1]:.2f}s]")

                    # Check if there's a pause after this word that indicates
                    # end of utterance (we'll detect on next Step)
                    self.last_word_end_time = stop_time

            elif msg_type == "Step":
                # Server processed a step
                # Check for silence-based utterance boundary
                self._check_and_emit_utterance()

            elif msg_type == "Marker":
                # End of stream marker received
                logger.info("Kyutai: End of stream marker received")
                # Emit any remaining transcript
                self._emit_current_utterance()

            elif msg_type == "Ready":
                pass

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
            # Silently drop audio during shutdown - this is expected behavior
            # Audio may still be queued from the meeting adapter while we're closing
            return

        try:
            # Convert int16 bytes to numpy array (zero-copy view)
            audio_samples = np.frombuffer(audio_data, dtype=np.int16)

            # Convert int16 to float32 (normalize to [-1.0, 1.0])
            # Using astype creates a copy, but this is necessary for type conversion
            audio_float = audio_samples.astype(np.float32) / 32768.0

            # Convert to Python list (use .tolist() for performance)
            pcm_list = audio_float.tolist()

            # Pack with MessagePack
            message = msgpack.packb({"type": "Audio", "pcm": pcm_list}, use_bin_type=True, use_single_float=True)

            # Send as BINARY WebSocket frame
            self.ws.send(message, opcode=websocket.ABNF.OPCODE_BINARY)
            self.last_send_time = time.time()

        except Exception as e:
            logger.error(f"Error sending audio to Kyutai: {e}", exc_info=True)

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
        """
        Check if there's a natural pause in speech to emit utterance.

        We emit when there's been > 0.5 seconds of audio silence
        (detected by comparing current audio time vs last word end time).
        """
        if not self.current_transcript:
            return

        # For now, use wall-clock time as a proxy
        # TODO: Track current audio timestamp from server
        current_time = time.time()
        if (current_time - self.last_utterance_time) > 0.5:
            self._emit_current_utterance()

    def _emit_current_utterance(self):
        """Emit the current transcript as an utterance and clear it."""
        if self.current_transcript and self.callback:
            # Convert list of word objects to text
            transcript_text = " ".join([w["text"] for w in self.current_transcript])

            # Compute duration from word timestamps
            # timestamp is [start_time, end_time] in seconds
            start_time = self.current_transcript[0]["timestamp"][0]
            end_time = self.current_transcript[-1]["timestamp"][1]
            duration_seconds = end_time - start_time
            duration_ms = int(duration_seconds * 1000)

            logger.info(f"Kyutai: Emitting utterance ({duration_ms}ms): " f"{transcript_text}")

            # Call callback with duration in metadata
            metadata = {"duration_ms": duration_ms}
            self.callback(transcript_text, metadata)

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
