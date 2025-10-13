import asyncio
import audioop
import logging
import queue
import threading
import time

import msgpack
import numpy as np
import websockets

logger = logging.getLogger(__name__)

# Global callback queue - all speakers enqueue callbacks here
# A single consumer thread processes them sequentially to ensure DB order
_callback_queue = queue.Queue()
_callback_consumer_thread = None
_callback_consumer_running = False
_callback_consumer_lock = threading.Lock()


def _ensure_callback_consumer_started():
    """Ensure the callback consumer thread is running (lazy initialization)."""
    global _callback_consumer_thread, _callback_consumer_running

    # Double-checked locking pattern for thread-safe lazy init
    if _callback_consumer_running:
        return

    with _callback_consumer_lock:
        # Check again inside lock
        if _callback_consumer_running:
            return

        _callback_consumer_running = True

        def consume_callbacks():
            """Process callbacks from queue sequentially."""
            logger.info("Kyutai callback consumer thread started")
            while _callback_consumer_running:
                try:
                    # Wait for callback with timeout to allow graceful shutdown
                    callback_func = _callback_queue.get(timeout=1.0)

                    try:
                        # Execute callback - this writes to DB
                        callback_func()
                    except Exception as e:
                        logger.error(f"Error executing callback: {e}", exc_info=True)
                    finally:
                        _callback_queue.task_done()

                except queue.Empty:
                    # Timeout - check if we should keep running
                    continue

        _callback_consumer_thread = threading.Thread(target=consume_callbacks, daemon=True, name="kyutai-callback-consumer")
        _callback_consumer_thread.start()
        logger.info("Kyutai callback consumer thread initialized")


# Kyutai server expects audio at exactly 24000 Hz
KYUTAI_SAMPLE_RATE = 24000
SAMPLE_WIDTH = 2  # 16-bit PCM
CHANNELS = 1  # mono

# Kyutai's semantic VAD has multiple prediction heads for different pause lengths
# Index 0: 0.5s, Index 1: 1.0s, Index 2: 2.0s, Index 3: 3.0s
# We use 0.5 seconds as a good balance for natural speech segmentation
PAUSE_PREDICTION_HEAD_INDEX = 0
PAUSE_THRESHOLD = 0.25  # Confidence threshold for detecting pauses


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
        debug_logging=False,
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
            debug_logging: Enable verbose debug logging for every message
                (default: False, logs only important events)
        """
        self.server_url = server_url
        self.sample_rate = sample_rate
        self.metadata = metadata or {}
        self.interim_results = interim_results
        self.model = model
        self.api_key = api_key
        self.callback = callback
        self.max_retry_time = max_retry_time
        self.debug_logging = debug_logging

        # Performance optimization: Cache resampling state
        self._resampler_state = None if sample_rate == KYUTAI_SAMPLE_RATE else None

        # Performance optimization: Track audio send stats
        self._audio_chunks_sent = 0
        self._last_log_chunk_count = 0

        # Client-side silence suppression: Stop sending after prolonged silence
        self._silence_suppression_enabled = True  # Enable to reduce server load
        self._silence_chunks_dropped = 0
        self._silence_threshold = 300  # RMS energy threshold (lowered for quiet speakers)
        self._silence_duration_threshold = 5.0  # Stop sending after 5 seconds (more conservative)
        self._consecutive_silence_time = 0.0  # Track continuous silence duration
        self._last_audio_time = time.time()  # Track when we last detected audio

        # Extract participant name from metadata for better logging
        # Metadata uses "participant_full_name" key from adapter
        self._participant_name = metadata.get("participant_full_name", "Unknown") if metadata else "Unknown"

        # Instance identifier for debugging multi-speaker scenarios
        import random

        self._instance_id = random.randint(1000, 9999)

        # No audio queue - send directly to WebSocket for zero latency
        self._audio_sender_task = None
        self._receiver_task = None
        self._ws_connection = None

        # Event loop management - run asyncio in background thread
        self._loop = None
        self._loop_thread = None
        self._connect_future = None

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

        # Rate limiting for utterance emission checks
        self._last_utterance_check_time = 0.0
        self._utterance_check_interval = 0.1  # Check at most every 100ms

        # WebSocket connection
        self.ws = None
        self.connected = False
        self.should_stop = False
        self.finished = False  # Track if finish() has been called

        # Threading for async operations
        self.receive_thread = None
        self.send_thread = None

        # Start event loop in background thread and initialize connection
        self._start_event_loop()

    def _start_event_loop(self):
        """Start asyncio event loop in background thread."""
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._run_event_loop, daemon=True, name="kyutai-event-loop")
        self._loop_thread.start()

        # Wait for loop to start
        time.sleep(0.1)

        # Schedule connection in the loop
        self._connect_future = asyncio.run_coroutine_threadsafe(self._connect(), self._loop)

    def _run_event_loop(self):
        """Run the event loop in background thread."""
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_forever()
        finally:
            self._loop.close()

    async def _connect(self):
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

        while not self.should_stop:
            elapsed_time = time.time() - start_time

            # Check if we've exceeded max retry time
            if elapsed_time >= self.max_retry_time:
                logger.error(f"Failed to connect to Kyutai server after " f"{self.max_retry_time}s. Giving up.")
                return

            try:
                attempt += 1
                logger.info(f"Attempting to connect to Kyutai server " f"(attempt {attempt}, elapsed: {elapsed_time:.1f}s)")

                # Add authentication header if API key is provided
                additional_headers = {}
                if self.api_key:
                    additional_headers["kyutai-api-key"] = self.api_key

                # Connect with websockets library (async!)
                async with websockets.connect(
                    self.server_url,
                    additional_headers=additional_headers,
                    ping_interval=20,  # Send ping every 20 seconds
                    ping_timeout=10,  # Timeout if no pong within 10 seconds
                ) as ws:
                    self._ws_connection = ws
                    self.connected = True
                    logger.info(f"✅ Successfully connected to Kyutai server " f"after {attempt} attempt(s)")

                    # Start receiver task only (no sender task - we send directly)
                    self._receiver_task = asyncio.create_task(self._receiver_loop())

                    # Wait for receiver task
                    await self._receiver_task

                # Connection closed - check if intentional
                if self.should_stop:
                    logger.info("Kyutai connection closed (shutdown)")
                    return

                logger.warning("Kyutai connection closed unexpectedly, will retry...")

            except asyncio.CancelledError:
                logger.info("Kyutai connection cancelled")
                return
            except Exception as e:
                logger.error(f"Error connecting to Kyutai server (attempt {attempt}): {e}")

            # Connection failed, determine retry delay
            if attempt <= len(exponential_delays):
                # Use exponential backoff
                delay = exponential_delays[attempt - 1]
                logger.warning(f"Retrying in {delay}s (exponential backoff)...")
            else:
                # Use fixed delay
                delay = fixed_delay
                logger.warning(f"Retrying in {delay}s...")

            # Check if delay would exceed max retry time
            if elapsed_time + delay > self.max_retry_time:
                remaining_time = self.max_retry_time - elapsed_time
                if remaining_time > 0:
                    logger.info(f"Only {remaining_time:.1f}s remaining before timeout")
                    await asyncio.sleep(remaining_time)
                break
            else:
                await asyncio.sleep(delay)

            # Reset connection state for retry
            self.connected = False
            self._ws_connection = None

    async def _receiver_loop(self):
        """
        Async receiver loop - processes messages from WebSocket.
        """
        try:
            async for message in self._ws_connection:
                await self._process_message(message)
        except websockets.exceptions.ConnectionClosed:
            logger.warning("Kyutai WebSocket connection closed")
        except Exception as e:
            logger.error(f"Error in receiver loop: {e}", exc_info=True)
        finally:
            self.connected = False

    async def _process_message(self, message):
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
                stop_time = data.get("stop_time", None)

                # Debug logging only (verbose)
                if self.debug_logging:
                    wall_clock_now = time.time()
                    audio_offset = None
                    if self.audio_stream_anchor_time is not None:
                        audio_offset = wall_clock_now - self.audio_stream_anchor_time
                    logger.debug(f"Kyutai Word: '{text}' start={start_time:.4f}s " f"offset={audio_offset:.4f}s " f"transcript_len={len(self.current_transcript)}")

                if text:
                    # Check for significant gap - emit previous utterance
                    if self.current_transcript and self.current_utterance_last_word_stop_time is not None and start_time - self.current_utterance_last_word_stop_time > 1.0:
                        if self.debug_logging:
                            gap = start_time - self.current_utterance_last_word_stop_time
                            logger.debug(f"Kyutai: {gap:.2f}s silence, " "emitting utterance")
                        self._emit_current_utterance()

                    # Track first word's start_time for this utterance
                    if not self.current_transcript:
                        self.current_utterance_first_word_start_time = start_time

                    # Track when this word was received (wall clock)
                    self.last_word_received_time = time.time()

                    # Mark that speech has started (for semantic VAD)
                    self.speech_started = True

                    # Add to current transcript
                    self.current_transcript.append({"text": text, "timestamp": [start_time, start_time]})

            elif msg_type == "EndWord":
                # Update the end time of the last word
                stop_time = data.get("stop_time", 0.0)
                if self.current_transcript:
                    # Update timestamp efficiently
                    self.current_transcript[-1]["timestamp"][1] = stop_time

                    # Track the last word's stop time for utterance
                    self.current_utterance_last_word_stop_time = stop_time

                    # Debug logging only
                    if self.debug_logging:
                        word_data = self.current_transcript[-1]
                        logger.debug(f"Kyutai EndWord: '{word_data['text']}' " f"[{word_data['timestamp'][0]:.2f}s - " f"{word_data['timestamp'][1]:.2f}s]")

            elif msg_type == "Step":
                # Step messages contain semantic VAD predictions
                # The "prs" field contains pause predictions for different lengths
                if "prs" in data and len(data["prs"]) > PAUSE_PREDICTION_HEAD_INDEX:
                    pause_prediction = data["prs"][PAUSE_PREDICTION_HEAD_INDEX]

                    # Detect pause: high confidence prediction + speech has started
                    if pause_prediction > PAUSE_THRESHOLD and self.speech_started:
                        self.semantic_vad_detected_pause = True
                        # Emit utterance on natural pause
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

    def send(self, audio_data):
        """
        Send audio data to the Kyutai server.

        Args:
            audio_data: Audio data as bytes (int16 PCM)
        """
        if not self.connected or self.should_stop:
            # Silently drop audio during shutdown - expected behavior
            return

        try:
            # Performance profiling: Track timing for each operation
            t0 = time.perf_counter()

            # Resample if needed (cache resampler state for performance)
            if self.sample_rate != KYUTAI_SAMPLE_RATE:
                audio_data, self._resampler_state = audioop.ratecv(
                    audio_data,
                    SAMPLE_WIDTH,
                    CHANNELS,
                    self.sample_rate,
                    KYUTAI_SAMPLE_RATE,
                    self._resampler_state,
                )
            t1 = time.perf_counter()

            # Convert int16 bytes to float32 in one operation
            # np.frombuffer is zero-copy, astype creates new array
            audio_samples = np.frombuffer(audio_data, dtype=np.int16)
            t2 = time.perf_counter()

            # Client-side silence suppression: Calculate audio energy
            # Use RMS (root mean square) as energy metric
            audio_energy = np.sqrt(np.mean(audio_samples.astype(np.float32) ** 2))

            # Track silence duration (only if suppression is enabled)
            if self._silence_suppression_enabled:
                current_time = time.time()
                if audio_energy < self._silence_threshold:
                    # Silence detected - accumulate silence duration
                    silence_duration = current_time - self._last_audio_time

                    # After threshold seconds of continuous silence, stop sending
                    if silence_duration > self._silence_duration_threshold:
                        self._silence_chunks_dropped += 1

                        # Log when we start suppressing silence
                        if self._silence_chunks_dropped == 1 or self._silence_chunks_dropped % 1000 == 0:
                            logger.info(f"Silence suppression [{self._participant_name}/" f"#{self._instance_id}]: " f"Stopped sending after {silence_duration:.1f}s " f"silence (dropped {self._silence_chunks_dropped} " f"chunks, energy={audio_energy:.1f} < " f"{self._silence_threshold})")
                        return  # Don't send to server
                else:
                    # Audio detected - reset silence tracking
                    if self._silence_chunks_dropped > 0:
                        # Log when we resume after silence
                        silence_duration = current_time - self._last_audio_time
                        logger.info(f"Audio resumed [{self._participant_name}/" f"#{self._instance_id}]: " f"Speech detected after {silence_duration:.1f}s " f"silence (energy={audio_energy:.1f}, dropped " f"{self._silence_chunks_dropped} chunks)")
                        self._silence_chunks_dropped = 0

                    self._last_audio_time = current_time

            # Optimize: Use numpy's in-place division for better performance
            audio_float = audio_samples.astype(np.float32)
            audio_float /= 32768.0
            t3 = time.perf_counter()

            # Pack with MessagePack (tolist() is required for msgpack)
            # NOTE: This is a known bottleneck for multi-speaker scenarios
            audio_list = audio_float.tolist()
            t4 = time.perf_counter()

            message = msgpack.packb(
                {"type": "Audio", "pcm": audio_list},
                use_bin_type=True,
                use_single_float=True,
            )
            t5 = time.perf_counter()

            # Send directly to WebSocket (zero buffering!)
            # Use asyncio.run_coroutine_threadsafe to bridge sync->async
            if self.connected and self._ws_connection:

                async def send_audio():
                    await self._ws_connection.send(message)

                # Fire and forget for minimal latency
                asyncio.run_coroutine_threadsafe(send_audio(), self._loop)
                t6 = time.perf_counter()
            else:
                t6 = time.perf_counter()
                return  # Not connected, drop audio

            # Performance: Log every ~100 chunks instead of time-based
            self._audio_chunks_sent += 1

            # Log first chunk to confirm audio routing is working
            if self._audio_chunks_sent == 1:
                logger.info(f"✅ Kyutai [{self._participant_name}/#{self._instance_id}]: " f"First audio chunk sent (connected={self.connected})")

            if self._audio_chunks_sent % 100 == 0:
                # Calculate timing for each operation
                resample_ms = 1000 * (t1 - t0)
                convert_ms = 1000 * (t3 - t2)
                tolist_ms = 1000 * (t4 - t3)
                msgpack_ms = 1000 * (t5 - t4)
                queue_ms = 1000 * (t6 - t5)
                total_ms = 1000 * (t6 - t0)

                # Calculate sent vs dropped ratio
                total_chunks = self._audio_chunks_sent + self._silence_chunks_dropped
                send_ratio = (self._audio_chunks_sent / total_chunks * 100) if total_chunks > 0 else 0

                logger.info(f"Kyutai Audio Pipeline [{self._participant_name}/" f"#{self._instance_id}] " f"(chunk #{self._audio_chunks_sent}, " f"sent={send_ratio:.1f}%, " f"dropped={self._silence_chunks_dropped}): " f"total={total_ms:.2f}ms | " f"resample={resample_ms:.2f}ms | " f"convert={convert_ms:.2f}ms | " f"tolist={tolist_ms:.2f}ms | " f"msgpack={msgpack_ms:.2f}ms | " f"queue={queue_ms:.2f}ms")
            elif self.debug_logging and self._audio_chunks_sent - self._last_log_chunk_count >= 100:
                logger.debug(f"Kyutai: Sent {self._audio_chunks_sent} audio chunks " f"({len(audio_samples)} samples/chunk, " f"{KYUTAI_SAMPLE_RATE}Hz)")
                self._last_log_chunk_count = self._audio_chunks_sent

        except Exception as e:
            logger.error(f"Error sending audio to Kyutai: {e}", exc_info=True)

    def _check_and_emit_utterance(self):
        """
        Check if there's a natural pause in speech to emit utterance.
        Uses semantic VAD from Kyutai when available, falls back to timing.
        Rate-limited to avoid excessive webhook calls.
        """
        if not self.current_transcript:
            return

        # Check if we've received any words yet
        if self.last_word_received_time is None:
            return

        # Priority 1: Semantic VAD detected a natural pause
        if self.semantic_vad_detected_pause:
            # Only emit if utterance meets quality thresholds:
            # - At least 5 words (complete thought)
            # - OR more than 2 second of speech (substantial content)
            word_count = len(self.current_transcript)

            # Calculate utterance duration if possible
            utterance_duration = 0
            if self.current_utterance_first_word_start_time is not None and self.current_utterance_last_word_stop_time is not None:
                utterance_duration = self.current_utterance_last_word_stop_time - self.current_utterance_first_word_start_time

            # Emit only if meets minimum quality criteria
            if word_count >= 5 or utterance_duration > 2.0:
                logger.info(f"Kyutai [{self._participant_name}]: " f"Emitting utterance on semantic VAD pause " f"({word_count} words, {utterance_duration:.1f}s)")
                self._emit_current_utterance()
                self.semantic_vad_detected_pause = False  # Reset flag
            else:
                # Skip emission - incomplete utterance
                logger.debug(f"Kyutai [{self._participant_name}]: " f"Skipping emission - too short " f"({word_count} words, {utterance_duration:.1f}s)")
                self.semantic_vad_detected_pause = False  # Reset flag
            return

        # Rate limiting: Don't check too frequently (causes webhook spam)
        current_time = time.time()
        time_since_last_check = current_time - self._last_utterance_check_time
        if time_since_last_check < self._utterance_check_interval:
            return  # Skip this check, too soon

        self._last_utterance_check_time = current_time

        # Priority 2: Time-based silence detection (fallback)
        silence_duration = current_time - self.last_word_received_time

        # Require minimum silence before emitting to avoid fragmentation
        MIN_SILENCE_FOR_EMIT = 0.8  # 800ms minimum silence

        # For single-word utterances, be more patient waiting for EndWord
        if len(self.current_transcript) == 1:
            # Wait up to 1.5s for EndWord on single-word utterances
            if self.current_utterance_last_word_stop_time is None:
                if silence_duration > 1.5:
                    logger.info(f"Kyutai [{self._participant_name}]: " f"Single-word utterance, no EndWord after " f"{silence_duration:.2f}s - emitting anyway")
                    self._emit_current_utterance()
            else:
                # Have EndWord, can emit after minimum silence
                if silence_duration > MIN_SILENCE_FOR_EMIT:
                    self._emit_current_utterance()
        else:
            # Multi-word utterance: emit after minimum silence
            if silence_duration > MIN_SILENCE_FOR_EMIT:
                self._emit_current_utterance()

    def _emit_current_utterance(self):
        """Emit the current transcript as an utterance and clear it."""
        if self.current_transcript and self.callback:
            # Convert list of word objects to text efficiently
            transcript_text = " ".join([w["text"] for w in self.current_transcript])

            # Calculate timestamp and duration using audio stream positions
            if self.audio_stream_anchor_time is not None and self.current_utterance_first_word_start_time is not None:
                # Timestamp: When utterance started in wall-clock time
                timestamp_ms = int((self.audio_stream_anchor_time + self.current_utterance_first_word_start_time) * 1000)

                # Duration: Speaking duration from first to last word
                if self.current_utterance_last_word_stop_time is not None:
                    # Have EndWord timing - use it
                    duration_seconds = self.current_utterance_last_word_stop_time - self.current_utterance_first_word_start_time
                    duration_ms = int(duration_seconds * 1000)
                else:
                    # EndWord not received (rare - silence timeout)
                    if self.current_transcript:
                        last_word_start = self.current_transcript[-1]["timestamp"][0]
                        duration_seconds = last_word_start - self.current_utterance_first_word_start_time
                        duration_ms = int(duration_seconds * 1000)
                        if self.debug_logging:
                            logger.warning(f"Kyutai: EndWord missing, " f"est. duration={duration_ms}ms")
                    else:
                        duration_ms = 0
            else:
                # Fallback if we don't have proper anchoring
                if self.debug_logging:
                    logger.warning("Kyutai: Missing timing anchors")
                timestamp_ms = int(time.time() * 1000)
                duration_ms = 0

            # Always log emitted utterances (important for monitoring)
            logger.info(
                f"Kyutai [{self._participant_name}/#{self._instance_id}]: " f"Emitting utterance [{duration_ms}ms, " f"{len(self.current_transcript)} words]: " f"{transcript_text[:100]}"  # Truncate long utterances
            )

            # Call callback with duration and timestamp in metadata
            metadata = {
                "duration_ms": duration_ms,
                "timestamp_ms": timestamp_ms,
            }

            # Enqueue callback to global queue for sequential processing
            # All speakers share one queue, processed by single consumer thread
            # This ensures DB writes happen in chronological order
            def run_callback():
                try:
                    self.callback(transcript_text, metadata)
                except Exception as e:
                    logger.error(f"Error in callback: {e}", exc_info=True)

            # Ensure consumer thread is running (lazy init for Celery workers)
            _ensure_callback_consumer_started()

            # Add to queue - consumer thread will process sequentially
            _callback_queue.put(run_callback)

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
        Fast cleanup optimized for multi-speaker scenarios.
        """
        if self.finished:
            return  # Already finished

        self.finished = True
        logger.info(f"Finishing Kyutai transcriber [{self._participant_name}/" f"#{self._instance_id}]")

        # Emit any remaining transcript before closing
        self._emit_current_utterance()
        self.should_stop = True

        try:
            # Signal stop to async tasks
            if self._loop and self._loop.is_running():
                # Send Marker message to indicate end of stream
                if self.connected and self._ws_connection:

                    async def send_marker_and_close():
                        try:
                            # Send marker (fire and forget)
                            marker_msg = msgpack.packb({"type": "Marker", "id": 0}, use_bin_type=True)
                            await self._ws_connection.send(marker_msg)
                            # Close WebSocket immediately
                            await self._ws_connection.close()
                        except Exception as e:
                            logger.error(f"Error closing WebSocket: {e}")

                    # Schedule close but don't wait for it
                    asyncio.run_coroutine_threadsafe(send_marker_and_close(), self._loop)

                # Stop the event loop immediately (don't wait)
                self._loop.call_soon_threadsafe(self._loop.stop)

            # Don't wait for thread - let it finish in background
            # This releases the connection immediately for other speakers
            logger.info(f"Released connection [{self._participant_name}/" f"#{self._instance_id}] (background cleanup)")

        except Exception as e:
            logger.error(f"Error finishing Kyutai transcriber " f"[{self._participant_name}]: {e}")
        finally:
            self.connected = False
