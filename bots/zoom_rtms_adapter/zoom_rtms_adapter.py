import json
import os
import struct
import subprocess
import threading
import time
from datetime import datetime

import gi

from bots.bot_adapter import BotAdapter

gi.require_version("GLib", "2.0")
import logging

from gi.repository import GLib

from bots.automatic_leave_configuration import AutomaticLeaveConfiguration
from bots.models import ParticipantEventTypes

logger = logging.getLogger(__name__)


def iter_annexb_nals(bs: bytes):
    i, n = 0, len(bs)

    def _next_start(i0):
        i = i0
        while i + 3 < n:
            if bs[i] == 0 and bs[i + 1] == 0 and bs[i + 2] == 1:
                return i
            if i + 4 < n and bs[i] == 0 and bs[i + 1] == 0 and bs[i + 2] == 0 and bs[i + 3] == 1:
                return i
            i += 1
        return -1

    start = _next_start(0)
    while start != -1:
        next_start = _next_start(start + 3)
        nal = bs[start : next_start if next_start != -1 else n]
        # nal header is after the start code
        hdr_idx = start + (4 if bs[start + 2] == 0 else 3)
        if hdr_idx < len(bs):
            nal_type = bs[hdr_idx] & 0x1F
            yield nal, nal_type
        start = next_start


def is_keyframe(bs: bytes) -> bool:
    """Return True if this buffer contains SPS(7)/PPS(8) and an IDR(5)."""
    saw_sps = saw_pps = saw_idr = False
    for _, t in iter_annexb_nals(bs):
        if t == 7:
            saw_sps = True
        elif t == 8:
            saw_pps = True
        elif t == 5:
            saw_idr = True
    # IDR is the key; SPS/PPS often precede it (parser can also inject), but prefer all three.
    return saw_idr or (saw_sps and saw_pps)


def make_black_h264_annexb(width: int, height: int, fps=(30, 1)) -> bytes:
    """
    One *AU-aligned* Annex-B black frame (AUD+SPS+PPS+IDR) that matches:
        video/x-h264,stream-format=byte-stream,alignment=au
    """
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    Gst.init(None)

    pipe = Gst.parse_launch(
        f"videotestsrc pattern=black is-live=false num-buffers=1 ! "
        f"video/x-raw,format=I420,width={width},height={height},"
        f"framerate={fps[0]}/{fps[1]} ! "
        # x264 in Annex-B
        "x264enc tune=zerolatency speed-preset=ultrafast key-int-max=1 bframes=0 "
        "byte-stream=true aud=true ! "
        # parse + FORCE downstream caps to Annex-B + AU alignment
        "h264parse disable-passthrough=true ! "
        "video/x-h264,stream-format=byte-stream,alignment=au ! "
        "appsink name=bsink emit-signals=true sync=false max-buffers=1 drop=false"
    )

    sink = pipe.get_by_name("bsink")
    pipe.set_state(Gst.State.PLAYING)
    sample = sink.emit("pull-sample")  # our single AU
    buf = sample.get_buffer()
    data = buf.extract_dup(0, buf.get_size())
    pipe.set_state(Gst.State.NULL)

    # sanity: Annex-B should start with 00 00 00 01
    assert data.startswith(b"\x00\x00\x00\x01") or data.startswith(b"\x00\x00\x01")
    return data


class ZoomRTMSAdapter(BotAdapter):
    def __init__(
        self,
        *,
        use_one_way_audio,
        use_mixed_audio,
        use_video,
        send_message_callback,
        add_audio_chunk_callback,
        zoom_rtms,
        add_video_frame_callback,
        wants_any_video_frames_callback,
        add_mixed_audio_chunk_callback,
        zoom_client_id,
        zoom_client_secret,
        recording_file_path,
        automatic_leave_configuration: AutomaticLeaveConfiguration,
        upsert_chat_message_callback,
        upsert_caption_callback,
        add_participant_event_callback,
        video_frame_size: tuple[int, int],
    ):
        self.zoom_rtms = zoom_rtms
        self.use_one_way_audio = use_one_way_audio
        self.use_mixed_audio = use_mixed_audio
        self.use_video = use_video
        self.send_message_callback = send_message_callback
        self.add_audio_chunk_callback = add_audio_chunk_callback
        self.add_mixed_audio_chunk_callback = add_mixed_audio_chunk_callback
        self.add_video_frame_callback = add_video_frame_callback
        self.wants_any_video_frames_callback = wants_any_video_frames_callback

        self.zoom_client_id = zoom_client_id
        self.zoom_client_secret = zoom_client_secret
        self.recording_file_path = recording_file_path

        self.meeting_service = None
        self.setting_service = None
        self.auth_service = None

        self.auth_event = None
        self.recording_event = None
        self.meeting_service_event = None

        self.audio_source = None
        self.audio_helper = None

        self.audio_settings = None

        self.use_raw_recording = True
        self.recording_permission_granted = False

        self.reminder_controller = None

        self.recording_ctrl = None

        self.audio_raw_data_sender = None
        self.virtual_audio_mic_event_passthrough = None

        self.my_participant_id = None
        self.participants_ctrl = None
        self.meeting_reminder_event = None
        self.on_mic_start_send_callback_called = False
        self.on_virtual_camera_start_send_callback_called = False

        self.meeting_video_controller = None
        self.video_sender = None
        self.virtual_camera_video_source = None
        self.video_source_helper = None
        self.video_frame_size = video_frame_size
        self.send_image_timeout_id = None

        self.automatic_leave_configuration = automatic_leave_configuration

        self.only_one_participant_in_meeting_at = None
        self.last_audio_received_at = None
        self.last_video_received_at = None
        self.silence_detection_activated = False
        self.cleaned_up = False
        self.requested_leave = False
        self.joined_at = None
        self.left_meeting = False

        if self.use_video:
            self.video_input_manager = None
        else:
            self.video_input_manager = None

        self.meeting_sharing_controller = None
        self.meeting_share_ctrl_event = None

        self.active_speaker_id = None
        self.active_sharer_id = None
        self.active_sharer_source_id = None

        self._participant_cache = {}

        self.meeting_status = None

        self.suggested_video_cap = None

        self.upsert_chat_message_callback = upsert_chat_message_callback
        self.add_participant_event_callback = add_participant_event_callback
        self.upsert_caption_callback = upsert_caption_callback

        # Stdout IO watch
        self.stdout_watch_id = None

        self.first_buffer_timestamp_ms = None

        self.rtms_process = None

        # Anonymous pipe FDs (parent reads, child writes)
        self.audio_rfd = None
        self.audio_wfd = None
        self.video_rfd = None
        self.video_wfd = None

        # Pipe reading threads and control
        self.audio_pipe_thread = None
        self.video_pipe_thread = None
        self.stop_pipe_reading = False
        self.pipe_reading_lock = threading.Lock()

        self.last_audio_frame_speaker_name = None
        self.black_frame = make_black_h264_annexb(self.video_frame_size[0], self.video_frame_size[1])

        self.black_frame_timer_id = None
        self.connected_at = None
        self.waiting_for_keyframe = False
        self.last_keyframe_received_at = time.time()
        self.rtms_paused = False

    def send_black_frame(self):
        current_time = time.time()
        if current_time - self.connected_at >= 1.0:
            if self.last_video_received_at is None or current_time - self.last_video_received_at >= 0.25:
                # Create a black frame of the same dimensions
                self._on_video_frame(self.black_frame, self.last_audio_frame_speaker_name, -1)
                logger.info("Sent black frame")

        return not self.cleaned_up

    def _read_exact(self, f, n: int) -> bytes | None:
        """Read exactly n bytes from file-like f (blocking). Return None on EOF."""
        buf = bytearray()
        while len(buf) < n:
            chunk = f.read(n - len(buf))
            if not chunk:
                return None
            buf += chunk
        return bytes(buf)

    def _read_audio_frames_from_fd(self, fd: int, on_frame):
        """
        Blocking loop: read [int32LE username_length][username][int32LE userId][int32LE data_length][payload] audio frames from fd and call on_frame(bytes, username, userId).
        """
        try:
            # buffering=0 gives us unbuffered reads so length boundaries behave nicely
            with os.fdopen(fd, "rb", buffering=0) as f:
                while True:
                    with self.pipe_reading_lock:
                        if self.stop_pipe_reading:
                            break

                    # Audio format: username_length + username + userId + data_length + data
                    username_length_header = self._read_exact(f, 4)
                    if username_length_header is None:
                        break
                    (username_length,) = struct.unpack("<i", username_length_header)

                    if username_length < 0:
                        continue

                    username_bytes = b""
                    if username_length > 0:
                        username_bytes = self._read_exact(f, username_length)
                        if username_bytes is None:
                            break

                    username = username_bytes.decode("utf-8", errors="replace")

                    # Now read the userId
                    user_id_header = self._read_exact(f, 4)
                    if user_id_header is None:
                        break
                    (user_id,) = struct.unpack("<i", user_id_header)

                    # Now read the audio data length and data
                    data_length_header = self._read_exact(f, 4)
                    if data_length_header is None:
                        break
                    (data_length,) = struct.unpack("<i", data_length_header)

                    if data_length <= 0:
                        continue

                    data = self._read_exact(f, data_length)
                    if data is None:
                        break

                    try:
                        on_frame(data, username, user_id)
                    except Exception:
                        logger.exception("Error in audio frame callback")
        except Exception:
            # Normal during shutdown if we close fds
            logger.exception("Audio pipe reader exiting")

    def _read_video_frames_from_fd(self, fd: int, on_frame):
        """
        Blocking loop: read [int32LE username_length][username][int32LE userId][int32LE data_length][payload] video frames from fd and call on_frame(bytes, username, userId).
        """
        try:
            # buffering=0 gives us unbuffered reads so length boundaries behave nicely
            with os.fdopen(fd, "rb", buffering=0) as f:
                while True:
                    with self.pipe_reading_lock:
                        if self.stop_pipe_reading:
                            break

                    # Video format: username_length + username + userId + data_length + data
                    username_length_header = self._read_exact(f, 4)
                    if username_length_header is None:
                        break
                    (username_length,) = struct.unpack("<i", username_length_header)

                    if username_length < 0:
                        continue

                    username_bytes = b""
                    if username_length > 0:
                        username_bytes = self._read_exact(f, username_length)
                        if username_bytes is None:
                            break

                    username = username_bytes.decode("utf-8", errors="replace")

                    # Now read the userId
                    user_id_header = self._read_exact(f, 4)
                    if user_id_header is None:
                        break
                    (user_id,) = struct.unpack("<i", user_id_header)

                    # Now read the video data length and data
                    data_length_header = self._read_exact(f, 4)
                    if data_length_header is None:
                        break
                    (data_length,) = struct.unpack("<i", data_length_header)

                    if data_length <= 0:
                        continue

                    data = self._read_exact(f, data_length)
                    if data is None:
                        break

                    try:
                        on_frame(data, username, user_id)
                    except Exception:
                        logger.exception("Error in video frame callback")
        except Exception:
            # Normal during shutdown if we close fds
            logger.exception("Video pipe reader exiting")

    def _on_audio_frame(self, frame: bytes, userName: str, userId: int):
        """
        Called for each Opus audio frame (mixed, 16kHz mono).

        Args:
            frame: The audio frame data as bytes
            userName: The username associated with the frame
            userId: The user ID as an integer
        """
        self.last_audio_received_at = time.time()
        if not self.rtms_paused:
            self.last_audio_frame_speaker_name = userName
        try:
            if self.use_mixed_audio and self.add_mixed_audio_chunk_callback:
                self.add_mixed_audio_chunk_callback(frame)
            if self.use_one_way_audio and self.add_audio_chunk_callback:
                current_time = datetime.utcnow()
                self.add_audio_chunk_callback(userId, current_time, frame)

        except Exception:
            logger.exception("Audio frame handling failed")

    def _on_video_frame(self, frame: bytes, userName: str, userId: int):
        """
        Called for each H.264 frame with username and user ID.

        Args:
            frame: The video frame data as bytes
            userName: The username associated with the frame
            userId: The user ID as an integer
        """
        if frame != self.black_frame:
            if is_keyframe(frame):
                self.waiting_for_keyframe = False
                self.last_keyframe_received_at = time.time()
                logger.info("Received keyframe")
            else:
                if self.waiting_for_keyframe:
                    logger.info("Received video frame but not a keyframe. Waiting for keyframe...")
                    return

        self.last_video_received_at = time.time()
        try:
            # If you don't currently need video frames, still drain the pipe
            # but skip invoking the callback to save downstream work.
            if self.wants_any_video_frames_callback and not self.wants_any_video_frames_callback():
                return
            if self.add_video_frame_callback:
                # Many pipelines just accept the encoded bytes; if yours needs metadata,
                # pass it here (e.g., codec, width/height).
                self.add_video_frame_callback(frame, time.time_ns(), overlay_text=self.last_audio_frame_speaker_name or "")
        except Exception:
            logger.exception("Video frame handling failed")

    def _start_fd_readers(self):
        """Kick off reader threads for any open read-ends."""
        if self.audio_rfd is not None and (self.audio_pipe_thread is None or not self.audio_pipe_thread.is_alive()):
            self.audio_pipe_thread = threading.Thread(
                target=self._read_audio_frames_from_fd,
                args=(self.audio_rfd, self._on_audio_frame),
                daemon=True,
            )
            self.audio_pipe_thread.start()
            logger.info(f"Audio pipe reader started (fd={self.audio_rfd})")

        if self.video_rfd is not None and (self.video_pipe_thread is None or not self.video_pipe_thread.is_alive()):
            self.video_pipe_thread = threading.Thread(
                target=self._read_video_frames_from_fd,
                args=(self.video_rfd, self._on_video_frame),
                daemon=True,
            )
            self.video_pipe_thread.start()
            logger.info(f"Video pipe reader started (fd={self.video_rfd})")

    def _stop_fd_readers(self):
        """Signal threads to stop; they’ll also exit when the child closes the write ends."""
        with self.pipe_reading_lock:
            self.stop_pipe_reading = True
        for t in (self.audio_pipe_thread, self.video_pipe_thread):
            if t and t.is_alive():
                t.join(timeout=2.0)

        # Close our read ends explicitly (optional; threads should have exited on EOF)
        for fd_attr in ("audio_rfd", "video_rfd"):
            fd = getattr(self, fd_attr, None)
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
                setattr(self, fd_attr, None)

    def cleanup(self):
        logger.info("cleanup called")
        self.cleaned_up = True

        # Remove stdout IO watch
        if self.stdout_watch_id:
            GLib.source_remove(self.stdout_watch_id)
            self.stdout_watch_id = None

        if self.black_frame_timer_id is not None:
            GLib.source_remove(self.black_frame_timer_id)
            self.black_frame_timer_id = None

        self._stop_fd_readers()

    def init(self):
        logger.info("init called")
        self.initialize_rtms_connection()

        return

    def initialize_rtms_connection(self):
        logger.info("Initializing RTMS connection...")

        # Define recording file path - you may want to customize this
        recording_file_path = self.recording_file_path + ".temp"

        # Construct the command
        cmd_env = {
            "ZM_RTMS_CLIENT": self.zoom_client_id,
            "ZM_RTMS_SECRET": self.zoom_client_secret,
        }

        need_audio = self.use_one_way_audio or self.use_mixed_audio
        need_video = self.use_video

        pass_fds = []

        # Create anonymous pipes: parent reads rfd, child writes wfd
        if need_audio:
            self.audio_rfd, self.audio_wfd = os.pipe()
            cmd_env["AUDIO_FD"] = str(self.audio_wfd)
            pass_fds.append(self.audio_wfd)

        if need_video:
            self.video_rfd, self.video_wfd = os.pipe()
            cmd_env["VIDEO_FD"] = str(self.video_wfd)
            pass_fds.append(self.video_wfd)

        cmd = ["node", "/home/nduncan/Documents/attendee_stuff/rtms-developer-preview-js/index.js", "--", f"--recording_file_path={recording_file_path}", f"--join_payload={json.dumps(self.zoom_rtms)}"]

        logger.info(f"Executing RTMS client with command: {' '.join(cmd)}")

        try:
            # Start the subprocess with stdin and stdout pipes opened
            process = subprocess.Popen(
                cmd,
                env=cmd_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True,
                pass_fds=pass_fds,
            )

            # You might want to store the process to interact with it later
            self.rtms_process = process

            # Close the child's write-ends in the parent; we only keep the read-ends
            if self.audio_wfd is not None:
                os.close(self.audio_wfd)
                self.audio_wfd = None
            if self.video_wfd is not None:
                os.close(self.video_wfd)
                self.video_wfd = None

            # Set up stdout monitoring
            self.setup_stdout_monitoring()

            # Start pipe readers right away; they will block until Node opens pipes
            self._start_fd_readers()

            # Start black frame timer
            self.black_frame_timer_id = GLib.timeout_add(250, self.send_black_frame)
            self.connected_at = time.time()

            logger.info("RTMS client started successfully")
            self.send_message_callback({"message": self.Messages.APP_SESSION_CONNECTED})

        except Exception as e:
            logger.error(f"Failed to start RTMS client: {e}")

        return

    def setup_stdout_monitoring(self):
        """Set up GLib IO watch to monitor stdout from the RTMS process."""
        if not self.rtms_process or not self.rtms_process.stdout:
            logger.warning("No stdout available to monitor")
            return

        # Get the file descriptor for stdout
        stdout_fd = self.rtms_process.stdout.fileno()

        # Set up GLib IO watch
        self.stdout_watch_id = GLib.io_add_watch(stdout_fd, GLib.IO_IN | GLib.IO_HUP, self._on_stdout_data_available)
        logger.info("Set up stdout monitoring with GLib IO watch")

    def _on_stdout_data_available(self, fd, condition):
        """Callback called when stdout data is available."""
        if condition & GLib.IO_HUP:
            # Process has closed stdout
            logger.info("RTMS process stdout closed")
            return False  # Remove the watch

        if condition & GLib.IO_IN:
            try:
                # Read a line from stdout
                line = self.rtms_process.stdout.readline()
                if line:
                    # Remove trailing newline and call the callback
                    data = line.rstrip("\n")
                    logger.debug(f"Read from RTMS stdout: {data}")
                    # Handle stdout data internally
                    self.on_stdout_data_received(data)
                else:
                    # EOF reached
                    logger.info("RTMS process stdout EOF")
                    return False  # Remove the watch
            except Exception as e:
                logger.error(f"Error reading stdout: {e}")
                return False  # Remove the watch

        return True  # Continue watching

    def on_stdout_data_received(self, data):
        """
        Handle stdout data received from the RTMS process.
        Override this method or modify it to handle stdout data as needed.

        Args:
            data (str): The stdout line received from the RTMS process
        """
        logger.info(f"RTMS stdout: {data}")
        if data.startswith("rtmsdata."):
            self.handle_rtms_json_message(data[9:])

    def get_participant(self, participant_id):
        return self._participant_cache.get(participant_id)

    def handle_rtms_json_message(self, json_data):
        logger.info(f"handle_rtms_json_message called with json_data: {json_data}")
        json_data = json.loads(json_data)
        if json_data.get("type") == "userUpdate":
            logger.info(f"RTMS userUpdate: {json_data}")
            # {'op': 0, 'user': {'id': 16778240, 'name': 'Noah Duncan'}, 'type': 'userUpdate'}
            user_id = json_data.get("user").get("id")
            user_name = json_data.get("user").get("name")

            self._participant_cache[user_id] = {
                "participant_uuid": user_id,
                "participant_user_uuid": None,
                "participant_full_name": user_name,
                "participant_is_the_bot": False,
            }

            self.add_participant_event_callback({"participant_uuid": user_id, "event_type": ParticipantEventTypes.JOIN if json_data.get("op") == 0 else ParticipantEventTypes.LEAVE, "event_data": {}, "timestamp_ms": int(time.time() * 1000)})
        elif json_data.get("type") == "transcriptUpdate":
            # Don't need captions if we're transcribing from audio
            if self.add_audio_chunk_callback:
                return

            logger.info(f"RTMS transcriptUpdate: {json_data}")
            # {'user': {'userId': 16778240, 'name': 'Noah Duncan'}, 'text': 'Hello, how are you?', 'type': 'transcriptUpdate'}

            itemConverted = {"deviceId": json_data.get("user").get("userId"), "captionId": json_data.get("id"), "text": json_data.get("text"), "isFinal": True}

            self.upsert_caption_callback(itemConverted)

        elif json_data.get("type") == "firstVideoFrameReceived":
            self.first_buffer_timestamp_ms = time.time() * 1000

        elif json_data.get("type") == "sessionUpdate":
            op = json_data.get("op")
            # This means it was paused
            if op == 2:
                logger.info("RTMS sessionUpdate: Paused")
                self.last_audio_frame_speaker_name = "Paused"
                self.rtms_paused = True
            # This means it was resumed
            if op == 3:
                logger.info("RTMS sessionUpdate: Resumed")
                self.last_audio_frame_speaker_name = None
                self.waiting_for_keyframe = True
                self.rtms_paused = False

    def send_to_rtms_stdin(self, data):
        """
        Send data to the RTMS process via stdin.

        Args:
            data (str): The string data to send to the process

        Returns:
            bool: True if data was sent successfully, False otherwise
        """
        if not hasattr(self, "rtms_process") or not self.rtms_process or self.rtms_process.stdin is None:
            logger.warning("No RTMS process available or stdin not accessible")
            return False

        try:
            # Ensure data ends with newline if needed
            if not data.endswith("\n"):
                data += "\n"

            # Write to stdin
            self.rtms_process.stdin.write(data)
            self.rtms_process.stdin.flush()  # Important to flush the buffer

            logger.info(f"Sent data to RTMS process stdin: {data.strip()}")
            return True
        except Exception as e:
            logger.error(f"Failed to send data to RTMS process stdin: {e}")
            return False

    def send_raw_image(self, png_image_bytes):
        return

    def send_raw_audio(self, bytes, sample_rate):
        return

    def disconnect(self):
        if self.left_meeting:
            return
        logger.info("disconnect called")

        self.send_to_rtms_stdin("leave")
        logger.info("sent leave command to RTMS process")
        self.rtms_process.wait(timeout=20)
        logger.info("RTMS process exited")
        self.left_meeting = True
        self.send_message_callback({"message": self.Messages.APP_SESSION_DISCONNECTED})
        return

    def leave(self):
        return self.disconnect()

    def get_first_buffer_timestamp_ms(self):
        return self.first_buffer_timestamp_ms

    def check_auto_leave_conditions(self):
        return

    def is_sent_video_still_playing(self):
        return False

    def send_video(self, video_url):
        logger.info(f"send_video called with video_url = {video_url}. This is not supported for zoom")
        return

    def get_first_buffer_timestamp_ms_offset(self):
        return 0
