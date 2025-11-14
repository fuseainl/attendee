import json
import logging
import os
from datetime import datetime

import jwt

from bots.meeting_url_utils import parse_zoom_join_url
from bots.web_bot_adapter import WebBotAdapter
from bots.zoom_web_bot_adapter.zoom_web_ui_methods import ZoomWebUIMethods

logger = logging.getLogger(__name__)


def zoom_meeting_sdk_signature(
    meeting_number: str | int,
    role: int,
    *,
    expiration_seconds: int = 2 * 60 * 60,  # default 2 h
    video_webrtc_mode: int | None = None,
    sdk_key: str | None = None,
    sdk_secret: str | None = None,
) -> dict[str, str]:
    """
    Create a Zoom Meeting SDK JWT signature.

    Parameters
    ----------
    meeting_number : str | int
    role           : 0 for attendee, 1 for host
    expiration_seconds : lifetime for the token (min 1 800, max 172 800)
    video_webrtc_mode  : 0 or 1 (optional)
    sdk_key, sdk_secret: if omitted, read from env vars
                         ZOOM_MEETING_SDK_KEY / ZOOM_MEETING_SDK_SECRET

    Returns
    -------
    {"signature": "<jwt>", "sdkKey": "<sdk_key>"}
    """

    sdk_key = sdk_key or os.getenv("ZOOM_MEETING_SDK_KEY")
    sdk_secret = sdk_secret or os.getenv("ZOOM_MEETING_SDK_SECRET")
    if not sdk_key or not sdk_secret:
        raise RuntimeError("SDK key/secret missing (env vars or arguments)")

    iat = int(datetime.utcnow().timestamp())
    exp = iat + expiration_seconds

    payload = {
        "appKey": sdk_key,
        "sdkKey": sdk_key,
        "mn": str(meeting_number),
        "role": role,
        "iat": iat,
        "exp": exp,
        "tokenExp": exp,
    }
    if video_webrtc_mode is not None:
        payload["video_webrtc_mode"] = video_webrtc_mode

    token = jwt.encode(payload, sdk_secret, algorithm="HS256")
    return {"signature": token, "sdkKey": sdk_key}


class ZoomWebBotAdapter(WebBotAdapter, ZoomWebUIMethods):
    def __init__(
        self,
        *args,
        zoom_client_id: str,
        zoom_client_secret: str,
        zoom_closed_captions_language: str | None,
        should_ask_for_recording_permission: bool,
        zoom_tokens: dict,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.meeting_id, self.meeting_password = parse_zoom_join_url(self.meeting_url)
        self.sdk_signature = zoom_meeting_sdk_signature(self.meeting_id, 0, sdk_key=zoom_client_id, sdk_secret=zoom_client_secret)
        self.zoom_closed_captions_language = zoom_closed_captions_language
        self.should_ask_for_recording_permission = should_ask_for_recording_permission
        self.zoom_tokens = zoom_tokens

    def get_chromedriver_payload_file_name(self):
        return "zoom_web_bot_adapter/zoom_web_chromedriver_payload.js"

    def get_websocket_port(self):
        return 8765

    def is_sent_video_still_playing(self):
        result = self.driver.execute_script("return window.botOutputManager.isVideoPlaying();")
        logger.info(f"is_sent_video_still_playing result = {result}")
        return result

    def send_video(self, video_url):
        logger.info(f"send_video called with video_url = {video_url}")
        self.driver.execute_script(f"window.botOutputManager.playVideo({json.dumps(video_url)})")

    def change_gallery_view_page(self, next_page: bool):
        self.driver.execute_script(f"window?.changeGalleryViewPage({json.dumps(next_page)})")

    def send_chat_message(self, text, to_user_uuid):
        self.driver.execute_script(f"window?.sendChatMessage({json.dumps(text)}, {json.dumps(to_user_uuid)})")

    def get_staged_bot_join_delay_seconds(self):
        return 5

    def subclass_specific_initial_data_code(self):
        return f"""
            window.zoomInitialData = {{
                signature: {json.dumps(self.sdk_signature["signature"])},
                sdkKey: {json.dumps(self.sdk_signature["sdkKey"])},
                meetingNumber: {json.dumps(self.meeting_id)},
                meetingPassword: {json.dumps(self.meeting_password)},
                zakToken: {json.dumps(self.zoom_tokens.get("zak_token", ""))},
                joinToken: {json.dumps(self.zoom_tokens.get("join_token", ""))},
                appPrivilegeToken: {json.dumps(self.zoom_tokens.get("app_privilege_token", ""))},
            }}
        """

    def subclass_specific_after_bot_joined_meeting(self):
        if self.should_ask_for_recording_permission:
            self.driver.execute_script("window?.askForMediaCapturePermission()")
        else:
            self.after_bot_can_record_meeting()

    def subclass_specific_handle_failed_to_join(self, reason):
        # Special case for removed from waiting room
        if reason.get("method") == "removed_from_waiting_room":
            self.send_request_to_join_denied_message()
            return

        if reason.get("method") != "join":
            return

        # Special case for external meeting issue
        if reason.get("errorCode") == 4011:
            self.send_message_callback(
                {
                    "message": self.Messages.ZOOM_MEETING_STATUS_FAILED_UNABLE_TO_JOIN_EXTERNAL_MEETING,
                    "zoom_result_code": str(reason.get("errorCode")) + ": " + str(reason.get("errorMessage")),
                }
            )
            return

        self.send_message_callback(
            {
                "message": self.Messages.ZOOM_MEETING_STATUS_FAILED,
                "zoom_result_code": str(reason.get("errorCode")) + ": " + str(reason.get("errorMessage")),
            }
        )
