import threading
import time
from unittest.mock import MagicMock, patch

import requests
from django.db import connection
from django.test import TransactionTestCase
from selenium.common.exceptions import NoSuchElementException

from bots.bot_controller.bot_controller import BotController
from bots.models import Bot, BotEventManager, BotEventSubTypes, BotEventTypes, BotStates, Credentials, Organization, Project, Recording, RecordingTypes, TranscriptionProviders, TranscriptionTypes, WebhookDeliveryAttempt, WebhookSubscription, WebhookTriggerTypes, ZoomMeetingToZoomOAuthConnectionMapping, ZoomOAuthApp, ZoomOAuthConnection, ZoomOAuthConnectionStates


# Helper functions for creating mocks
def create_mock_file_uploader():
    mock_file_uploader = MagicMock()
    mock_file_uploader.upload_file.return_value = None
    mock_file_uploader.wait_for_upload.return_value = None
    mock_file_uploader.delete_file.return_value = None
    mock_file_uploader.filename = "test-recording-key"
    return mock_file_uploader


def create_mock_zoom_web_driver():
    mock_driver = MagicMock()
    mock_driver.execute_script.return_value = "test_result"
    return mock_driver


class TestZoomWebBot(TransactionTestCase):
    def setUp(self):
        # Recreate organization and project for each test
        self.organization = Organization.objects.create(name="Test Org")
        self.project = Project.objects.create(name="Test Project", organization=self.organization)

        # Recreate zoom oauth app
        self.zoom_credentials = Credentials.objects.create(project=self.project, credential_type=Credentials.CredentialTypes.ZOOM_OAUTH)
        self.zoom_credentials.set_credentials({"client_id": "123", "client_secret": "test_client_secret"})

        # Create a bot for each test
        self.bot = Bot.objects.create(
            name="Test Zoom Web Bot",
            meeting_url="https://zoom.us/j/123123213?p=123123213",
            state=BotStates.READY,
            project=self.project,
            settings={
                "zoom_settings": {
                    "sdk": "web",
                },
            },
        )

        # Create default recording
        self.recording = Recording.objects.create(
            bot=self.bot,
            recording_type=RecordingTypes.AUDIO_AND_VIDEO,
            transcription_type=TranscriptionTypes.NON_REALTIME,
            transcription_provider=TranscriptionProviders.DEEPGRAM,
            is_default_recording=True,
        )

        # Try to transition the state from READY to JOINING
        BotEventManager.create_event(self.bot, BotEventTypes.JOIN_REQUESTED)

    @patch("bots.web_bot_adapter.web_bot_adapter.Display")
    @patch("bots.web_bot_adapter.web_bot_adapter.webdriver.Chrome")
    @patch("bots.bot_controller.bot_controller.S3FileUploader")
    def test_join_meeting(
        self,
        MockFileUploader,
        MockChromeDriver,
        MockDisplay,
    ):
        # Configure the mock uploader
        mock_uploader = create_mock_file_uploader()
        MockFileUploader.return_value = mock_uploader

        # Mock the Chrome driver
        mock_driver = create_mock_zoom_web_driver()
        MockChromeDriver.return_value = mock_driver

        # Mock virtual display
        mock_display = MagicMock()
        MockDisplay.return_value = mock_display

        # Create bot controller
        controller = BotController(self.bot.id)

        # Set up a side effect that raises an exception on first attempt, then succeeds on second attempt
        with patch("bots.zoom_web_bot_adapter.zoom_web_ui_methods.ZoomWebUIMethods.attempt_to_join_meeting") as mock_attempt_to_join:
            mock_attempt_to_join.side_effect = [
                None,  # First call succeeds
            ]

            # Run the bot in a separate thread since it has an event loop
            bot_thread = threading.Thread(target=controller.run)
            bot_thread.daemon = True
            bot_thread.start()

            # Allow time for the retry logic to run
            time.sleep(5)

            # Simulate meeting ending to trigger cleanup
            controller.adapter.only_one_participant_in_meeting_at = time.time() - 10000000000
            time.sleep(4)

            # Verify the attempt_to_join_meeting method was called twice
            self.assertEqual(mock_attempt_to_join.call_count, 1, "attempt_to_join_meeting should be called once")

            # Verify joining succeeded after retry by checking that these methods were called
            self.assertTrue(mock_driver.execute_script.called, "execute_script should be called after join")

            # Now wait for the thread to finish naturally
            bot_thread.join(timeout=5)  # Give it time to clean up

            # If thread is still running after timeout, that's a problem to report
            if bot_thread.is_alive():
                print("WARNING: Bot thread did not terminate properly after cleanup")

            # Close the database connection since we're in a thread
            connection.close()

    @patch("bots.web_bot_adapter.web_bot_adapter.Display")
    @patch("bots.web_bot_adapter.web_bot_adapter.webdriver.Chrome")
    @patch("bots.bot_controller.bot_controller.S3FileUploader")
    @patch("bots.bot_controller.screen_and_audio_recorder.ScreenAndAudioRecorder.pause_recording", return_value=True)
    @patch("bots.bot_controller.screen_and_audio_recorder.ScreenAndAudioRecorder.resume_recording", return_value=True)
    def test_recording_permission_denied(
        self,
        mock_pause_recording,
        mock_resume_recording,
        MockFileUploader,
        MockChromeDriver,
        MockDisplay,
    ):
        # Configure the mock uploader
        mock_uploader = create_mock_file_uploader()
        MockFileUploader.return_value = mock_uploader

        # Mock the Chrome driver
        mock_driver = create_mock_zoom_web_driver()
        MockChromeDriver.return_value = mock_driver

        # Mock virtual display
        mock_display = MagicMock()
        MockDisplay.return_value = mock_display

        # Create bot controller
        controller = BotController(self.bot.id)

        # Set up a side effect that succeeds on joining meeting
        with patch("bots.zoom_web_bot_adapter.zoom_web_ui_methods.ZoomWebUIMethods.attempt_to_join_meeting") as mock_attempt_to_join:
            mock_attempt_to_join.side_effect = [
                None,  # First call succeeds
            ]

            # Run the bot in a separate thread since it has an event loop
            bot_thread = threading.Thread(target=controller.run)
            bot_thread.daemon = True
            bot_thread.start()

            # Allow time for join processing
            time.sleep(2)

            # Simulate recording permission denied by calling the method directly
            # This simulates what would happen when a RecordingPermissionChange message
            # with "denied" change is received via websocket
            controller.adapter.after_bot_recording_permission_denied()

            # Allow time for the message to be processed
            time.sleep(2)

            # Verify that the adapter's pause_recording() method was called
            # The adapter is WebBotAdapter which sets recording_paused = True
            self.assertTrue(controller.adapter.recording_paused, "Adapter's recording_paused flag should be True after permission denied")

            # Refresh bot from database to check state changes
            self.bot.refresh_from_db()

            # Verify that the bot state changed to JOINED_RECORDING_PERMISSION_DENIED
            self.assertEqual(self.bot.state, BotStates.JOINED_RECORDING_PERMISSION_DENIED, "Bot should be in JOINED_RECORDING_PERMISSION_DENIED state after permission denied")

            # Verify that a BOT_RECORDING_PERMISSION_DENIED event was created
            permission_denied_events = self.bot.bot_events.filter(event_type=BotEventTypes.BOT_RECORDING_PERMISSION_DENIED, event_sub_type=BotEventSubTypes.BOT_RECORDING_PERMISSION_DENIED_HOST_DENIED_PERMISSION)
            self.assertTrue(permission_denied_events.exists(), "A BOT_RECORDING_PERMISSION_DENIED event should be created")

            # Simulate meeting ending to trigger cleanup
            controller.adapter.only_one_participant_in_meeting_at = time.time() - 10000000000
            time.sleep(4)

            # Verify the attempt_to_join_meeting method was called once
            self.assertEqual(mock_attempt_to_join.call_count, 1, "attempt_to_join_meeting should be called once")

            # Now wait for the thread to finish naturally
            bot_thread.join(timeout=5)  # Give it time to clean up

            # If thread is still running after timeout, that's a problem to report
            if bot_thread.is_alive():
                print("WARNING: Bot thread did not terminate properly after cleanup")

            # Close the database connection since we're in a thread
            connection.close()

    @patch("bots.zoom_oauth_connections_utils.requests.post")
    @patch("bots.web_bot_adapter.web_bot_adapter.Display")
    @patch("bots.web_bot_adapter.web_bot_adapter.webdriver.Chrome")
    @patch("bots.bot_controller.bot_controller.S3FileUploader")
    def test_zoom_oauth_app_token_failure_with_local_recording_token(
        self,
        MockFileUploader,
        MockChromeDriver,
        MockDisplay,
        mock_requests_post,
    ):
        """Test that when OAuth token retrieval fails, the connection status is updated and webhook is sent"""
        # Configure the mock uploader
        mock_uploader = create_mock_file_uploader()
        MockFileUploader.return_value = mock_uploader

        # Mock the Chrome driver
        mock_driver = create_mock_zoom_web_driver()
        MockChromeDriver.return_value = mock_driver

        # Mock virtual display
        mock_display = MagicMock()
        MockDisplay.return_value = mock_display

        WebhookSubscription.objects.create(
            project=self.project,
            url="https://example.com/webhook",
            triggers=[WebhookTriggerTypes.ZOOM_OAUTH_CONNECTION_STATE_CHANGE],
            is_active=True,
        )

        # Create ZoomOAuthApp
        zoom_oauth_app = ZoomOAuthApp.objects.create(
            project=self.project,
            client_id="test_client_id",
        )
        zoom_oauth_app.set_credentials(
            {
                "client_secret": "test_client_secret",
                "webhook_secret": "test_webhook_secret",
            }
        )

        # Create ZoomOAuthConnection
        zoom_oauth_connection = ZoomOAuthConnection.objects.create(
            zoom_oauth_app=zoom_oauth_app,
            user_id="test_user_id",
            account_id="test_account_id",
            state=ZoomOAuthConnectionStates.CONNECTED,
        )
        zoom_oauth_connection.set_credentials(
            {
                "access_token": "test_access_token",
                "refresh_token": "test_refresh_token",
            }
        )

        # Create mapping for the meeting
        meeting_id = "123123213"
        self.bot.meeting_url = f"https://zoom.us/j/{meeting_id}"
        self.bot.save()

        ZoomMeetingToZoomOAuthConnectionMapping.objects.create(
            zoom_oauth_app=zoom_oauth_app,
            zoom_oauth_connection=zoom_oauth_connection,
            meeting_id=meeting_id,
        )

        # Mock the token refresh response to fail with authentication error
        mock_token_response = MagicMock()
        mock_token_response.status_code = 401
        mock_token_response.json.return_value = {
            "error": "invalid_grant",
            "error_description": "Invalid grant",
        }

        # Create a proper HTTPError with a response
        http_error = requests.HTTPError("401 Unauthorized")
        http_error.response = mock_token_response
        mock_token_response.raise_for_status.side_effect = http_error
        mock_requests_post.return_value = mock_token_response

        # Create bot controller
        controller = BotController(self.bot.id)

        # Run the bot in a separate thread since it has an event loop
        bot_thread = threading.Thread(target=controller.run)
        bot_thread.daemon = True
        bot_thread.start()

        # Give the bot some time to attempt token retrieval
        time.sleep(2)

        # Verify that the ZoomOAuthConnection state was updated to DISCONNECTED
        zoom_oauth_connection.refresh_from_db()
        self.assertEqual(zoom_oauth_connection.state, ZoomOAuthConnectionStates.DISCONNECTED, "ZoomOAuthConnection should be DISCONNECTED after authentication failure")

        # Verify that connection_failure_data was set
        self.assertIsNotNone(zoom_oauth_connection.connection_failure_data)
        self.assertIn("error", zoom_oauth_connection.connection_failure_data)

        # Verify that a webhook was triggered for the connection state change
        webhook_attempts = WebhookDeliveryAttempt.objects.filter(
            webhook_trigger_type=WebhookTriggerTypes.ZOOM_OAUTH_CONNECTION_STATE_CHANGE,
            zoom_oauth_connection=zoom_oauth_connection,
        )
        self.assertTrue(webhook_attempts.exists(), "A webhook should be triggered for ZoomOAuthConnection state change")

        # Verify the webhook payload contains the connection info
        webhook_attempt = webhook_attempts.first()
        self.assertEqual(webhook_attempt.payload["state"], "disconnected")
        self.assertIsNotNone(webhook_attempt.payload["connection_failure_data"])

        # Close the database connection since we're in a thread
        connection.close()

    @patch("bots.zoom_oauth_connections_utils.requests.post")
    @patch("bots.web_bot_adapter.web_bot_adapter.Display")
    @patch("bots.web_bot_adapter.web_bot_adapter.webdriver.Chrome")
    @patch("bots.bot_controller.bot_controller.S3FileUploader")
    def test_zoom_oauth_app_token_failure_with_onbehalf_token(
        self,
        MockFileUploader,
        MockChromeDriver,
        MockDisplay,
        mock_requests_post,
    ):
        """Test that when OAuth token retrieval fails, the connection status is updated and webhook is sent"""
        # Configure the mock uploader
        mock_uploader = create_mock_file_uploader()
        MockFileUploader.return_value = mock_uploader

        # Mock the Chrome driver
        mock_driver = create_mock_zoom_web_driver()
        MockChromeDriver.return_value = mock_driver

        # Mock virtual display
        mock_display = MagicMock()
        MockDisplay.return_value = mock_display

        WebhookSubscription.objects.create(
            project=self.project,
            url="https://example.com/webhook",
            triggers=[WebhookTriggerTypes.ZOOM_OAUTH_CONNECTION_STATE_CHANGE],
            is_active=True,
        )

        # Create ZoomOAuthApp
        zoom_oauth_app = ZoomOAuthApp.objects.create(
            project=self.project,
            client_id="test_client_id",
        )
        zoom_oauth_app.set_credentials(
            {
                "client_secret": "test_client_secret",
                "webhook_secret": "test_webhook_secret",
            }
        )

        # Create ZoomOAuthConnection
        zoom_oauth_connection = ZoomOAuthConnection.objects.create(
            zoom_oauth_app=zoom_oauth_app,
            user_id="test_user_id",
            account_id="test_account_id",
            state=ZoomOAuthConnectionStates.CONNECTED,
            is_onbehalf_token_supported=True,
        )
        zoom_oauth_connection.set_credentials(
            {
                "access_token": "test_access_token",
                "refresh_token": "test_refresh_token",
            }
        )

        # Create mapping for the meeting
        meeting_id = "123123213"
        self.bot.meeting_url = f"https://zoom.us/j/{meeting_id}"
        self.bot.settings = {
            "zoom_settings": {
                "sdk": "web",
                "onbehalf_token": {
                    "zoom_oauth_connection_user_id": "test_user_id",
                },
            },
        }
        self.bot.save()

        # Mock the token refresh response to fail with authentication error
        mock_token_response = MagicMock()
        mock_token_response.status_code = 401
        mock_token_response.json.return_value = {
            "error": "invalid_grant",
            "error_description": "Invalid grant",
        }

        # Create a proper HTTPError with a response
        http_error = requests.HTTPError("401 Unauthorized")
        http_error.response = mock_token_response
        mock_token_response.raise_for_status.side_effect = http_error
        mock_requests_post.return_value = mock_token_response

        # Create bot controller
        controller = BotController(self.bot.id)

        # Run the bot in a separate thread since it has an event loop
        bot_thread = threading.Thread(target=controller.run)
        bot_thread.daemon = True
        bot_thread.start()

        # Give the bot some time to attempt token retrieval
        time.sleep(2)

        # Verify that the ZoomOAuthConnection state was updated to DISCONNECTED
        zoom_oauth_connection.refresh_from_db()
        self.assertEqual(zoom_oauth_connection.state, ZoomOAuthConnectionStates.DISCONNECTED, "ZoomOAuthConnection should be DISCONNECTED after authentication failure")

        # Verify that connection_failure_data was set
        self.assertIsNotNone(zoom_oauth_connection.connection_failure_data)
        self.assertIn("error", zoom_oauth_connection.connection_failure_data)

        # Verify that a webhook was triggered for the connection state change
        webhook_attempts = WebhookDeliveryAttempt.objects.filter(
            webhook_trigger_type=WebhookTriggerTypes.ZOOM_OAUTH_CONNECTION_STATE_CHANGE,
            zoom_oauth_connection=zoom_oauth_connection,
        )
        self.assertTrue(webhook_attempts.exists(), "A webhook should be triggered for ZoomOAuthConnection state change")

        # Verify the webhook payload contains the connection info
        webhook_attempt = webhook_attempts.first()
        self.assertEqual(webhook_attempt.payload["state"], "disconnected")
        self.assertIsNotNone(webhook_attempt.payload["connection_failure_data"])

        # Close the database connection since we're in a thread
        connection.close()

    @patch("bots.zoom_web_bot_adapter.zoom_web_ui_methods.start_zoom_web_static_server", return_value=8080)
    @patch("bots.web_bot_adapter.web_bot_adapter.Display")
    @patch("bots.web_bot_adapter.web_bot_adapter.webdriver.Chrome")
    @patch("bots.bot_controller.bot_controller.S3FileUploader")
    def test_waiting_room_timeout(
        self,
        MockFileUploader,
        MockChromeDriver,
        MockDisplay,
        mock_start_static_server,
    ):
        """Test that bot times out if waiting room timeout is exceeded."""
        # Configure the mock uploader
        mock_uploader = create_mock_file_uploader()
        MockFileUploader.return_value = mock_uploader

        # Mock the Chrome driver
        mock_driver = create_mock_zoom_web_driver()
        MockChromeDriver.return_value = mock_driver

        # Mock virtual display
        mock_display = MagicMock()
        MockDisplay.return_value = mock_display

        self.bot.settings = {
            "zoom_settings": {
                "sdk": "web",
            },
            "automatic_leave_settings": {
                "waiting_room_timeout_seconds": 3,
            },
        }
        self.bot.save()

        # Mock execute_script to handle different script calls
        def execute_script_side_effect(script, *args):
            if "userHasEnteredMeeting" in script:
                return False  # User has NOT entered the meeting
            if "userHasEncounteredOnBehalfTokenUserNotInMeetingError" in script:
                return False  # No onbehalf token error
            return None

        mock_driver.execute_script.side_effect = execute_script_side_effect

        # Mock find_element to not find the "host to start meeting" text (so we're in waiting room, not waiting for host)
        mock_driver.find_element.side_effect = NoSuchElementException("Element not found")

        # Create bot controller
        controller = BotController(self.bot.id)

        # Run the bot in a separate thread since it has an event loop
        bot_thread = threading.Thread(target=controller.run)
        bot_thread.daemon = True
        bot_thread.start()

        # Give the bot time to process
        bot_thread.join(timeout=10)

        # Refresh the bot from the database
        self.bot.refresh_from_db()

        # Assert that the bot is in the FATAL_ERROR state after timeout
        self.assertEqual(self.bot.state, BotStates.FATAL_ERROR)

        # Verify bot events in sequence
        bot_events = self.bot.bot_events.all()

        # Should have at least 2 events: JOIN_REQUESTED and COULD_NOT_JOIN
        self.assertGreaterEqual(len(bot_events), 2)

        # Verify join_requested_event (Event 1)
        join_requested_event = bot_events[0]
        self.assertEqual(join_requested_event.event_type, BotEventTypes.JOIN_REQUESTED)
        self.assertEqual(join_requested_event.old_state, BotStates.READY)
        self.assertEqual(join_requested_event.new_state, BotStates.JOINING)

        # Find the COULD_NOT_JOIN event
        could_not_join_events = [e for e in bot_events if e.event_type == BotEventTypes.COULD_NOT_JOIN]
        self.assertGreaterEqual(len(could_not_join_events), 1)

        # Verify the event has the correct subtype
        could_not_join_event = could_not_join_events[0]
        self.assertEqual(could_not_join_event.event_sub_type, BotEventSubTypes.COULD_NOT_JOIN_MEETING_WAITING_ROOM_TIMEOUT_EXCEEDED)

        # Cleanup
        controller.cleanup()
        bot_thread.join(timeout=5)

        # Close the database connection since we're in a thread
        connection.close()
