import unittest
from unittest.mock import MagicMock, patch, call
import threading
import time
import os

from bots.bot_controller.webpage_streamer_manager import WebpageStreamerManager

class TestWebpageStreamerManager(unittest.TestCase):
    def setUp(self):
        self.mock_is_bot_ready = MagicMock(return_value=True)
        self.mock_get_offer = MagicMock(return_value={"sdp": "fake_sdp", "type": "offer"})
        self.mock_start_peer = MagicMock()
        self.mock_play_stream = MagicMock()
        self.mock_stop_stream = MagicMock()
        self.mock_on_ready = MagicMock()
        self.hostname = "test-hostname"
        
        self.manager = WebpageStreamerManager(
            is_bot_ready_for_webpage_streamer_callback=self.mock_is_bot_ready,
            get_peer_connection_offer_callback=self.mock_get_offer,
            start_peer_connection_callback=self.mock_start_peer,
            play_bot_output_media_stream_callback=self.mock_play_stream,
            stop_bot_output_media_stream_callback=self.mock_stop_stream,
            on_message_that_webpage_streamer_connection_can_start_callback=self.mock_on_ready,
            webpage_streamer_service_hostname=self.hostname
        )

    def tearDown(self):
        self.manager.cleanup()
        if self.manager.keepalive_task and self.manager.keepalive_task.is_alive():
            self.manager.keepalive_task.join(timeout=1)

    def test_init_starts_keepalive_thread(self):
        with patch.object(self.manager, 'send_webpage_streamer_keepalive_periodically') as mock_keepalive:
            # We don't want the thread to actually run the loop in this test
            self.manager.keepalive_task = MagicMock() 
            self.manager.init()
            # Since I assigned a mock to keepalive_task before init, init returns early. 
            # Let's reset and test properly.
            self.manager.keepalive_task = None
            
            # We actually want to check if a thread is started targeting the method
            with patch('threading.Thread') as mock_thread:
                self.manager.init()
                mock_thread.assert_called_once_with(target=self.manager.send_webpage_streamer_keepalive_periodically, daemon=True)
                mock_thread.return_value.start.assert_called_once()

    @patch('bots.bot_controller.webpage_streamer_manager.requests.post')
    def test_update_starts_webrtc_connection(self, mock_post):
        # Setup successful responses
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"status": "ok"}
        
        self.manager.webpage_streamer_connection_can_start = True
        url = "http://example.com"
        output_dest = "camera"
        
        self.manager.update(url, output_dest)
        
        # Check offer was requested
        self.mock_get_offer.assert_called_once()
        
        # Check offer was sent to streaming service
        expected_offer_url = f"http://attendee-webpage-streamer-local:8000/offer"
        # Note: hostname defaults to attendee-webpage-streamer-local if env var not set
        
        # Verify calls to requests.post
        # 1. Offer
        # 2. Start streaming
        
        # Filter calls to ignore keepalive if any (though usually init() starts it)
        # Here we didn't call init(), so keepalive shouldn't be running.
        
        calls = mock_post.call_args_list
        self.assertTrue(len(calls) >= 2)
        
        # Verify offer call
        offer_call = [c for c in calls if c.args[0].endswith('/offer')][0]
        self.assertEqual(offer_call.kwargs['json'], {"sdp": "fake_sdp", "type": "offer"})
        
        # Verify start streaming call
        start_call = [c for c in calls if c.args[0].endswith('/start_streaming')][0]
        self.assertEqual(start_call.kwargs['json'], {"url": url})
        
        # Verify callbacks
        self.mock_start_peer.assert_called_once()
        self.mock_play_stream.assert_called_once_with(output_dest)
        
        self.assertTrue(self.manager.webrtc_connection_started)
        self.assertEqual(self.manager.url, url)
        self.assertEqual(self.manager.output_destination, output_dest)

    @patch('bots.bot_controller.webpage_streamer_manager.requests.post')
    def test_update_updates_existing_connection(self, mock_post):
        mock_post.return_value.status_code = 200
        self.manager.webpage_streamer_connection_can_start = True
        self.manager.webrtc_connection_started = True
        self.manager.url = "http://old.com"
        self.manager.output_destination = "camera"
        
        new_url = "http://new.com"
        self.manager.update(new_url, "camera")
        
        # Should call start_streaming (which updates if already started)
        start_calls = [c for c in mock_post.call_args_list if c.args[0].endswith('/start_streaming')]
        self.assertEqual(len(start_calls), 1)
        self.assertEqual(start_calls[0].kwargs['json'], {"url": new_url})
        
        # Should not get new offer
        self.mock_get_offer.assert_not_called()
        
    @patch('bots.bot_controller.webpage_streamer_manager.requests.post')
    def test_update_change_output_destination(self, mock_post):
        mock_post.return_value.status_code = 200
        self.manager.webpage_streamer_connection_can_start = True
        self.manager.webrtc_connection_started = True
        self.manager.url = "http://example.com"
        self.manager.output_destination = "camera"
        self.manager.last_non_empty_url = "http://example.com"
        
        new_dest = "screenshare"
        
        # To avoid sleep in test
        with patch('time.sleep'):
            self.manager.update("http://example.com", new_dest)
            
        self.mock_stop_stream.assert_called_once()
        self.mock_play_stream.assert_called_once_with(new_dest)
        self.assertEqual(self.manager.output_destination, new_dest)

    @patch('bots.bot_controller.webpage_streamer_manager.requests.post')
    def test_update_stop_streaming_if_no_url(self, mock_post):
        self.manager.webpage_streamer_connection_can_start = True
        self.manager.webrtc_connection_started = True
        self.manager.url = "http://example.com"
        self.manager.output_destination = "camera"
        
        self.manager.update(None, "camera")
        
        self.mock_stop_stream.assert_called_once()
        self.assertIsNone(self.manager.url)

    @patch('bots.bot_controller.webpage_streamer_manager.requests.post')
    def test_keepalive_loop(self, mock_post):
        mock_post.return_value.status_code = 200
        
        # Mock time.sleep to raise an exception after a few iterations or just run once then stop
        # Better: run the loop in a separate thread (as intended) but control it via cleaned_up
        
        # We want to test that it calls keepalive and eventually sets webpage_streamer_connection_can_start
        
        # Using a controlled execution
        # Let's override send_webpage_streamer_keepalive_periodically to run once logic logic 
        # but that is testing the python loop.
        # Instead, let's call the logic inside the loop manually or adjust sleep.
        
        # Simulating one iteration of the loop where connection is not yet allowed
        self.manager.webpage_streamer_connection_can_start = False
        
        with patch('time.sleep'): # skip sleep
            # We need to break the loop. Let's make cleaned_up True after one call to requests.post?
            # Or mock requests.post to set cleaned_up = True as side effect?
            def side_effect(*args, **kwargs):
                self.manager.cleaned_up = True
                return MagicMock(status_code=200)
            mock_post.side_effect = side_effect
            
            self.manager.send_webpage_streamer_keepalive_periodically()
            
        mock_post.assert_called_with(f"http://attendee-webpage-streamer-local:8000/keepalive", json={})
        
        # Check callbacks
        self.mock_is_bot_ready.assert_called()
        self.mock_on_ready.assert_called()
        self.assertTrue(self.manager.webpage_streamer_connection_can_start)

    @patch('bots.bot_controller.webpage_streamer_manager.requests.post')
    def test_cleanup(self, mock_post):
        self.manager.cleanup()
        self.assertTrue(self.manager.cleaned_up)
        mock_post.assert_called_with(f"http://attendee-webpage-streamer-local:8000/shutdown", json={})

