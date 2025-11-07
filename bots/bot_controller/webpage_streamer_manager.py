import logging
import os
import threading
import time

import requests

logger = logging.getLogger(__name__)


class WebpageStreamerManager:
    def __init__(
        self,
        url,
        output_destination,
        get_peer_connection_offer_callback,
        start_peer_connection_callback,
        play_bot_output_media_stream_callback,
        webpage_streamer_service_hostname,
    ):
        self.url = url
        self.output_destination = output_destination
        self.get_peer_connection_offer_callback = get_peer_connection_offer_callback
        self.start_peer_connection_callback = start_peer_connection_callback
        self.cleaned_up = False
        self.webpage_streamer_keepalive_task = None
        self.webpage_streamer_service_hostname = webpage_streamer_service_hostname
        self.play_bot_output_media_stream_callback = play_bot_output_media_stream_callback

    def cleanup(self):
        try:
            self.send_webpage_streamer_shutdown_request()
        except Exception as e:
            logger.info(f"Error sending webpage streamer shutdown request: {e}")
        self.cleaned_up = True

    def streaming_service_hostname(self):
        # If we're running in k8s, the streaming service will be on another pod which is addressable using via a per-pod service
        if os.getenv("LAUNCH_BOT_METHOD") == "kubernetes":
            return f"{self.webpage_streamer_service_hostname}"
        # Otherwise the streaming service will be running in a separate docker compose service, so we address it using the service name
        return "attendee-webpage-streamer-local"

    def start(self):
        logger.info(f"Open webpage streaming connection. Settings are url={self.url} and output_destination={self.output_destination}")
        peerConnectionOffer = self.get_peer_connection_offer_callback()
        logger.info(f"Peer connection offer: {peerConnectionOffer}")
        if peerConnectionOffer.get("error"):
            logger.error(f"Error getting peer connection offer: {peerConnectionOffer.get('error')}, returning")
            return

        offer_response = requests.post(f"http://{self.streaming_service_hostname()}:8000/offer", json={"sdp": peerConnectionOffer["sdp"], "type": peerConnectionOffer["type"]})
        logger.info(f"Offer response: {offer_response.json()}")
        self.start_peer_connection_callback(offer_response.json())

        start_streaming_response = requests.post(f"http://{self.streaming_service_hostname()}:8000/start_streaming", json={"url": self.url})
        logger.info(f"Start streaming response: {start_streaming_response}")

        if start_streaming_response.status_code != 200:
            logger.info(f"Failed to start streaming, not starting webpage streamer keepalive task. Response: {start_streaming_response.status_code}")
            return

        # Start the keepalive task after successful streaming start
        if self.webpage_streamer_keepalive_task is None or not self.webpage_streamer_keepalive_task.is_alive():
            self.webpage_streamer_keepalive_task = threading.Thread(target=self.send_webpage_streamer_keepalive_periodically, daemon=True)
            self.webpage_streamer_keepalive_task.start()

        # Tell the adapter to start rendering the bot output media stream in the webcam / screenshare
        self.play_bot_output_media_stream_callback(self.output_destination)

    def send_webpage_streamer_keepalive_periodically(self):
        """Send keepalive requests to the streaming service every 60 seconds."""
        while not self.cleaned_up:
            try:
                time.sleep(60)  # Wait 60 seconds between keepalive requests

                if self.cleaned_up:
                    break

                response = requests.post(f"http://{self.streaming_service_hostname()}:8000/keepalive", json={})
                logger.info(f"Webpage streamer keepalive response: {response.status_code}")

            except Exception as e:
                logger.info(f"Failed to send webpage streamer keepalive: {e}")
                # Continue the loop even if a single keepalive fails

        logger.info("Webpage streamer keepalive task stopped")

    def send_webpage_streamer_shutdown_request(self):
        try:
            response = requests.post(f"http://{self.streaming_service_hostname()}:8000/shutdown", json={})
            logger.info(f"Webpage streamer shutdown response: {response.json()}")
        except Exception as e:
            logger.info(f"Webpage streamer shutdown response: {e}")
