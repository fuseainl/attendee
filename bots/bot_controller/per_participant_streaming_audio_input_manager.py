import logging
import queue
import time

import numpy as np
import webrtcvad

from bots.models import (
    Credentials,
    Participant,
    TranscriptionProviders,
    Utterance,
    WebhookTriggerTypes,
)
from bots.transcription_providers.deepgram.deepgram_streaming_transcriber import (  # noqa: E501
    DeepgramStreamingTranscriber,
)
from bots.transcription_providers.kyutai.kyutai_streaming_transcriber import (  # noqa: E501
    KyutaiStreamingTranscriber,
)
from bots.webhook_payloads import utterance_webhook_payload
from bots.webhook_utils import trigger_webhook

logger = logging.getLogger(__name__)


def calculate_normalized_rms(audio_bytes):
    samples = np.frombuffer(audio_bytes, dtype=np.int16)
    rms = np.sqrt(np.mean(np.square(samples)))
    # Normalize by max possible value for 16-bit audio (32768)
    return rms / 32768


class PerParticipantStreamingAudioInputManager:
    def __init__(self, *, get_participant_callback, sample_rate, transcription_provider, bot):
        self.queue = queue.Queue()

        self.get_participant_callback = get_participant_callback

        self.utterances = {}
        self.sample_rate = sample_rate

        self.last_nonsilent_audio_time = {}

        self.SILENCE_DURATION_LIMIT = 10  # seconds

        self.vad = webrtcvad.Vad()
        self.transcription_provider = transcription_provider
        self.streaming_transcribers = {}
        self.last_nonsilent_audio_time = {}

        self.project = bot.project
        self.bot = bot
        self.deepgram_api_key = self.get_deepgram_api_key()
        self.kyutai_server_url, self.kyutai_api_key = self.get_kyutai_server_url()

    def silence_detected(self, chunk_bytes):
        if calculate_normalized_rms(chunk_bytes) < 0.0025:
            return True
        return not self.vad.is_speech(chunk_bytes, self.sample_rate)

    def get_deepgram_api_key(self):
        deepgram_credentials_record = self.project.credentials.filter(credential_type=Credentials.CredentialTypes.DEEPGRAM).first()
        if not deepgram_credentials_record:
            return None

        deepgram_credentials = deepgram_credentials_record.get_credentials()
        return deepgram_credentials["api_key"]

    def get_kyutai_server_url(self):
        # First try to get from transcription settings (preferred)
        server_url = self.bot.transcription_settings.kyutai_server_url()
        api_key = self.bot.transcription_settings.kyutai_api_key()

        # Fall back to project credentials if not in settings
        if not server_url:
            kyutai_credentials_record = self.project.credentials.filter(credential_type=Credentials.CredentialTypes.KYUTAI).first()
            if not kyutai_credentials_record:
                return None, None

            kyutai_credentials = kyutai_credentials_record.get_credentials()
            server_url = kyutai_credentials.get("server_url", "ws://127.0.0.1:8012/api/asr-streaming")
            api_key = kyutai_credentials.get("api_key", None)

        return server_url, api_key

    def _create_utterance_callback(self, speaker_id, metadata):
        """
        Create a callback function that saves utterances and triggers webhooks.
        This is called by Kyutai when an utterance boundary is detected.

        Emits webhooks quickly for real-time updates. Grouping happens
        in the UI via generate_aggregated_utterances().
        """

        def callback(transcript_text):
            try:
                # Get participant info
                participant_data = self.get_participant_callback(speaker_id)
                if not participant_data:
                    logger.warning(f"Could not get participant data for speaker " f"{speaker_id}")
                    return

                participant, _ = Participant.objects.get_or_create(
                    bot=self.bot,
                    uuid=participant_data.get("participant_uuid"),
                    defaults={
                        "user_uuid": participant_data.get("participant_user_uuid"),
                        "full_name": participant_data.get("participant_full_name"),
                        "is_the_bot": participant_data.get("participant_is_the_bot", False),
                        "is_host": participant_data.get("participant_is_host", False),
                    },
                )

                # Get current recording
                from bots.models import Recording, RecordingStates

                recording = Recording.objects.filter(
                    bot=self.bot,
                    state=RecordingStates.IN_PROGRESS,
                ).first()

                if not recording:
                    logger.warning(f"No recording in progress for bot " f"{self.bot.object_id}")
                    return

                # Create unique source UUID for this utterance
                import uuid

                source_uuid = f"{recording.object_id}-{uuid.uuid4()}"

                # Create utterance
                utterance, _ = Utterance.objects.update_or_create(
                    recording=recording,
                    source_uuid=source_uuid,
                    defaults={
                        "source": Utterance.Sources.PER_PARTICIPANT_AUDIO,
                        "participant": participant,
                        "transcription": {"transcript": transcript_text},
                        "timestamp_ms": int(time.time() * 1000),
                        # Duration calculated later if needed
                        "duration_ms": 0,
                        "sample_rate": self.sample_rate,
                    },
                )

                logger.info(f"✅ Created utterance for {participant.full_name}: " f"{transcript_text}")

                # Trigger webhook
                trigger_webhook(
                    webhook_trigger_type=WebhookTriggerTypes.TRANSCRIPT_UPDATE,
                    bot=self.bot,
                    payload=utterance_webhook_payload(utterance),
                )

                logger.info("✅ Triggered transcript.update webhook")

            except Exception as e:
                logger.error(f"Error in utterance callback: {e}")
                import traceback

                logger.error(traceback.format_exc())

        return callback

    def create_streaming_transcriber(self, speaker_id, metadata):
        logger.info(f"Creating streaming transcriber for speaker {speaker_id}, " f"provider: {self.transcription_provider}")
        if self.transcription_provider == TranscriptionProviders.DEEPGRAM:
            metadata_list = [f"{key}:{value}" for key, value in metadata.items()] if metadata else None
            return DeepgramStreamingTranscriber(
                deepgram_api_key=self.deepgram_api_key,
                interim_results=True,
                language=self.bot.transcription_settings.deepgram_language(),
                model=self.bot.transcription_settings.deepgram_model(),
                callback=self.bot.transcription_settings.deepgram_callback(),
                sample_rate=self.sample_rate,
                metadata=metadata_list,
                redaction_settings=(self.bot.transcription_settings.deepgram_redaction_settings()),
            )
        elif self.transcription_provider == TranscriptionProviders.KYUTAI:
            logger.info(f"Creating Kyutai transcriber: " f"server={self.kyutai_server_url}, " f"api_key={'***' if self.kyutai_api_key else 'None'}")
            # Create callback for utterance emission
            callback = self._create_utterance_callback(speaker_id, metadata)

            return KyutaiStreamingTranscriber(
                server_url=self.kyutai_server_url,
                sample_rate=self.sample_rate,
                metadata=metadata,
                interim_results=True,
                model=self.bot.transcription_settings.kyutai_model(),
                api_key=self.kyutai_api_key,
                callback=callback,
            )
        else:
            raise Exception(f"Unsupported transcription provider: " f"{self.transcription_provider}")

    def find_or_create_streaming_transcriber_for_speaker(self, speaker_id):
        if speaker_id not in self.streaming_transcribers:
            try:
                metadata = {"bot_id": self.bot.object_id, **(self.bot.metadata or {}), **self.get_participant_callback(speaker_id)}
                logger.info(f"Creating new transcriber for speaker {speaker_id}")
                transcriber = self.create_streaming_transcriber(speaker_id, metadata)

                # Only add if successfully created and connected
                if transcriber:
                    self.streaming_transcribers[speaker_id] = transcriber
                    logger.info(f"✅ Transcriber created for speaker {speaker_id}")
                else:
                    logger.error(f"❌ Failed to create transcriber for " f"speaker {speaker_id}")
                    return None
            except Exception as e:
                logger.error(f"❌ Exception creating transcriber for " f"speaker {speaker_id}: {e}")
                return None
        return self.streaming_transcribers.get(speaker_id)

    def add_chunk(self, speaker_id, chunk_time, chunk_bytes):
        # Check if we have credentials for the transcription provider
        if self.transcription_provider == TranscriptionProviders.DEEPGRAM:
            if not self.deepgram_api_key:
                logger.warning("No Deepgram API key available")
                return
        elif self.transcription_provider == TranscriptionProviders.KYUTAI:
            if not self.kyutai_server_url:
                logger.warning("No Kyutai server URL available")
                return

        audio_is_silent = self.silence_detected(chunk_bytes)

        if not audio_is_silent:
            self.last_nonsilent_audio_time[speaker_id] = time.time()

        if audio_is_silent and speaker_id not in self.streaming_transcribers:
            return

        streaming_transcriber = self.find_or_create_streaming_transcriber_for_speaker(speaker_id)

        # Only send audio if transcriber was successfully created
        if streaming_transcriber:
            streaming_transcriber.send(chunk_bytes)

    def monitor_transcription(self):
        speakers_to_remove = []
        streaming_transcriber_keys = list(self.streaming_transcribers.keys())
        for speaker_id in streaming_transcriber_keys:
            streaming_transcriber = self.streaming_transcribers[speaker_id]
            silence_limit = self.SILENCE_DURATION_LIMIT
            time_since_audio = time.time() - self.last_nonsilent_audio_time[speaker_id]
            if time_since_audio > silence_limit:
                streaming_transcriber.finish()
                speakers_to_remove.append(speaker_id)
                logger.info(f"Speaker {speaker_id} has been silent for too long, " f"stopping streaming transcriber")

        for speaker_id in speakers_to_remove:
            del self.streaming_transcribers[speaker_id]

        # If Number of streaming transcribers is greater than 4,
        # stop the oldest one
        if len(self.streaming_transcribers) > 4:
            oldest_transcriber = min(self.streaming_transcribers.values(), key=lambda x: x.last_send_time)
            oldest_transcriber.finish()
            del self.streaming_transcribers[oldest_transcriber.speaker_id]
            logger.info(f"Stopped oldest streaming transcriber for speaker " f"{oldest_transcriber.speaker_id}")
