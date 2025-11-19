import logging

from django.core.exceptions import ValidationError
from django.db import IntegrityError, transaction

from .bots_api_utils import create_webhook_subscriptions
from .models import (
    Bot,
    BotCreationSource,
    BotEventManager,
    BotEventTypes,
    BotStates,
    Project,
    Recording,
    SessionTypes,
    TranscriptionTypes,
)
from .serializers import (
    CreateAppSessionSerializer,
)
from .utils import transcription_provider_from_bot_creation_data

logger = logging.getLogger(__name__)


def create_app_session(data: dict, source: BotCreationSource, project: Project) -> tuple[Bot | None, dict | None]:
    # Given them a small grace period before we start rejecting requests
    if project.organization.out_of_credits():
        logger.error(f"Organization {project.organization.id} has insufficient credits. Please add credits in the Account -> Billing page.")
        return None, {"error": "Organization has run out of credits. Please add more credits in the Account -> Billing page."}

    serializer = CreateAppSessionSerializer(data=data)
    if not serializer.is_valid():
        return None, serializer.errors

    transcription_settings = serializer.validated_data["transcription_settings"]
    rtmp_settings = serializer.validated_data["rtmp_settings"]
    recording_settings = serializer.validated_data["recording_settings"]
    debug_settings = serializer.validated_data["debug_settings"]

    metadata = serializer.validated_data["metadata"]
    websocket_settings = serializer.validated_data["websocket_settings"]
    deduplication_key = serializer.validated_data["deduplication_key"]
    webhook_subscriptions = serializer.validated_data["webhooks"]
    zoom_rtms = serializer.validated_data["zoom_rtms"]
    initial_state = BotStates.READY

    settings = {
        "transcription_settings": transcription_settings,
        "rtmp_settings": rtmp_settings,
        "recording_settings": recording_settings,
        "debug_settings": debug_settings,
        "websocket_settings": websocket_settings,
        "zoom_rtms": zoom_rtms,
    }

    try:
        with transaction.atomic():
            app_session = Bot.objects.create(
                project=project,
                settings=settings,
                metadata=metadata,
                deduplication_key=deduplication_key,
                state=initial_state,
                zoom_rtms_stream_id=zoom_rtms.get("rtms_stream_id"),
                meeting_url="app_session",
                name="App Session",
                session_type=SessionTypes.APP_SESSION,
            )

            Recording.objects.create(
                bot=app_session,
                recording_type=app_session.recording_type(),
                transcription_type=TranscriptionTypes.NON_REALTIME,
                transcription_provider=transcription_provider_from_bot_creation_data(serializer.validated_data),
                is_default_recording=True,
            )

            # Create bot-level webhook subscriptions if provided
            if webhook_subscriptions:
                create_webhook_subscriptions(webhook_subscriptions, project, app_session)

            BotEventManager.create_event(bot=app_session, event_type=BotEventTypes.APP_SESSION_CONNECTION_REQUESTED, event_metadata={"source": source})

            return app_session, None

    except ValidationError as e:
        logger.error(f"ValidationError creating app session: {e}")
        return None, {"error": e.messages[0]}
    except Exception as e:
        if isinstance(e, IntegrityError) and "unique_bot_deduplication_key" in str(e):
            logger.error(f"IntegrityError due to unique_bot_deduplication_key constraint violation creating app session: {e}")
            return None, {"error": "Deduplication key already in use. A app session in a non-terminal state with this deduplication key already exists. Please use a different deduplication key or wait for that app session to terminate."}

        logger.error(f"Error creating app session: {e}")
        return None, {"error": str(e)}
