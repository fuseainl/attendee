import logging

from rest_framework import serializers

from .serializers import BotSerializer

logger = logging.getLogger(__name__)

import jsonschema
from drf_spectacular.utils import (
    extend_schema_field,
)


@extend_schema_field(
    {
        "type": "object",
        "properties": {
            "meeting_uuid": {
                "type": "string",
                "description": "The UUID of the Zoom meeting",
            },
            "rtms_stream_id": {
                "type": "string",
                "description": "The RTMS stream ID for the Zoom meeting",
            },
            "server_urls": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of server URLs for the RTMS connection",
            },
        },
        "required": ["meeting_uuid", "rtms_stream_id", "server_urls"],
        "additionalProperties": False,
    }
)
class ZoomRTMSJSONField(serializers.JSONField):
    pass


class CreateAppSessionSerializer(BotSerializer):
    zoom_rtms = ZoomRTMSJSONField(help_text="Zoom RTMS configuration containing meeting UUID, stream ID, and server URLs", required=True)

    ZOOM_RTMS_SCHEMA = {
        "type": "object",
        "properties": {
            "meeting_uuid": {"type": "string"},
            "rtms_stream_id": {"type": "string"},
            "server_urls": {"type": "string"},
            "operator_id": {"type": "string"},
        },
        "required": ["meeting_uuid", "rtms_stream_id", "server_urls"],
        "additionalProperties": False,
    }

    def validate_zoom_rtms(self, value):
        if value is None:
            raise serializers.ValidationError("zoom_rtms is required")

        try:
            jsonschema.validate(instance=value, schema=self.ZOOM_RTMS_SCHEMA)
        except jsonschema.exceptions.ValidationError as e:
            raise serializers.ValidationError(e.message)

        return value


class AppSessionSerializer(BotSerializer):
    class Meta(BotSerializer.Meta):
        fields = [field for field in BotSerializer.Meta.fields if field not in ["name", "meeting_url"]] + ["zoom_rtms_stream_id"]
