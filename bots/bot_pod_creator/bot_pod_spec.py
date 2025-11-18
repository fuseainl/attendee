from enum import Enum


class BotPodSpecType(str, Enum):
    DEFAULT = "default"
    SCHEDULED = "scheduled"
