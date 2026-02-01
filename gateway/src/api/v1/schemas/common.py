"""Helper Classes"""

from enum import Enum

class StatusEnum(str, Enum):
    pending = "pending"
    indexing = "indexing"
    ready = "ready"
    failed = "failed"