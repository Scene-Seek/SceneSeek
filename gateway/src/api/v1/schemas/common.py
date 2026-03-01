"""Helper Classes"""

from enum import Enum


class StatusEnum(str, Enum):
    pending = "pending"
    indexing = "indexing"
    ready = "ready"
    completed = "completed"
    not_found = "not_found"
    failed = "failed"
