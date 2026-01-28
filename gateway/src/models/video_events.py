from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Float, ForeignKey, Index, Text, func, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base
from src.models.types import Vector


class VideoEvents(Base):
    __tablename__ = "video_events"
    __table_args__ = (
        Index(
            "idx_events_embedding",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"}
        ),
        Index(
            "idx_events_yolo",
            "yolo_metadata",
            postgresql_using="gin"
        ),
        Index(
            "idx_events_video_id",
            "video_id",
            "timestamp"
        ),
    )

    event_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    video_id: Mapped[int] = mapped_column(
        ForeignKey("videos.video_id", ondelete="CASCADE"),
        nullable=False
    )
    timestamp: Mapped[float] = mapped_column(Float, nullable=False)
    caption: Mapped[str] = mapped_column(Text, nullable=False)
    yolo_metadata: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb")
    )
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(384),
        nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=func.now()
    )

    video = relationship(
        "Videos",
        back_populates="events"
    )
    search_results = relationship(
        "SearchResults",
        back_populates="found_event",
        passive_deletes=True
    )