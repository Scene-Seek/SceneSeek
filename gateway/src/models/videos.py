from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base


class Videos(Base):
    __tablename__ = "videos"

    video_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    uploaded_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.user_id", ondelete="SET NULL"),
        nullable=True
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    path: Mapped[str] = mapped_column(String(512), nullable=False)
    duration: Mapped[float | None] = mapped_column(Float, nullable=True)
    fps: Mapped[float | None] = mapped_column(Float, nullable=True)
    resolution: Mapped[str | None] = mapped_column(String(20), nullable=True)
    processing_status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        server_default="pending"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=func.now()
    )

    uploaded_by = relationship(
        "Users",
        back_populates="videos"
    )
    events = relationship(
        "VideoEvents",
        back_populates="video",
        passive_deletes=True
    )