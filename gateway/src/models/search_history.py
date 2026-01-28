from datetime import datetime

from sqlalchemy import BigInteger, DateTime, ForeignKey, Text, func, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base
from src.models.types import Vector


class SearchHistory(Base):
    __tablename__ = "search_history"

    query_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False
    )
    video_id: Mapped[int] = mapped_column(
        ForeignKey("videos.video_id", ondelete="CASCADE"),
        nullable=False
    )
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    query_embedding: Mapped[list[float] | None] = mapped_column(
        Vector(384),
        nullable=True
    )
    search_date: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=func.now()
    )
    ### START: processing_status
    processing_status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        server_default="pending"
    )
    ### END

    user = relationship(
        "Users",
        back_populates="search_history"
    )
    video = relationship("Videos")
    results = relationship(
        "SearchResults",
        back_populates="query",
        passive_deletes=True
    )