from sqlalchemy import BigInteger, Boolean, Float, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base


class SearchResults(Base):
    __tablename__ = "search_results"

    result_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    query_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("search_history.query_id", ondelete="CASCADE"),
        nullable=False
    )
    found_event_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("video_events.event_id", ondelete="CASCADE"),
        nullable=False
    )
    similarity_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_relevant: Mapped[bool | None] = mapped_column(
        Boolean,
        nullable=True,
        default=None
    )

    query = relationship(
        "SearchHistory",
        back_populates="results"
    )
    found_event = relationship(
        "VideoEvents",
        back_populates="search_results"
    )