from datetime import datetime

from sqlalchemy import BigInteger, DateTime, ForeignKey, Text, func
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

    user = relationship(
        "Users",
        back_populates="search_history"
    )
    results = relationship(
        "SearchResults",
        back_populates="query",
        passive_deletes=True
    )