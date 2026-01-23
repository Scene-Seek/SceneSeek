from sqlalchemy.orm import mapped_column, Mapped
from sqlalchemy.types import Integer

from src.models.base import Base

class SearchHistory(Base):
    __tablename__ = "search_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)