from sqlalchemy.orm import mapped_column, Mapped
from sqlalchemy.types import Integer

from src.models.base import Base

class VideoEvents(Base):
    __tablename__ = "video_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
