from sqlalchemy import select

from src.core.database import session_factory
from src.models.videos import Videos


class DatabaseService:
    def __init__(self) -> None:
        pass

    async def create_video(
        self,
        *,
        uploaded_by_user_id: int | None,
        title: str,
        path: str,
        duration: float | None = None,
        fps: float | None = None,
        resolution: str | None = None,
        processing_status: str = "pending"
    ) -> Videos:
        async with session_factory() as session:
            video = Videos(
                uploaded_by_user_id=uploaded_by_user_id,
                title=title,
                path=path,
                duration=duration,
                fps=fps,
                resolution=resolution,
                processing_status=processing_status
            )
            session.add(video)
            await session.commit()
            await session.refresh(video)
            return video

    async def get_video_by_id(
        self,
        *,
        video_id: int
    ) -> Videos | None:
        async with session_factory() as session:
            result = await session.execute(
                select(Videos).where(Videos.video_id == video_id)
            )
            return result.scalar_one_or_none()

database_service = DatabaseService()