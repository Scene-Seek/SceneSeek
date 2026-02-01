from sqlalchemy import select

from src.models.search_results import SearchResults
from src.core.database import session_factory
from src.models.videos import Videos
from src.models.search_history import SearchHistory
from src.models.users import Users

class DatabaseService:
    def __init__(self) -> None:
        pass

    # Video
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

    async def get_user_by_id(
        self,
        *,
        user_id: int
    ) -> Users | None:
        async with session_factory() as session:
            result = await session.execute(
                select(Users).where(Users.user_id == user_id)
            )
            return result.scalar_one_or_none()

    # Query
    async def create_query(
        self,
        *,
        user_id: int,
        video_id: int,
        query: str
    ) -> SearchHistory:
        async with session_factory() as session:
            query = SearchHistory(
                user_id=user_id,
                video_id=video_id,
                query_text=query
            )
            session.add(query)
            await session.commit()
            await session.refresh(query)
            return query

    async def get_query_by_id(
        self,
        *,
        query_id: int
    ) -> SearchHistory | None:
        async with session_factory() as session:
            result = await session.execute(
                select(SearchHistory).where(SearchHistory.query_id == query_id)
            )
            return result.scalar_one_or_none()

    async def get_query_results_by_id(
        self,
        *,
        query_id: int
    ) -> list[SearchResults]:
        async with session_factory() as session:
            result = await session.execute(
                select(SearchResults).where(SearchResults.query_id == query_id)
            )
            return result.scalars().all()
    
    # User
    async def find_or_create_user(
        self,
        *,
        username: str,
        role: str = "user"
    ) -> Users:
        async with session_factory() as session:
            result = await session.execute(
                select(Users).where(Users.username == username)
            )
            user = result.scalar_one_or_none()
            if not user:
                user = Users(username=username, role=role)
                session.add(user)
                await session.commit()
                await session.refresh(user)
                return user
            else:
                return user
                

database_service = DatabaseService()
