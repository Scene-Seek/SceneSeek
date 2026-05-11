from sqlalchemy import func, select
from sqlalchemy.orm import selectinload
from src.core.database import session_factory
from src.models.search_history import SearchHistory
from src.models.search_results import SearchResults
from src.models.users import Users
from src.models.videos import Videos


class DatabaseService:
    """Сервис для работы с БД"""
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
        processing_status: str = "pending",
    ) -> Videos:
        """Создает запись о видео и возвращает объект"""
        async with session_factory() as session:
            video = Videos(
                uploaded_by_user_id=uploaded_by_user_id,
                title=title,
                path=path,
                duration=duration,
                fps=fps,
                resolution=resolution,
                processing_status=processing_status,
            )
            session.add(video)
            await session.commit()
            await session.refresh(video)
            return video

    async def get_video_by_id(self, *, video_id: int) -> Videos | None:
        """Возвращает видео по идентификатору"""
        async with session_factory() as session:
            result = await session.execute(
                select(Videos).where(Videos.video_id == video_id)
            )
            return result.scalar_one_or_none()

    async def get_user_by_id(self, *, user_id: int) -> Users | None:
        """Возвращает пользователя по идентификатору"""
        async with session_factory() as session:
            result = await session.execute(
                select(Users).where(Users.user_id == user_id)
            )
            return result.scalar_one_or_none()

    async def get_user_by_username(self, *, username: str) -> Users | None:
        """Возвращает пользователя по имени"""
        async with session_factory() as session:
            result = await session.execute(
                select(Users).where(Users.username == username)
            )
            return result.scalar_one_or_none()

    # Query
    async def create_query(
        self, *, user_id: int, video_id: int, query: str
    ) -> SearchHistory:
        """Создает запись запроса и возвращает ее"""
        async with session_factory() as session:
            search_entry = SearchHistory(
                user_id=user_id, video_id=video_id, query_text=query
            )
            session.add(search_entry)
            await session.commit()
            await session.refresh(search_entry)
            return search_entry

    async def get_query_by_id(self, *, query_id: int) -> SearchHistory | None:
        """Возвращает запись запроса по идентификатору"""
        async with session_factory() as session:
            result = await session.execute(
                select(SearchHistory).where(SearchHistory.query_id == query_id)
            )
            return result.scalar_one_or_none()

    async def get_query_results_by_id(self, *, query_id: int) -> list[SearchResults]:
        """Возвращает результаты поиска для указанного запроса"""
        async with session_factory() as session:
            result = await session.execute(
                select(SearchResults)
                .where(SearchResults.query_id == query_id)
                .options(selectinload(SearchResults.found_event))
                .order_by(SearchResults.similarity_score.desc())
            )
            return result.scalars().all()

    async def get_video_history_by_user(self, *, user_id: int, limit: int = 30):
        """Возвращает историю видео и последний запрос по каждому видео"""
        async with session_factory() as session:
            latest_search = (
                select(
                    SearchHistory.query_id.label("query_id"),
                    SearchHistory.video_id.label("video_id"),
                    SearchHistory.query_text.label("query_text"),
                    SearchHistory.processing_status.label("processing_status"),
                    SearchHistory.search_date.label("search_date"),
                    func.row_number()
                    .over(
                        partition_by=SearchHistory.video_id,
                        order_by=(
                            SearchHistory.search_date.desc(),
                            SearchHistory.query_id.desc(),
                        ),
                    )
                    .label("rn"),
                )
                .where(SearchHistory.user_id == user_id)
                .subquery()
            )
            result = await session.execute(
                select(
                    Videos,
                    latest_search.c.query_id,
                    latest_search.c.query_text,
                    latest_search.c.processing_status,
                    latest_search.c.search_date,
                )
                .outerjoin(
                    latest_search,
                    (latest_search.c.video_id == Videos.video_id)
                    & (latest_search.c.rn == 1),
                )
                .where(Videos.uploaded_by_user_id == user_id)
                .order_by(
                    func.coalesce(latest_search.c.search_date, Videos.created_at).desc(),
                    Videos.video_id.desc(),
                )
                .limit(limit)
            )
            return result.all()

    # User
    async def create_user(
        self, *, username: str, password_hash: str | None, role: str = "user"
    ) -> Users:
        """Создает пользователя и возвращает объект"""
        async with session_factory() as session:
            user = Users(username=username, password_hash=password_hash, role=role)
            session.add(user)
            await session.commit()
            await session.refresh(user)
            return user


database_service = DatabaseService()
