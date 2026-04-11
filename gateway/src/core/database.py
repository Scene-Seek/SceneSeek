"""db"""

from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from src.core.config import settings
from src.models.base import Base

engine = create_async_engine(settings.DATABASE_URL, echo=False)

session_factory = async_sessionmaker(bind=engine)


async def create_tables():
    """
    Функция создания таблиц БД
    """
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        await conn.run_sync(Base.metadata.create_all)

        # Keep startup resilient on both clean and previously initialized databases.
        await conn.execute(text("ALTER TABLE IF EXISTS video_events ALTER COLUMN embedding TYPE vector(768);"))
        await conn.execute(text("ALTER TABLE IF EXISTS search_history ALTER COLUMN query_embedding TYPE vector(768);"))
        await conn.execute(text("ALTER TABLE IF EXISTS video_events ALTER COLUMN caption DROP NOT NULL;"))

        await conn.execute(text("ALTER TABLE IF EXISTS search_results ADD COLUMN IF NOT EXISTS segment_start DOUBLE PRECISION;"))
        await conn.execute(text("ALTER TABLE IF EXISTS search_results ADD COLUMN IF NOT EXISTS segment_end DOUBLE PRECISION;"))
        await conn.execute(text("ALTER TABLE IF EXISTS search_results ADD COLUMN IF NOT EXISTS best_ts DOUBLE PRECISION;"))
        await conn.execute(text("ALTER TABLE IF EXISTS search_results ADD COLUMN IF NOT EXISTS bbox JSONB DEFAULT '[]'::jsonb;"))
        await conn.execute(text("ALTER TABLE IF EXISTS search_results ADD COLUMN IF NOT EXISTS hit_type VARCHAR(20);"))
