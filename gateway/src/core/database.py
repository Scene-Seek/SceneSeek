from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from src.models.base import Base

from src.core.config import settings

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=True
)

session_factory = async_sessionmaker(bind=engine)

async def create_tables():
    """
    Функция создания таблиц БД
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        await conn.run_sync(Base.metadata.create_all)