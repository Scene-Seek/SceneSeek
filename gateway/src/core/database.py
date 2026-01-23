from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from src.models.base import Base

from src.core.config import settings

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=True
)

session_factory = async_sessionmaker(engine=engine)

async def create_tables():
    """
    Функция создания таблиц БД
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)