"""db"""

import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from src.core.config import settings
from src.models.base import Base

engine = create_async_engine(settings.DATABASE_URL, echo=False)

session_factory = async_sessionmaker(bind=engine)

logger = logging.getLogger(__name__)

VECTOR_COLUMN_TYPES = {
    ("video_events", "embedding"): "vector(768)",
    ("search_history", "query_embedding"): "vector(768)",
}


async def _get_column_type(*, table_name: str, column_name: str) -> str | None:
    sql = text(
        """
        SELECT format_type(a.atttypid, a.atttypmod)
        FROM pg_attribute a
        JOIN pg_class c ON c.oid = a.attrelid
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = current_schema()
          AND c.relname = :table_name
          AND a.attname = :column_name
          AND a.attnum > 0
          AND NOT a.attisdropped
        """
    )
    async with engine.begin() as conn:
        result = await conn.execute(sql, {"table_name": table_name, "column_name": column_name})
        return result.scalar_one_or_none()


async def _schema_requires_reset() -> tuple[bool, list[str]]:
    reasons: list[str] = []
    for (table_name, column_name), expected_type in VECTOR_COLUMN_TYPES.items():
        current_type = await _get_column_type(table_name=table_name, column_name=column_name)
        if current_type is None:
            continue
        if current_type != expected_type:
            reasons.append(f"{table_name}.{column_name}: {current_type} -> {expected_type}")
    return bool(reasons), reasons


async def _reset_schema(*, reasons: list[str]) -> None:
    logger.warning("Resetting database schema because of incompatible column types: %s", "; ".join(reasons))
    async with engine.begin() as conn:
        result = await conn.execute(
            text(
                """
                SELECT tablename
                FROM pg_tables
                WHERE schemaname = current_schema()
                """
            )
        )
        table_names = [row[0] for row in result.fetchall()]
        for table_name in table_names:
            await conn.execute(text(f'DROP TABLE IF EXISTS "{table_name}" CASCADE;'))
        await conn.run_sync(Base.metadata.create_all)
    logger.warning("Database schema reset complete")


async def _run_non_destructive_alters() -> None:
    # Run potentially-failing ALTER statements in separate transactions so a
    # failure in one does not abort the whole startup sequence.
    alter_statements = [
        ("ALTER TABLE IF EXISTS video_events ALTER COLUMN embedding TYPE vector(768);", "video_events.embedding"),
        ("ALTER TABLE IF EXISTS search_history ALTER COLUMN query_embedding TYPE vector(768);", "search_history.query_embedding"),
        ("ALTER TABLE IF EXISTS video_events ALTER COLUMN caption DROP NOT NULL;", "video_events.caption"),
        ("ALTER TABLE IF EXISTS search_results ADD COLUMN IF NOT EXISTS segment_start DOUBLE PRECISION;", "search_results.segment_start"),
        ("ALTER TABLE IF EXISTS search_results ADD COLUMN IF NOT EXISTS segment_end DOUBLE PRECISION;", "search_results.segment_end"),
        ("ALTER TABLE IF EXISTS search_results ADD COLUMN IF NOT EXISTS best_ts DOUBLE PRECISION;", "search_results.best_ts"),
        ("ALTER TABLE IF EXISTS search_results ADD COLUMN IF NOT EXISTS bbox JSONB DEFAULT '[]'::jsonb;", "search_results.bbox"),
        ("ALTER TABLE IF EXISTS search_results ADD COLUMN IF NOT EXISTS hit_type VARCHAR(20);", "search_results.hit_type"),
        ("ALTER TABLE IF EXISTS users ADD COLUMN IF NOT EXISTS password_hash VARCHAR(255);", "users.password_hash"),
        ("CREATE UNIQUE INDEX IF NOT EXISTS ix_users_username ON users(username);", "users.username"),
    ]

    for sql, desc in alter_statements:
        try:
            async with engine.begin() as conn:
                await conn.execute(text(sql))
        except Exception as e:
            logger.warning("Could not run alter for %s: %s", desc, e)


async def create_tables():
    """
    Функция создания таблиц БД
    """
    # Create extension and tables in a single transaction
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        await conn.run_sync(Base.metadata.create_all)

    requires_reset, reasons = await _schema_requires_reset()
    if requires_reset:
        await _reset_schema(reasons=reasons)

    await _run_non_destructive_alters()
