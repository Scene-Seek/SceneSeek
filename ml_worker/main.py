import asyncpg
from faststream import FastStream
from pgvector.asyncpg import register_vector

from broker import broker, set_engine
from config import settings
from engine import VideoSearchEngine, IndexerConfig

app = FastStream(broker)
@app.on_startup
async def init_app() -> None:
    conf = IndexerConfig(
        db_dsn=settings.DATABASE_URL.replace("postgresql+asyncpg", "postgresql"),
        frame_skip=15
    )
    engine = VideoSearchEngine(config=conf)
    engine.pool = await asyncpg.create_pool(dsn=conf.db_dsn, init=register_vector)
    set_engine(engine)


@app.on_shutdown
async def close_app() -> None:
    from broker import get_engine

    engine = get_engine()
    await engine.close()


if __name__ == "__main__":
    app.run()
