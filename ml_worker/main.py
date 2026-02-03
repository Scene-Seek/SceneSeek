import asyncpg
from broker import broker, set_engine
from config import settings
from engine import IndexerConfig, VideoSearchEngine
from faststream import FastStream
from pgvector.asyncpg import register_vector

app = FastStream(broker)


@app.on_startup
async def init_app() -> None:
    # Create IndexerConfig from settings to use environment variables
    conf = IndexerConfig.from_settings(settings)
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
