import asyncpg
from faststream import FastStream
from pgvector.asyncpg import register_vector

from broker import GatewayVideoSearchEngine, broker, set_engine
from config import settings
from engine import IndexerConfig

app = FastStream(broker)


@app.on_startup
async def init_engine() -> None:
    engine = GatewayVideoSearchEngine(
        config=IndexerConfig(db_dsn=settings.DATABASE_URL)
    )
    engine.pool = await asyncpg.create_pool(
        dsn=settings.DATABASE_URL,
        init=register_vector
    )
    set_engine(engine)


@app.on_shutdown
async def close_engine() -> None:
    from broker import get_engine

    engine = get_engine()
    await engine.close()


if __name__ == "__main__":
    app.run()
