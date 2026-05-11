import logging
import os

import asyncpg
from broker import QUEUE_SEARCHES, QUEUE_VIDEOS, broker, set_engine
from config import settings
from engine import IndexerConfig, VideoSearchEngine
from faststream import FastStream
from pgvector.asyncpg import register_vector

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

logger = logging.getLogger(__name__)

app = FastStream(broker)


@app.on_startup
async def init_app() -> None:
    """Поднимает движок, инициализирует пул БД и подписки брокера"""
    logger.info("ML worker startup initiated")
    conf = IndexerConfig.from_settings(settings)
    logger.info(
        "Resolved worker config: frame_skip=%s batch_size=%s motion_threshold=%s yolo_model=%s siglip_model=%s",
        conf.frame_skip,
        conf.batch_size,
        conf.motion_threshold,
        conf.model_path,
        conf.siglip_model_id,
    )
    engine = VideoSearchEngine(config=conf)
    logger.info("Creating asyncpg connection pool for worker")
    engine.pool = await asyncpg.create_pool(dsn=conf.db_dsn, init=register_vector)
    set_engine(engine)
    logger.info("ML worker is ready; subscribed queues: %s, %s", QUEUE_VIDEOS, QUEUE_SEARCHES)


@app.on_shutdown
async def close_app() -> None:
    """Корректно завершает работу движка и освобождает ресурсы"""
    from broker import get_engine

    logger.info("ML worker shutdown initiated")
    engine = get_engine()
    await engine.close()
    logger.info("ML worker shutdown complete")


if __name__ == "__main__":
    app.run()
