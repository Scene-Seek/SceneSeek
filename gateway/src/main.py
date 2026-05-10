import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.v1.router import router
from src.core.broker import broker
from src.core.database import create_tables

logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

from src.models.search_history import SearchHistory  # noqa: F401
from src.models.search_results import SearchResults  # noqa: F401
from src.models.users import Users  # noqa: F401
from src.models.video_events import VideoEvents  # noqa: F401
from src.models.videos import Videos  # noqa: F401
from src.services.minio_service import minio_service

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Init DB")
    await create_tables()

    logger.info("Init RabbitMQ")
    await broker.start()

    logger.info("Init MinIO")
    minio_service.create_buckets()

    yield

    await broker.stop()
    logger.info("Termination")


app = FastAPI(lifespan=lifespan)

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router=router)
