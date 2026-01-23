import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.core.database import engine, create_tables
from src.core.broker import broker
from src.services.minio_service import minio_service

from src.api.v1.router import router

from src.models.search_history import SearchHistory
from src.models.search_results import SearchResults
from src.models.users import Users
from src.models.video_events import VideoEvents
from src.models.videos import Videos


@asynccontextmanager
async def lifespan(app: FastAPI):

    print("INFO: init DB")
    await create_tables()

    print("INFO: Init RabbitMQ")
    await broker.start()

    print("INFO: Init MinIO")
    minio_service.create_buckets()


    yield

    await broker.stop()

    # await engine.dispose()

    print("INFO: Termination")


app = FastAPI(lifespan=lifespan)

origins = [
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router=router)

if __name__ == "__main__":
    asyncio.run(app.run())