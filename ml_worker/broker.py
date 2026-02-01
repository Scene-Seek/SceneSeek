"""broker"""

import json
import os
import tempfile
from typing import Any
from urllib.request import urlretrieve

import cv2
from faststream.rabbit import RabbitBroker

from config import settings
from engine import IndexerConfig, VideoSearchEngine

broker = RabbitBroker(url=settings.RABBITMQ_URL)

QUEUE_VIDEOS = "videos"
QUEUE_SEARCHES = "searches"

class GatewayVideoSearchEngine(VideoSearchEngine):
    def __init__(self, config: IndexerConfig, use_float16: bool = True) -> None:
        super().__init__(config=config, use_float16=use_float16)
        self._target_video_id: int | None = None

    def set_target_video_id(self, video_id: int) -> None:
        self._target_video_id = video_id

    async def _create_video_entry(self, video_path: str, user_id: int) -> int:
        if self._target_video_id is None:
            raise ValueError("target_video_id is not set")
        if not self.pool:
            await self.initialize_db()

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps if fps else 0.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        resolution = f"{w}x{h}"

        sql = """
            UPDATE videos
            SET duration = $1,
                fps = $2,
                resolution = $3,
                processing_status = 'indexing'
            WHERE video_id = $4
        """

        async with self.pool.acquire() as conn:
            await conn.execute(sql, duration, fps, resolution, self._target_video_id)

        return self._target_video_id


engine: GatewayVideoSearchEngine | None = None


def set_engine(value: GatewayVideoSearchEngine) -> None:
    global engine
    engine = value


def get_engine() -> GatewayVideoSearchEngine:
    if engine is None:
        raise RuntimeError("Engine is not initialized")
    return engine

@broker.subscriber(queue=QUEUE_VIDEOS)
async def get_video(message: Any) -> None:
    """
    Получение сообщения о видео из очереди
    """
    try:
        payload = message
        if isinstance(message, str):
            try:
                payload = json.loads(message)
            except json.JSONDecodeError:
                payload = message

        if isinstance(payload, dict):
            video_id = payload.get("video_id")
            user_id = payload.get("user_id", 1)
            video_url = payload.get("video_url")
            object_name = payload.get("object_name", "video.mp4")
            bucket = payload.get("bucket")

            print(f"[QUEUE_VIDEOS] video_id={video_id} object_name={object_name} bucket={bucket}")

            if not video_id or not video_url:
                print("[QUEUE_VIDEOS] missing video_id or video_url")
                return

            suffix = os.path.splitext(object_name)[1] or ".mp4"
            fd, tmp_path = tempfile.mkstemp(prefix="video_", suffix=suffix)
            os.close(fd)

            urlretrieve(video_url, tmp_path)

            current_engine = get_engine()
            current_engine.set_target_video_id(video_id)
            await current_engine.run_indexing(tmp_path, user_id=int(user_id))

            os.remove(tmp_path)
        else:
            print(f"[QUEUE_VIDEOS] message={payload}")
    except Exception as e:
        print(f"Error: {e}")


@broker.subscriber(queue=QUEUE_SEARCHES)
async def get_seacrh(message: Any) -> None:
    """
    Получение сообщения о промпте из очереди
    """
    try:
        payload = message
        if isinstance(message, str):
            try:
                payload = json.loads(message)
            except json.JSONDecodeError:
                payload = message

        if isinstance(payload, dict):
            query_id = payload.get("query_id")
            user_id = payload.get("user_id")
            video_id = payload.get("video_id")
            query_text = payload.get("query_text")
            print(
                f"[QUEUE_SEARCHES] query_id={query_id} user_id={user_id} "
                f"video_id={video_id} query_text={query_text}"
            )
            if query_text:
                current_engine = get_engine()
                results = await current_engine.search(query_text)
                if video_id is not None:
                    results = [r for r in results if r.get("video_id") == int(video_id)]
                print(f"[QUEUE_SEARCHES] results={results}")
        else:
            print(f"[QUEUE_SEARCHES] message={payload}")
    except Exception as e:
        print(f"Error: {e}")