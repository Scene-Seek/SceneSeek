"""broker"""

import json
import os
import tempfile
from typing import Any
from urllib.request import urlretrieve
import cv2
from faststream.rabbit import RabbitBroker
from config import settings
from engine import VideoSearchEngine


broker = RabbitBroker(url=settings.RABBITMQ_URL)
# broker = RabbitBroker()

QUEUE_VIDEOS = "videos"
QUEUE_SEARCHES = "searches"


def _parse_payload(message: Any) -> dict | None:
    if isinstance(message, dict):
        return message
    if isinstance(message, str):
        try:
            parsed = json.loads(message)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None
    return None


engine: VideoSearchEngine | None = None


def set_engine(value: VideoSearchEngine) -> None:
    global engine
    engine = value


def get_engine() -> VideoSearchEngine:
    if engine is None:
        raise RuntimeError("Engine is not initialized")
    return engine


@broker.subscriber(queue=QUEUE_VIDEOS)
async def get_msg_videos(message: Any) -> None:
    print("INFO START HUINYI")
    payload = _parse_payload(message) or {}
    video_id = payload.get("video_id")
    user_id = payload.get("user_id", 1)
    object_name = payload.get("object_name")
    bucket = payload.get("bucket")
    video_url = payload.get("video_url")

    if not video_url:
        return

    suffix = os.path.splitext(object_name or "video.mp4")[1] or ".mp4"
    fd, tmp_path = tempfile.mkstemp(prefix="video_", suffix=suffix)
    os.close(fd)

    try:
        urlretrieve(video_url, tmp_path)
        current_engine = get_engine()
        await current_engine.run_indexing(video_path=tmp_path, user_id=int(user_id))
    finally:
        os.remove(tmp_path)



@broker.subscriber(queue=QUEUE_SEARCHES)
async def get_msg_searches(message: Any) -> None:
    payload = _parse_payload(message) or {}
    query_id = payload.get("query_id")
    user_id = payload.get("user_id")
    video_id = payload.get("video_id")
    query_text = payload.get("query_text")

    if not query_text:
        return

    current_engine = get_engine()
    results = await current_engine.search(query=query_text, query_id=query_id)
    if video_id is not None:
        results = [r for r in results if r.get("video_id") == int(video_id)]