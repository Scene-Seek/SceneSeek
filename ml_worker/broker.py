import json
import logging
import os
import tempfile
from time import perf_counter
from typing import Any
from urllib.request import urlretrieve

from config import settings
from engine import VideoSearchEngine
from faststream.rabbit import RabbitBroker

broker = RabbitBroker(url=settings.RABBITMQ_URL)

QUEUE_VIDEOS = "videos"
QUEUE_SEARCHES = "searches"

logger = logging.getLogger(__name__)


def _parse_payload(message: Any) -> dict | None:
    """Пытается извлечь словарь полезной нагрузки из входного сообщения"""
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
    """Сохраняет экземпляр движка для использования в обработчиках"""
    global engine
    engine = value


def get_engine() -> VideoSearchEngine:
    """Возвращает инициализированный движок или выбрасывает ошибку"""
    if engine is None:
        raise RuntimeError("Engine is not initialized")
    return engine


def _preview_text(value: str, limit: int = 80) -> str:
    """Сжимает текст для логов и убирает переводы строк"""
    text = value.strip().replace("\n", " ")
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


@broker.subscriber(queue=QUEUE_VIDEOS)
async def get_msg_videos(message: Any) -> None:
    """Обрабатывает сообщения о новых видео и запускает индексацию"""
    payload = _parse_payload(message) or {}
    video_id = payload.get("video_id")
    user_id = payload.get("user_id", 1)
    object_name = payload.get("object_name")
    # bucket = payload.get("bucket")
    video_url = payload.get("video_url")

    logger.info(
        "Received video indexing task: video_id=%s user_id=%s object_name=%s has_url=%s",
        video_id,
        user_id,
        object_name,
        bool(video_url),
    )

    if not video_url:
        logger.warning("Video task skipped because video_url is missing: video_id=%s object_name=%s", video_id, object_name)
        return

    suffix = os.path.splitext(object_name or "video.mp4")[1] or ".mp4"
    fd, tmp_path = tempfile.mkstemp(prefix="video_", suffix=suffix)
    os.close(fd)

    try:
        logger.info("Downloading source video for video_id=%s into %s", video_id, tmp_path)
        download_started = perf_counter()
        urlretrieve(video_url, tmp_path)
        logger.info(
            "Source video downloaded: video_id=%s size_bytes=%s duration_sec=%.2f",
            video_id,
            os.path.getsize(tmp_path),
            perf_counter() - download_started,
        )
        current_engine = get_engine()
        logger.info("Dispatching video_id=%s to indexing engine", video_id)
        await current_engine.run_indexing(
            video_path=tmp_path,
            user_id=int(user_id),
            video_id=int(video_id) if video_id is not None else None,
        )
        logger.info("Video indexing task completed successfully: video_id=%s", video_id)
    except Exception:
        logger.exception(
            "Video indexing task failed: video_id=%s user_id=%s object_name=%s",
            video_id,
            user_id,
            object_name,
        )
        if video_id is not None:
            try:
                current_engine = get_engine()
                await current_engine._finalize_video_status(int(video_id), "failed")
            except Exception:
                logger.exception("Failed to mark video as failed after indexing error: video_id=%s", video_id)
        raise
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            logger.info("Removed temporary source video for video_id=%s", video_id)


@broker.subscriber(queue=QUEUE_SEARCHES)
async def get_msg_searches(message: Any) -> None:
    """Обрабатывает поисковые запросы и сохраняет результат в базе"""
    payload = _parse_payload(message) or {}
    query_id = payload.get("query_id")
    # user_id = payload.get("user_id")
    video_id = payload.get("video_id")
    query_text = payload.get("query_text")

    logger.info(
        "Received search task: query_id=%s video_id=%s query_len=%s preview=%r",
        query_id,
        video_id,
        len(query_text) if isinstance(query_text, str) else None,
        _preview_text(query_text) if isinstance(query_text, str) else None,
    )

    if not query_text:
        logger.warning("Search task skipped because query_text is missing: query_id=%s video_id=%s", query_id, video_id)
        return

    try:
        current_engine = get_engine()
        results = await current_engine.search(
            query=query_text,
            query_id=query_id,
            video_id=int(video_id) if video_id is not None else None,
        )

        if query_id is not None:
            status = "ready" if results else "not_found"
            await current_engine.update_search_status(query_id=query_id, query_text=query_text, status=status)
        logger.info(
            "Search task completed: query_id=%s video_id=%s results=%s final_status=%s",
            query_id,
            video_id,
            len(results),
            "ready" if results else "not_found",
        )
    except Exception:
        logger.exception("Search task failed: query_id=%s video_id=%s", query_id, video_id)
        if query_id is not None:
            try:
                current_engine = get_engine()
                await current_engine.update_search_status(query_id=query_id, query_text=query_text, status="failed")
            except Exception:
                logger.exception("Failed to mark search as failed after error: query_id=%s", query_id)
