import logging
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile
from fastapi.responses import RedirectResponse
from src.api.dependencies import get_current_user_id
from src.api.v1.schemas.video import (
    GetVideoHistoryScheme,
    GetVideoResponseScheme,
    UploadVideoResponseScheme,
    VideoHistoryItemScheme,
)
from src.services.broker_service import broker_service
from src.services.database_service import database_service
from src.services.minio_service import minio_service

router = APIRouter()
logger = logging.getLogger(__name__)

SUPPORTED_VIDEO_CONTENT_TYPES = {
    "application/mp4",
    "application/octet-stream",
    "video/mp4",
    "video/ogg",
    "video/quicktime",
    "video/webm",
    "video/x-m4v",
}
SUPPORTED_VIDEO_EXTENSIONS = {".m4v", ".mov", ".mp4", ".ogg", ".webm"}


def validate_video_file(file: UploadFile) -> str:
    original_name = file.filename or "video.mp4"
    content_type = (file.content_type or "").lower()
    filename = original_name.lower()
    has_video_extension = any(
        filename.endswith(ext) for ext in SUPPORTED_VIDEO_EXTENSIONS
    )
    has_video_content_type = (
        content_type.startswith("video/")
        or content_type in SUPPORTED_VIDEO_CONTENT_TYPES
    )

    if not has_video_content_type or not has_video_extension:
        raise HTTPException(status_code=415, detail="unsupported video file type")

    return original_name


@router.post("/videos", response_model=UploadVideoResponseScheme)
async def post_videos(
    file: UploadFile, current_user_id: int = Depends(get_current_user_id)
):
    """
    Создать новое видео
    """

    try:
        # Если нет пользователя, то исключение
        user = await database_service.get_user_by_id(user_id=current_user_id)
        if not user:
            raise HTTPException(status_code=404, detail="user not found")
        original_name = validate_video_file(file)
        object_name = f"{current_user_id}/{uuid4()}_{original_name}"
        minio_service.save_obj(
            obj=file,
            bucket=minio_service.BUCKET_VIDEOS_IN,
            object_name=object_name,
        )
        object_key = f"{minio_service.BUCKET_VIDEOS_IN}/{object_name}"
        # db
        video = await database_service.create_video(
            uploaded_by_user_id=current_user_id,
            title=original_name,
            path=object_key,
            duration=None,
            fps=None,
            resolution=None,
            processing_status="pending",
        )
        video_url = minio_service.get_video_url_internal(object_name=object_name)
        # broker
        await broker_service.pub(
            message={
                "video_id": video.video_id,
                "user_id": current_user_id,
                "object_name": object_name,
                "bucket": minio_service.BUCKET_VIDEOS_IN,
                "video_url": video_url,
            },
            queue=broker_service.QUEUE_VIDEOS,
        )
        return UploadVideoResponseScheme(
            video_id=video.video_id, status=video.processing_status
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Unexpected error while uploading video")
        raise HTTPException(status_code=500, detail="internal server error")


@router.get("/videos/history", response_model=GetVideoHistoryScheme)
async def get_videos_history(
    current_user_id: int = Depends(get_current_user_id),
    limit: int = Query(default=30, ge=1, le=100),
):
    """
    Получить историю видео пользователя
    """
    try:
        history = await database_service.get_video_history_by_user(
            user_id=current_user_id,
            limit=limit,
        )
        return GetVideoHistoryScheme(
            history=[
                VideoHistoryItemScheme(
                    video_id=video.video_id,
                    video_title=video.title,
                    video_status=video.processing_status,
                    created_at=video.created_at,
                    latest_query_id=query_id,
                    latest_query_text=query_text,
                    latest_search_status=search_status,
                    latest_search_date=search_date,
                )
                for video, query_id, query_text, search_status, search_date in history
            ]
        )
    except Exception:
        logger.exception("Unexpected error while getting video history")
        raise HTTPException(status_code=500, detail="internal server error")


@router.get("/videos/{video_id}", response_model=GetVideoResponseScheme)
async def get_videos(
    video_id: int, current_user_id: int = Depends(get_current_user_id)
):
    """
    Получить ссылку на видео по id
    """
    try:
        # db
        video = await database_service.get_video_by_id(video_id=video_id)
        if not video:
            raise HTTPException(status_code=404, detail="video not found")
        if video.uploaded_by_user_id != current_user_id:
            raise HTTPException(status_code=403, detail="video does not belong to user")
        # формат: "bucket/object_name"
        parts = video.path.split("/", 1)
        if len(parts) == 2:
            bucket, obj_name = parts
            video_url = minio_service.get_presigned_url(
                bucket=bucket,
                object_name=obj_name,
            )
        else:
            video_url = video.path
        return GetVideoResponseScheme(
            video_id=video_id, video_path=video_url, status=video.processing_status
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Unexpected error while getting video")
        raise HTTPException(status_code=500, detail="internal server error")


@router.get("/videos/{video_id}/content")
async def get_video_content(
    video_id: int, current_user_id: int = Depends(get_current_user_id)
):
    try:
        video = await database_service.get_video_by_id(video_id=video_id)
        if not video:
            raise HTTPException(status_code=404, detail="video not found")
        if video.uploaded_by_user_id != current_user_id:
            raise HTTPException(status_code=403, detail="video does not belong to user")

        parts = video.path.split("/", 1)
        if len(parts) == 2:
            bucket, obj_name = parts
            video_url = minio_service.get_presigned_url(
                bucket=bucket, object_name=obj_name, expires_seconds=3600
            )
            return RedirectResponse(url=video_url)
        else:
            return RedirectResponse(url=video.path)

    except HTTPException:
        raise
    except Exception:
        logger.exception(f"Error redirecting to video content: {video_id}")
        raise HTTPException(status_code=500, detail="internal server error")
