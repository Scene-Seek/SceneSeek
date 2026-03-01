from uuid import uuid4

from fastapi import APIRouter, Form, HTTPException, UploadFile
from src.api.v1.schemas.video import GetVideoResponseScheme, UploadVideoResponseScheme
from src.services.broker_service import broker_service
from src.services.database_service import database_service
from src.services.minio_service import minio_service

router = APIRouter()


@router.post("/videos", response_model=UploadVideoResponseScheme)
async def post_videos(file: UploadFile, user_id: int = Form(...)):
    """
    Создать новое видео
    """

    try:
        # Если нет пользователя, то исключение
        user = await database_service.get_user_by_id(user_id=user_id)
        if not user:
            raise HTTPException(status_code=404, detail="user not found")
        # Generate unique object name to avoid collisions
        original_name = file.filename or "video.mp4"
        object_name = f"{user_id}/{uuid4()}_{original_name}"
        # minio - store with unique name
        minio_service.save_obj(
            obj=file,
            bucket=minio_service.BUCKET_VIDEOS_IN,
            object_name=object_name,
        )
        # Store object key in DB (not presigned URL)
        object_key = f"{minio_service.BUCKET_VIDEOS_IN}/{object_name}"
        # db
        video = await database_service.create_video(uploaded_by_user_id=user_id, title=original_name, path=object_key, duration=None, fps=None, resolution=None, processing_status="pending")
        # Generate fresh presigned URL for ML worker
        video_url = minio_service.get_video_url_internal(object_name=object_name)
        # broker
        await broker_service.pub(
            message={"video_id": video.video_id, "user_id": user_id, "object_name": object_name, "bucket": minio_service.BUCKET_VIDEOS_IN, "video_url": video_url}, queue=broker_service.QUEUE_VIDEOS
        )
        return UploadVideoResponseScheme(video_id=video.video_id, status=video.processing_status)
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/videos/{video_id}", response_model=GetVideoResponseScheme)
async def get_videos(video_id: int):
    """
    Получить ссылку на видео по id
    """
    try:
        # db
        video = await database_service.get_video_by_id(video_id=video_id)
        if not video:
            raise HTTPException(status_code=404, detail="video not found")
        # Generate fresh presigned URL from stored object key
        # path format: "bucket/object_name"
        parts = video.path.split("/", 1)
        if len(parts) == 2:
            bucket, obj_name = parts
            video_url = minio_service.get_presigned_url(
                bucket=bucket,
                object_name=obj_name,
            )
        else:
            # Fallback: path is already a URL or simple key
            video_url = video.path
        return GetVideoResponseScheme(video_id=video_id, video_path=video_url, status=video.processing_status)
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
