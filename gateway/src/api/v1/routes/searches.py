from fastapi import APIRouter, HTTPException
from src.services.broker_service import broker_service
from src.services.database_service import database_service
from src.api.v1.schemas.search import (
    UploadSearchRequestScheme,
    UploadSearchResponseScheme,
    GetSearchStatusResponseScheme,
    GetSearchResultsScheme
)

router = APIRouter()

@router.post("/searches", response_model=UploadSearchResponseScheme)
async def post_searches(payload: UploadSearchRequestScheme):
    """
    Создать новый промпт
    """
    try:
        # Если нет пользователя, то исключение
        user = await database_service.get_user_by_id(user_id=payload.user_id)
        if not user:
            raise HTTPException(status_code=404, detail="user not found")
        # Если нет видео, то исключение
        video = await database_service.get_video_by_id(video_id=payload.video_id)
        if not video:
            raise HTTPException(status_code=404, detail="video not found")
        if video.uploaded_by_user_id != payload.user_id:
            raise HTTPException(status_code=403, detail="video does not belong to user")
        # db
        _query = await database_service.create_query(
            user_id=payload.user_id,
            video_id=payload.video_id,
            query=payload.query_text
        )
        # broker
        await broker_service.pub(
            message={
                "query_id": _query.query_id,
                "user_id": _query.user_id,
                "video_id": _query.video_id,
                "query_text": _query.query_text
            },
            queue=broker_service.QUEUE_SEARCHES
        )
        return UploadSearchResponseScheme(
            query_id=_query.query_id,
            user_id=_query.user_id,
            video_id=_query.video_id,
            query_text=_query.query_text,
            status=_query.processing_status
        )
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/searches/{query_id}", response_model=GetSearchStatusResponseScheme)
async def get_searches_status(query_id: int):
    """
    Получить статус поиска
    """
    try:
        # db
        query = await database_service.get_query_by_id(query_id=query_id)
        return GetSearchStatusResponseScheme(
            query_id=query_id,
            user_id=query.user_id,
            video_id=query.video_id,
            query_text=query.query_text,
            status=query.processing_status
        )
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/searches/{query_id}/results", response_model=GetSearchResultsScheme)
async def get_searches_results(query_id: int):
    """
    Получить результаты поиска
    """
    try:
        # db
        results = await database_service.get_query_results_by_id(query_id=query_id)
        # TODO переделать на таймкоды
        results = [result.similarity_score for result in results] 
        return GetSearchResultsScheme(
            query_id=query_id,
            result=results,
        )
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))