from fastapi import APIRouter, HTTPException
from src.services.broker_service import broker_service
from src.services.database_service import database_service
from src.api.v1.schemas.search import (
    UploadSearchRequestScheme,
    UploadSearchResponseScheme,
    GetSearchStatusResponseScheme
)

router = APIRouter()

@router.post("/searches", response_model=UploadSearchResponseScheme)
async def post_searches(payload: UploadSearchRequestScheme):
    """
    Создать новый промпт
    """
    try:
        # broker
        await broker_service.pub(message=payload.query_text, queue=broker_service.QUEUE_SEARCHES)
        # db
        _query = await database_service.create_query(
            user_id=payload.user_id,
            video_id=payload.video_id,
            query=payload.query_text
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

@router.get("/searches/{query_id}/results")
async def get_searches_results(query_id: int):
    """
    Получить результаты поиска
    """
    pass