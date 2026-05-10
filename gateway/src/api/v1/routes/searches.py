import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from src.api.dependencies import get_current_user_id
from src.api.v1.schemas.search import (
    GetSearchHistoryScheme,
    GetSearchResultsScheme,
    GetSearchStatusResponseScheme,
    SearchHistoryItemScheme,
    SearchResultItem,
    UploadSearchRequestScheme,
    UploadSearchResponseScheme,
)
from src.services.broker_service import broker_service
from src.services.database_service import database_service

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/searches", response_model=UploadSearchResponseScheme)
async def post_searches(
    payload: UploadSearchRequestScheme,
    current_user_id: int = Depends(get_current_user_id),
):
    """
    Создать новый промпт
    """
    try:
        user = await database_service.get_user_by_id(user_id=current_user_id)
        if not user:
            raise HTTPException(status_code=404, detail="user not found")
        # Если нет видео, то исключение
        video = await database_service.get_video_by_id(video_id=payload.video_id)
        if not video:
            raise HTTPException(status_code=404, detail="video not found")
        if video.uploaded_by_user_id != current_user_id:
            raise HTTPException(status_code=403, detail="video does not belong to user")
        # db
        _query = await database_service.create_query(
            user_id=current_user_id, video_id=payload.video_id, query=payload.query_text
        )
        # broker
        await broker_service.pub(
            message={
                "query_id": _query.query_id,
                "user_id": _query.user_id,
                "video_id": _query.video_id,
                "query_text": _query.query_text,
            },
            queue=broker_service.QUEUE_SEARCHES,
        )
        return UploadSearchResponseScheme(
            query_id=_query.query_id,
            user_id=_query.user_id,
            video_id=_query.video_id,
            query_text=_query.query_text,
            status=_query.processing_status,
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Unexpected error while creating search prompt")
        raise HTTPException(status_code=500, detail="internal server error")


@router.get("/searches/history", response_model=GetSearchHistoryScheme)
async def get_searches_history(
    current_user_id: int = Depends(get_current_user_id),
    limit: int = Query(default=30, ge=1, le=100),
):
    """
    Получить историю поиска пользователя
    """
    try:
        history = await database_service.get_search_history_by_user(
            user_id=current_user_id,
            limit=limit,
        )
        return GetSearchHistoryScheme(
            history=[
                SearchHistoryItemScheme(
                    query_id=item.query_id,
                    video_id=item.video_id,
                    video_title=item.video.title if item.video else "",
                    query_text=item.query_text,
                    search_status=item.processing_status,
                    video_status=item.video.processing_status
                    if item.video
                    else "failed",
                    search_date=item.search_date,
                )
                for item in history
            ]
        )
    except Exception:
        logger.exception("Unexpected error while getting search history")
        raise HTTPException(status_code=500, detail="internal server error")


@router.get("/searches/{query_id}", response_model=GetSearchStatusResponseScheme)
async def get_searches_status(
    query_id: int, current_user_id: int = Depends(get_current_user_id)
):
    """
    Получить статус поиска
    """
    try:
        # db
        query = await database_service.get_query_by_id(query_id=query_id)
        if not query:
            raise HTTPException(status_code=404, detail="query not found")
        if query.user_id != current_user_id:
            raise HTTPException(status_code=403, detail="query does not belong to user")
        return GetSearchStatusResponseScheme(
            query_id=query_id,
            user_id=query.user_id,
            video_id=query.video_id,
            query_text=query.query_text,
            status=query.processing_status,
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Unexpected error while getting search status")
        raise HTTPException(status_code=500, detail="internal server error")


@router.get("/searches/{query_id}/results", response_model=GetSearchResultsScheme)
async def get_searches_results(
    query_id: int, current_user_id: int = Depends(get_current_user_id)
):
    """
    Получить результаты поиска
    """
    try:
        query = await database_service.get_query_by_id(query_id=query_id)
        if not query:
            raise HTTPException(status_code=404, detail="query not found")
        if query.user_id != current_user_id:
            raise HTTPException(status_code=403, detail="query does not belong to user")

        results = await database_service.get_query_results_by_id(query_id=query_id)
        payload_items: list[SearchResultItem] = []
        for result in results:
            if (
                result.segment_start is None
                or result.segment_end is None
                or result.best_ts is None
            ):
                continue

            payload_items.append(
                SearchResultItem(
                    start=round(result.segment_start, 2),
                    end=round(result.segment_end, 2),
                    best_ts=round(result.best_ts, 2),
                    score=round(result.similarity_score, 4)
                    if result.similarity_score is not None
                    else 0.0,
                    bbox=result.bbox or [],
                    type=result.hit_type or "unknown",
                )
            )

        payload_items.sort(key=lambda x: x.score, reverse=True)

        return GetSearchResultsScheme(
            query_id=query_id,
            result=payload_items,
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Unexpected error while getting search results")
        raise HTTPException(status_code=500, detail="internal server error")
