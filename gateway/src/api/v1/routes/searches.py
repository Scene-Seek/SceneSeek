import re

from fastapi import APIRouter, HTTPException
from src.api.v1.schemas.search import GetSearchResultsScheme, GetSearchStatusResponseScheme, SearchResultItem, UploadSearchRequestScheme, UploadSearchResponseScheme
from src.services.broker_service import broker_service
from src.services.database_service import database_service

router = APIRouter()

# --- Grouping and threshold constants ---
GROUP_WINDOW_SEC = 5.0  # merge events within 5s of each other
GROUP_PADDING_SEC = 3.0  # ±3s padding around grouped interval
ADAPTIVE_THRESHOLD = 0.35  # score cutoff when many results
MIN_RESULTS_FOR_THRESHOLD = 5  # apply threshold only if more results than this
FALLBACK_TOP_N = 3  # if threshold removes all, keep top N

_PAD_RE = re.compile(r"\s*(<pad>)+\s*", re.IGNORECASE)


def _clean_caption(text: str | None) -> str | None:
    """Remove Florence-2 <pad> artefacts from caption."""
    if not text:
        return text
    return _PAD_RE.sub("", text).strip() or None


def _apply_adaptive_threshold(items: list[dict]) -> list[dict]:
    """Apply adaptive score filtering.
    - If ≤ MIN_RESULTS_FOR_THRESHOLD items: keep all (engine already filters by 0.25).
    - If > MIN_RESULTS_FOR_THRESHOLD items: drop those below ADAPTIVE_THRESHOLD.
    - If filtering removes everything: fallback to top FALLBACK_TOP_N by score.
    """
    if len(items) <= MIN_RESULTS_FOR_THRESHOLD:
        return items
    filtered = [it for it in items if it["score"] >= ADAPTIVE_THRESHOLD]
    if not filtered:
        # Fallback: return top N by score
        return sorted(items, key=lambda x: x["score"], reverse=True)[:FALLBACK_TOP_N]
    return filtered


def _group_events(items: list[dict], video_duration: float | None = None) -> list[SearchResultItem]:
    """Group temporally close events into intervals.
    Items must already be sorted by timestamp ASC.
    Returns SearchResultItem with start_time/end_time padded by GROUP_PADDING_SEC.
    Clamps end_time to video_duration if provided.
    """
    if not items:
        return []

    groups: list[list[dict]] = []
    current_group = [items[0]]

    for item in items[1:]:
        if item["timestamp"] - current_group[-1]["timestamp"] <= GROUP_WINDOW_SEC:
            current_group.append(item)
        else:
            groups.append(current_group)
            current_group = [item]
    groups.append(current_group)

    results = []
    for group in groups:
        best = max(group, key=lambda x: x["score"])
        min_ts = min(it["timestamp"] for it in group)
        max_ts = max(it["timestamp"] for it in group)

        # Clamp to video boundaries
        start_time = max(min_ts - GROUP_PADDING_SEC, 0.0)
        end_time = max_ts + GROUP_PADDING_SEC
        if video_duration is not None:
            end_time = min(end_time, video_duration)

        results.append(
            SearchResultItem(
                start_time=round(start_time, 2),
                end_time=round(end_time, 2),
                timestamp=best["timestamp"],
                score=best["score"],
                caption=best.get("caption"),
            )
        )

    # Sort final results by score descending
    results.sort(key=lambda r: r.score, reverse=True)
    return results


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
        _query = await database_service.create_query(user_id=payload.user_id, video_id=payload.video_id, query=payload.query_text)
        # broker
        await broker_service.pub(message={"query_id": _query.query_id, "user_id": _query.user_id, "video_id": _query.video_id, "query_text": _query.query_text}, queue=broker_service.QUEUE_SEARCHES)
        return UploadSearchResponseScheme(query_id=_query.query_id, user_id=_query.user_id, video_id=_query.video_id, query_text=_query.query_text, status=_query.processing_status)
    except HTTPException:
        raise
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
        if not query:
            raise HTTPException(status_code=404, detail="query not found")
        return GetSearchStatusResponseScheme(query_id=query_id, user_id=query.user_id, video_id=query.video_id, query_text=query.query_text, status=query.processing_status)
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/searches/{query_id}/results", response_model=GetSearchResultsScheme)
async def get_searches_results(query_id: int):
    """
    Получить результаты поиска
    """
    try:
        # Get query to find video_id
        query = await database_service.get_query_by_id(query_id=query_id)
        if not query:
            raise HTTPException(status_code=404, detail="query not found")

        # Get video to find duration
        video = await database_service.get_video_by_id(video_id=query.video_id)
        video_duration = video.duration if video and video.duration else None

        # Get results
        results = await database_service.get_query_results_by_id(query_id=query_id)
        # Extract valid results with existing events and timestamps
        raw_items = []
        for result in results:
            event = result.found_event
            if event is None or event.timestamp is None:
                continue
            raw_items.append(
                {
                    "timestamp": round(event.timestamp, 2),
                    "score": round(result.similarity_score, 4) if result.similarity_score is not None else 0.0,
                    "caption": _clean_caption(event.caption) if hasattr(event, "caption") else None,
                }
            )

        # Adaptive threshold filtering
        raw_items = _apply_adaptive_threshold(raw_items)

        # Sort by timestamp for grouping
        raw_items.sort(key=lambda x: x["timestamp"])

        # Group close events into intervals with video duration boundary
        grouped_results = _group_events(raw_items, video_duration=video_duration)

        return GetSearchResultsScheme(
            query_id=query_id,
            result=grouped_results,
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
