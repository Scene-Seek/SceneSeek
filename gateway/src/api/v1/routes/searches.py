from fastapi import APIRouter

router = APIRouter()

@router.post("/searches")
async def post_searches():
    """
    Создать поиск по видео
    """
    pass

@router.get("/searches/{query_id}")
async def get_searches_status(query_id: int):
    """
    Получить статус поиска
    """
    pass

@router.get("/searches/{query_id}/results")
async def get_searches_results(query_id: int):
    """
    Получить результаты поиска
    """
    pass