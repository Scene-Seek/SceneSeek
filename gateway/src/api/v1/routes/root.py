from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def root():
    """
    GET: проверяет состояние шлюза
    """
    return {"message": "SceneSeek gateway"}
