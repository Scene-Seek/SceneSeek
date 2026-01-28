from fastapi import APIRouter, HTTPException
from src.services.database_service import database_service
from src.api.v1.schemas.identification import IdentifyResponseScheme

router = APIRouter()

@router.post("/identify", response_model=IdentifyResponseScheme)
async def identify(nickname: str):
    """
    Идентифицировать пользователя
    """
    try:
        user = await database_service.find_or_create_user(
            username=nickname
        )
        return IdentifyResponseScheme(user_id=user.user_id, nickname=user.username, role=user.role)
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))