from fastapi import APIRouter, HTTPException
from src.api.v1.schemas.identification import IdentifyRequestScheme, IdentifyResponseScheme
from src.services.database_service import database_service

router = APIRouter()


@router.post("/identify", response_model=IdentifyResponseScheme)
async def identify(payload: IdentifyRequestScheme):
    """
    Идентифицировать пользователя
    """
    try:
        nickname = payload.nickname.strip()
        if not nickname:
            raise HTTPException(status_code=422, detail="nickname is required")
            # db
        user = await database_service.find_or_create_user(username=nickname)
        return IdentifyResponseScheme(user_id=user.user_id, nickname=user.username, role=user.role)
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
