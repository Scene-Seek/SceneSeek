from fastapi import APIRouter, HTTPException
from src.api.v1.schemas.identification import IdentifyRequestScheme, IdentifyResponseScheme
from src.services.database_service import database_service
from src.api.dependencies import create_access_token, hash_password, verify_password

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
        if not payload.password:
            raise HTTPException(status_code=422, detail="password is required")

        user = await database_service.get_user_by_username(username=nickname)
        if user:
            if user.password_hash:
                if not verify_password(payload.password, user.password_hash):
                    raise HTTPException(status_code=401, detail="invalid nickname or password")
            else:
                user = await database_service.set_user_password(
                    user_id=user.user_id,
                    password_hash=hash_password(payload.password),
                )
                if not user:
                    raise HTTPException(status_code=404, detail="user not found")
        else:
            user = await database_service.create_user(
                username=nickname,
                password_hash=hash_password(payload.password),
            )

        token = create_access_token(user.user_id)
        return IdentifyResponseScheme(user_id=user.user_id, nickname=user.username, token=token)
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

