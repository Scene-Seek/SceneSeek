from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import (
    create_access_token,
    get_current_user_id,
    hash_password,
    verify_password,
)
from src.api.v1.schemas.auth import (
    AuthRequestScheme,
    AuthResponseScheme,
    CurrentUserResponseScheme,
)
from src.models.users import Users
from src.services.database_service import database_service

router = APIRouter(prefix="/auth")


def _is_anonymous(user: Users) -> bool:
    return user.password_hash is None and user.username.startswith("anonymous_")


def _auth_response(user: Users) -> AuthResponseScheme:
    return AuthResponseScheme(
        user_id=user.user_id,
        username=user.username,
        token=create_access_token(user.user_id),
        is_anonymous=_is_anonymous(user),
    )


def _clean_username(username: str) -> str:
    username = username.strip()
    if not username:
        raise HTTPException(status_code=422, detail="username is required")
    return username


@router.post("/register", response_model=AuthResponseScheme)
async def register(payload: AuthRequestScheme):
    """
    POST: регистрирует нового пользователя
    """
    username = _clean_username(payload.username)

    existing_user = await database_service.get_user_by_username(username=username)
    if existing_user:
        raise HTTPException(status_code=409, detail="username already exists")

    user = await database_service.create_user(
        username=username,
        password_hash=hash_password(payload.password),
    )
    return _auth_response(user)


@router.post("/login", response_model=AuthResponseScheme)
async def login(payload: AuthRequestScheme):
    """
    POST: авторизует пользователя
    """
    username = _clean_username(payload.username)
    user = await database_service.get_user_by_username(username=username)

    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="invalid username or password")

    return _auth_response(user)
    
@router.post("/anonymous", response_model=AuthResponseScheme)
async def anonymous():
    """
    POST: создает анонимного пользователя
    """
    user = await database_service.create_user(
        username=f"anonymous_{uuid4().hex[:16]}",
        password_hash=None,
    )
    return _auth_response(user)
    
@router.get("/me", response_model=CurrentUserResponseScheme)
async def me(current_user_id: int = Depends(get_current_user_id)):
    """
    GET: возвращает информацию о текущем пользователе
    """
    user = await database_service.get_user_by_id(user_id=current_user_id)
    if not user:
        raise HTTPException(status_code=404, detail="user not found")

    return CurrentUserResponseScheme(
        user_id=user.user_id,
        username=user.username,
        is_anonymous=_is_anonymous(user),
    )
