import datetime
import hashlib
import hmac
import secrets

import jwt

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from src.core.config import settings

PASSWORD_HASH_ALGORITHM = "pbkdf2_sha256"

security = HTTPBearer(auto_error=False)


def create_access_token(user_id: int) -> str:
    expire = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": str(user_id),
        "exp": expire,
        "iat": datetime.datetime.now(datetime.timezone.utc),
        "type": "access",
    }
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        settings.PASSWORD_HASH_ITERATIONS,
    ).hex()
    return f"{PASSWORD_HASH_ALGORITHM}${settings.PASSWORD_HASH_ITERATIONS}${salt}${password_hash}"


def verify_password(password: str, stored_hash: str | None) -> bool:
    if not stored_hash:
        return False
    try:
        algorithm, iterations_raw, salt, expected_hash = stored_hash.split("$", 3)
        iterations = int(iterations_raw)
    except ValueError:
        return False

    if algorithm != PASSWORD_HASH_ALGORITHM:
        return False

    actual_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        iterations,
    ).hex()
    return hmac.compare_digest(actual_hash, expected_hash)


def get_current_user_id(credentials: HTTPAuthorizationCredentials | None = Depends(security)) -> int:
    if credentials is None:
        raise HTTPException(status_code=401, detail="Требуется авторизация")

    token = credentials.credentials
    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        sub = payload.get("sub")
        if sub is None:
            raise HTTPException(status_code=401, detail="Невалидный токен")
        return int(sub)
    except ValueError:
        raise HTTPException(status_code=401, detail="Невалидный токен")
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Токен истек. Войдите заново.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Неверный токен.")
