from fastapi import APIRouter

from src.api.v1.routes import root

router = APIRouter()

router.include_router(root.router, prefix="/api/v1")
