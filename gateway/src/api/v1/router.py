from fastapi import APIRouter

from src.api.v1.routes import root
from src.api.v1.routes import videos

router = APIRouter()

router.include_router(root.router, prefix="/api/v1")
router.include_router(videos.router, prefix="/api/v1")
