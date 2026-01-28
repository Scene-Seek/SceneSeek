from fastapi import APIRouter

from src.api.v1.routes import root
from src.api.v1.routes import videos
from src.api.v1.routes import searches
from src.api.v1.routes import identify

router = APIRouter()

router.include_router(root.router, prefix="/api/v1")
router.include_router(videos.router, prefix="/api/v1")
router.include_router(searches.router, prefix="/api/v1")
router.include_router(identify.router, prefix="/api/v1")
