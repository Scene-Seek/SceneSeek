"""Connect routes"""

from fastapi import APIRouter

from src.api.v1.routes import root
from src.api.v1.routes import videos
from src.api.v1.routes import searches
from src.api.v1.routes import identify

router = APIRouter()

prefix_api_v1 = "/api/v1"

router.include_router(root.router, prefix=prefix_api_v1)
router.include_router(videos.router, prefix=prefix_api_v1)
router.include_router(searches.router, prefix=prefix_api_v1)
router.include_router(identify.router, prefix=prefix_api_v1)
