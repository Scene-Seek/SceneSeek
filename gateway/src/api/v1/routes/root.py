from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
router = APIRouter()

@router.get("/")
async def root():
    return {
        "message": "root"
    }

@router.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    if exc.status_code == 404:
        return JSONResponse(
            status_code=404,
            content={
                "detail": "Endpoint not found"
            }
        )
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
