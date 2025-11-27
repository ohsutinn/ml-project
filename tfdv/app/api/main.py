from fastapi import APIRouter

from app.api.routes import data_validation

api_router = APIRouter()
api_router.include_router(data_validation.router)


@api_router.get("/health", tags=["health"])
async def health_check():
    return {
        "status": "ok",
        "service": "tfdv-validation-server",
    }
