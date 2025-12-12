from fastapi import APIRouter

from app.api.routes import dataset, dataset_baseline, dataset_version, model

api_router = APIRouter()

api_router.include_router(dataset.router)
api_router.include_router(dataset_version.router)
api_router.include_router(dataset_baseline.router)
api_router.include_router(model.router)
api_router.include_router(model.internal_router)


@api_router.get("/health", tags=["health"])
async def health_check():
    return {
        "status": "ok",
        "service": "api-server",
    }
