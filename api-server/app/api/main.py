from fastapi import APIRouter

from app.api.routes import dataset, model

api_router = APIRouter()

api_router.include_router(dataset.router)
api_router.include_router(model.router)
