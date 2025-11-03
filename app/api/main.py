from fastapi import APIRouter

from app.api.routes import dataset

api_router = APIRouter()
api_router.include_router(dataset.router)
