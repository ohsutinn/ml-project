from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.routing import APIRoute
from sqlmodel import SQLModel

from app.api.main import api_router
from app.core.config import settings
from app.core.db import async_engine


def custom_generate_unique_id(route: APIRoute) -> str:
    return f"{route.tags[0]}-{route.name}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with async_engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    yield
    await async_engine.dispose()


app = FastAPI(
    title="Machine Learning Project API server",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    generate_unique_id_function=custom_generate_unique_id,
    lifespan=lifespan,
)

app.include_router(api_router, prefix=settings.API_V1_STR)
