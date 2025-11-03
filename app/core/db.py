from sqlalchemy.ext.asyncio.session import AsyncSession

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import settings


async_engine = create_async_engine(
    str(settings.SQLALCHEMY_DATABASE_URI),
    echo=True,
    pool_pre_ping=True,
)

AsyncSessionFactory = async_sessionmaker[AsyncSession](
    bind=async_engine,
    expire_on_commit=False,
    autoflush=False,
)
