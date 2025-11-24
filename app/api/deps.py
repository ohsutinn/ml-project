from collections.abc import AsyncGenerator
from typing import Annotated

from sqlmodel.ext.asyncio.session import AsyncSession
from app.core.db import AsyncSessionFactory
from fastapi import Depends


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionFactory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise

SessionDep = Annotated[AsyncSession, Depends(get_db)]