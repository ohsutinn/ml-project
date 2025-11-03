from __future__ import annotations

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel

from app.core.config import settings
from app.models import *

# Alembic Config 객체
config = context.config

# 로거 설정 (alembic.ini의 [loggers] 등 반영)
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# 메타데이터 대상
target_metadata = SQLModel.metadata

# DB URL
ASYNC_DATABASE_URL = str(settings.SQLALCHEMY_DATABASE_URI)

if not config.get_main_option("sqlalchemy.url"):
    config.set_main_option("sqlalchemy.url", ASYNC_DATABASE_URL)


def run_migrations_offline() -> None:
    """오프라인 모드: DB 연결 없이 SQL 스크립트 생성"""
    url = config.get_main_option("sqlalchemy.url") or ASYNC_DATABASE_URL
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,   # 타입 변경 감지
        compare_server_default=True,  # server_default 비교
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """동기 컨텍스트에서 실제 마이그레이션 실행"""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """온라인 모드: 비동기 엔진으로 연결 → 동기 함수로 마이그레이션 실행"""
    connectable = create_async_engine(
        ASYNC_DATABASE_URL,
        poolclass=pool.NullPool,
        echo=True, 
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())