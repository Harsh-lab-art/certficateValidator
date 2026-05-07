"""
backend/app/core/database.py

Async SQLAlchemy engine and session factory.
All DB operations use async/await via asyncpg driver.
"""
from __future__ import annotations
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import (
    AsyncSession, AsyncEngine,
    create_async_engine, async_sessionmaker,
)
from sqlalchemy.orm import DeclarativeBase
from backend.app.core.config import get_settings


class Base(DeclarativeBase):
    pass


_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker | None = None


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        cfg = get_settings()
        _engine = create_async_engine(
            cfg.database_url,
            pool_size=cfg.db_pool_size,
            max_overflow=cfg.db_max_overflow,
            echo=cfg.debug,
            future=True,
        )
    return _engine


def get_session_factory() -> async_sessionmaker:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
    return _session_factory


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency — yields a DB session per request."""
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def create_all_tables():
    """Create all tables (used in tests and first-run setup)."""
    from backend.app.models.schema import Base as SchemaBase
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(SchemaBase.metadata.create_all)


async def drop_all_tables():
    """Drop all tables (used in tests)."""
    from backend.app.models.schema import Base as SchemaBase
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(SchemaBase.metadata.drop_all)
