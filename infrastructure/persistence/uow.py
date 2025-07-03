# infrastructure/persistence/uow.py

"""
Unit of Work implementation for Central Bank Speech Analysis Platform.

Coordinates transactional consistency for all repository operations,
using an async SQLAlchemy session.

Author: Central Bank Speech Analysis Platform
Date: 2025
"""

from sqlalchemy.ext.asyncio import AsyncSession

from infrastructure.persistence.repositories import (
    SqlAlchemySpeechRepository,
    SqlAlchemySpeakerRepository,
    SqlAlchemyInstitutionRepository,
    SqlAlchemyAnalysisRepository,
    SqlAlchemySpeechCollectionRepository
)
from domain.repositories import UnitOfWork

class SqlAlchemyUnitOfWork(UnitOfWork):
    """
    Coordinates all repositories and manages an async SQLAlchemy transaction scope.
    """

    def __init__(self, session: AsyncSession):
        self.session = session
        self.speeches = SqlAlchemySpeechRepository(session)
        self.speakers = SqlAlchemySpeakerRepository(session)
        self.institutions = SqlAlchemyInstitutionRepository(session)
        self.analyses = SqlAlchemyAnalysisRepository(session)
        self.collections = SqlAlchemySpeechCollectionRepository(session)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if exc_type:
            await self.rollback()
        else:
            await self.commit()

    async def commit(self):
        """Commit all staged operations to the database."""
        await self.session.commit()

    async def rollback(self):
        """Rollback all staged operations in the current transaction."""
        await self.session.rollback()
