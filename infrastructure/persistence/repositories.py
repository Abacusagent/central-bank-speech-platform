# infrastructure/persistence/repositories.py

"""
Concrete SQLAlchemy Repository Implementations for Central Bank Speech Analysis Platform

Implements repository interfaces for robust, async, type-safe access to the persistence layer.
Supports transactional UnitOfWork pattern.

Author: Central Bank Speech Analysis Platform
Date: 2025
"""

import logging
from typing import List, Optional, Dict, Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, func, and_, desc
from sqlalchemy.orm import selectinload

from infrastructure.persistence.models import (
    InstitutionModel, SpeakerModel, SpeechModel,
    SentimentAnalysisModel, SpeechCollectionModel
)
from domain.entities import (
    Institution, CentralBankSpeaker, CentralBankSpeech, SentimentAnalysis
)
from domain.value_objects import DateRange, Url, ContentHash
from domain.repositories import (
    SpeechRepository, SpeakerRepository, InstitutionRepository,
    SpeechCollectionRepository, AnalysisRepository, UnitOfWork,
    RepositoryError, EntityNotFoundError, DuplicateEntityError
)

logger = logging.getLogger(__name__)

# -------------------
# Repository Helpers
# -------------------

def to_dict(model):
    """Helper to convert SQLAlchemy model to dict."""
    return {c.name: getattr(model, c.name) for c in model.__table__.columns}

# -------------------
# Institution Repository
# -------------------

class SqlAlchemyInstitutionRepository(InstitutionRepository):
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_code(self, code: str) -> Optional[Institution]:
        stmt = select(InstitutionModel).where(InstitutionModel.code == code)
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        if model:
            return Institution(
                code=model.code,
                name=model.name,
                country=model.country,
                institution_type=model.institution_type,
                established_date=model.established_date,
                website_url=model.website_url,
                description=model.description,
            )
        return None

    async def save(self, institution: Institution) -> None:
        existing = await self.session.get(InstitutionModel, institution.code)
        if existing:
            for field in ['name', 'country', 'institution_type', 'established_date', 'website_url', 'description']:
                setattr(existing, field, getattr(institution, field, None))
        else:
            model = InstitutionModel(**institution.__dict__)
            self.session.add(model)
        await self.session.flush()

# -------------------
# Speaker Repository
# -------------------

class SqlAlchemySpeakerRepository(SpeakerRepository):
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_id(self, speaker_id: UUID) -> Optional[CentralBankSpeaker]:
        stmt = select(SpeakerModel).where(SpeakerModel.id == speaker_id).options(selectinload(SpeakerModel.institution))
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        if model:
            return self._to_entity(model)
        return None

    async def find_by_name(self, name: str) -> List[CentralBankSpeaker]:
        stmt = select(SpeakerModel).where(SpeakerModel.name.ilike(f"%{name}%")).options(selectinload(SpeakerModel.institution))
        result = await self.session.execute(stmt)
        return [self._to_entity(m) for m in result.scalars().all()]

    async def save(self, speaker: CentralBankSpeaker) -> None:
        existing = await self.session.get(SpeakerModel, speaker.id)
        if existing:
            for field in ['name', 'role', 'institution_code', 'start_date', 'end_date', 'voting_member', 'biographical_notes', 'alternate_names']:
                setattr(existing, field, getattr(speaker, field, None))
        else:
            model = SpeakerModel(**speaker.__dict__)
            self.session.add(model)
        await self.session.flush()

    def _to_entity(self, model: SpeakerModel) -> CentralBankSpeaker:
        return CentralBankSpeaker(
            id=model.id,
            name=model.name,
            role=model.role,
            institution=None,  # Populated in orchestration if needed
            start_date=model.start_date,
            end_date=model.end_date,
            voting_member=model.voting_member,
            biographical_notes=model.biographical_notes,
            alternate_names=set(model.alternate_names) if model.alternate_names else set(),
        )

# -------------------
# Speech Repository
# -------------------

class SqlAlchemySpeechRepository(SpeechRepository):
    def __init__(self, session: AsyncSession):
        self.session = session

    async def save(self, speech: CentralBankSpeech) -> None:
        existing = await self.session.get(SpeechModel, speech.id)
        if existing:
            # Update fields
            for field in [
                'title', 'url', 'speech_date', 'institution_code', 'speaker_id', 'raw_text', 'cleaned_text',
                'word_count', 'content_hash_sha256', 'content_hash_md5', 'extraction_method', 'extraction_confidence',
                'speech_type', 'location', 'language', 'tags', 'status', 'processing_history',
                'hawkish_dovish_score', 'policy_stance', 'uncertainty_score', 'confidence_score',
                'analysis_timestamp', 'analyzer_version', 'validation_status', 'validation_confidence', 'validation_issues'
            ]:
                setattr(existing, field, getattr(speech, field, None))
        else:
            # Build ORM model from domain entity
            model = SpeechModel(**{k: getattr(speech, k, None) for k in SpeechModel.__table__.columns.keys()})
            self.session.add(model)
        await self.session.flush()

    async def get_by_id(self, speech_id: UUID) -> Optional[CentralBankSpeech]:
        stmt = select(SpeechModel).where(SpeechModel.id == speech_id).options(selectinload(SpeechModel.institution), selectinload(SpeechModel.speaker))
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        if model:
            return self._to_entity(model)
        return None

    async def get_by_url(self, url: str) -> Optional[CentralBankSpeech]:
        stmt = select(SpeechModel).where(SpeechModel.url == url).options(selectinload(SpeechModel.institution), selectinload(SpeechModel.speaker))
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        if model:
            return self._to_entity(model)
        return None

    def _to_entity(self, model: SpeechModel) -> CentralBankSpeech:
        # This is a simplified version; in production, map all fields and relationships
        speech = CentralBankSpeech(
            id=model.id,
            title=model.title,
            url=model.url,
            speech_date=model.speech_date,
            institution=None,  # To be hydrated at app layer
            speaker=None,      # To be hydrated at app layer
            status=model.status,
            tags=set(model.tags) if model.tags else set(),
            created_at=model.created_at,
            updated_at=model.updated_at,
        )
        # Add NLP fields, content fields etc. as needed
        return speech

# -------------------
# Sentiment Analysis Repository
# -------------------

class SqlAlchemyAnalysisRepository(AnalysisRepository):
    def __init__(self, session: AsyncSession):
        self.session = session

    async def save(self, analysis: SentimentAnalysis) -> None:
        # Upsert logic as appropriate; simplified for demonstration
        model = SentimentAnalysisModel(**{k: getattr(analysis, k, None) for k in SentimentAnalysisModel.__table__.columns.keys()})
        self.session.add(model)
        await self.session.flush()

# -------------------
# Speech Collection Repository (if needed)
# -------------------

class SqlAlchemySpeechCollectionRepository(SpeechCollectionRepository):
    def __init__(self, session: AsyncSession):
        self.session = session

    # Implement collection save, get, find as needed

# -------------------
# Unit of Work
# -------------------

class SqlAlchemyUnitOfWork(UnitOfWork):
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
        await self.session.commit()

    async def rollback(self):
        await self.session.rollback()
