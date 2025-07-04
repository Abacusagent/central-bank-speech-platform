"""
Repository Implementations for Central Bank Speech Analysis Platform

This module provides concrete implementations of the repository interfaces using
SQLAlchemy and PostgreSQL. These implementations handle the technical details of
data persistence while maintaining the contracts defined in the domain layer.

Key Features:
- SQLAlchemy ORM with async operations for scalability
- PostgreSQL optimizations and full-text search
- Comprehensive error handling and logging
- Bulk operations and streaming for performance
- Connection pooling and transaction management
- Production-ready reliability and monitoring

Author: Central Bank Speech Analysis Platform
Date: 2025
Version: 2.0.0
"""

import asyncio
import logging
from datetime import date, datetime
from typing import List, Optional, Dict, Any, Set, AsyncIterator, Tuple
from uuid import UUID

from sqlalchemy import (
    select, delete, update, and_, or_, func, desc, asc, text,
    insert
)
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, AsyncEngine
from sqlalchemy.orm import selectinload, joinedload
from sqlalchemy.exc import SQLAlchemyError, IntegrityError

# Import models from separate module
from infrastructure.persistence.models import (
    Base, InstitutionModel, SpeakerModel, SpeechModel, 
    SentimentAnalysisModel, ProcessingLogModel
)

# Domain imports
from domain.entities import (
    CentralBankSpeech, CentralBankSpeaker, Institution, 
    SentimentAnalysis, SpeechStatus, PolicyStance, InstitutionType
)
from domain.value_objects import DateRange, Url, ContentHash, ConfidenceLevel
from domain.repositories import (
    SpeechRepository, SpeakerRepository, InstitutionRepository,
    AnalysisRepository, UnitOfWork, QuerySpecification,
    RepositoryError, EntityNotFoundError, DuplicateEntityError, ConcurrencyError
)
from interfaces.plugin_interfaces import SpeechMetadata, SpeechContent, ValidationResult

logger = logging.getLogger(__name__)


class SqlAlchemySpeechRepository(SpeechRepository):
    """
    SQLAlchemy implementation of SpeechRepository.
    
    Provides high-performance, async operations for speech persistence
    with comprehensive error handling and optimization features.
    """
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def save(self, speech: CentralBankSpeech) -> None:
        """Save a speech to the database."""
        try:
            # Check if speech already exists
            existing = await self.session.get(SpeechModel, speech.id)
            
            if existing:
                # Update existing speech
                await self._update_speech_model(existing, speech)
                self.logger.debug(f"Updated existing speech {speech.id}")
            else:
                # Create new speech
                model = await self._speech_to_model(speech)
                self.session.add(model)
                self.logger.debug(f"Added new speech {speech.id}")
            
            await self.session.flush()
            
        except IntegrityError as e:
            self.logger.error(f"Integrity error saving speech {speech.id}: {e}")
            raise DuplicateEntityError(f"Speech with ID {speech.id} already exists or violates constraints")
        except SQLAlchemyError as e:
            self.logger.error(f"Database error saving speech {speech.id}: {e}")
            raise RepositoryError(f"Failed to save speech: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error saving speech {speech.id}: {e}")
            raise RepositoryError(f"Unexpected error saving speech: {e}")
    
    async def save_batch(self, speeches: List[CentralBankSpeech]) -> None:
        """Save multiple speeches in a single batch operation."""
        if not speeches:
            return
        
        try:
            # Convert all speeches to models
            models = []
            for speech in speeches:
                model = await self._speech_to_model(speech)
                models.append(model)
            
            # Use bulk insert for better performance
            self.session.add_all(models)
            await self.session.flush()
            
            self.logger.info(f"Successfully saved batch of {len(speeches)} speeches")
            
        except IntegrityError as e:
            self.logger.error(f"Integrity error in batch save: {e}")
            raise DuplicateEntityError(f"One or more speeches violate constraints")
        except SQLAlchemyError as e:
            self.logger.error(f"Database error in batch save: {e}")
            raise RepositoryError(f"Failed to save speech batch: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error in batch save: {e}")
            raise RepositoryError(f"Unexpected error saving speech batch: {e}")
    
    async def get_by_id(self, speech_id: UUID) -> Optional[CentralBankSpeech]:
        """Retrieve a speech by its ID."""
        try:
            stmt = (
                select(SpeechModel)
                .options(
                    selectinload(SpeechModel.institution),
                    selectinload(SpeechModel.speaker).selectinload(SpeakerModel.institution)
                )
                .where(SpeechModel.id == speech_id)
            )
            
            result = await self.session.execute(stmt)
            model = result.scalar_one_or_none()
            
            if model:
                return await self._model_to_speech(model)
            return None
            
        except SQLAlchemyError as e:
            self.logger.error(f"Database error getting speech {speech_id}: {e}")
            raise RepositoryError(f"Failed to get speech: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error getting speech {speech_id}: {e}")
            raise RepositoryError(f"Unexpected error getting speech: {e}")
    
    async def get_by_url(self, url: str) -> Optional[CentralBankSpeech]:
        """Retrieve a speech by its source URL."""
        try:
            stmt = (
                select(SpeechModel)
                .options(
                    selectinload(SpeechModel.institution),
                    selectinload(SpeechModel.speaker).selectinload(SpeakerModel.institution)
                )
                .where(SpeechModel.url == url)
            )
            
            result = await self.session.execute(stmt)
            model = result.scalar_one_or_none()
            
            if model:
                return await self._model_to_speech(model)
            return None
            
        except SQLAlchemyError as e:
            self.logger.error(f"Database error getting speech by URL {url}: {e}")
            raise RepositoryError(f"Failed to get speech by URL: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error getting speech by URL {url}: {e}")
            raise RepositoryError(f"Unexpected error getting speech by URL: {e}")
    
    async def get_by_content_hash(self, content_hash: ContentHash) -> Optional[CentralBankSpeech]:
        """Retrieve a speech by its content hash."""
        try:
            stmt = (
                select(SpeechModel)
                .options(
                    selectinload(SpeechModel.institution),
                    selectinload(SpeechModel.speaker).selectinload(SpeakerModel.institution)
                )
                .where(SpeechModel.content_hash_sha256 == content_hash.sha256)
            )
            
            result = await self.session.execute(stmt)
            model = result.scalar_one_or_none()
            
            if model:
                return await self._model_to_speech(model)
            return None
            
        except SQLAlchemyError as e:
            self.logger.error(f"Database error getting speech by content hash: {e}")
            raise RepositoryError(f"Failed to get speech by content hash: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error getting speech by content hash: {e}")
            raise RepositoryError(f"Unexpected error getting speech by content hash: {e}")
    
    async def find_by_speaker(self, speaker: CentralBankSpeaker, limit: Optional[int] = None) -> List[CentralBankSpeech]:
        """Find speeches by speaker."""
        try:
            stmt = (
                select(SpeechModel)
                .options(
                    selectinload(SpeechModel.institution),
                    selectinload(SpeechModel.speaker).selectinload(SpeakerModel.institution)
                )
                .where(SpeechModel.speaker_id == speaker.id)
                .order_by(desc(SpeechModel.speech_date))
            )
            
            if limit:
                stmt = stmt.limit(limit)
            
            result = await self.session.execute(stmt)
            models = result.scalars().all()
            
            speeches = []
            for model in models:
                speech = await self._model_to_speech(model)
                speeches.append(speech)
            
            return speeches
            
        except SQLAlchemyError as e:
            self.logger.error(f"Database error finding speeches by speaker {speaker.id}: {e}")
            raise RepositoryError(f"Failed to find speeches by speaker: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error finding speeches by speaker {speaker.id}: {e}")
            raise RepositoryError(f"Unexpected error finding speeches by speaker: {e}")
    
    async def find_by_speaker_name(self, speaker_name: str, limit: Optional[int] = None) -> List[CentralBankSpeech]:
        """Find speeches by speaker name."""
        try:
            stmt = (
                select(SpeechModel)
                .join(SpeakerModel, SpeechModel.speaker_id == SpeakerModel.id)
                .options(
                    selectinload(SpeechModel.institution),
                    selectinload(SpeechModel.speaker).selectinload(SpeakerModel.institution)
                )
                .where(SpeakerModel.name.ilike(f"%{speaker_name}%"))
                .order_by(desc(SpeechModel.speech_date))
            )
            
            if limit:
                stmt = stmt.limit(limit)
            
            result = await self.session.execute(stmt)
            models = result.scalars().all()
            
            speeches = []
            for model in models:
                speech = await self._model_to_speech(model)
                speeches.append(speech)
            
            return speeches
            
        except SQLAlchemyError as e:
            self.logger.error(f"Database error finding speeches by speaker name {speaker_name}: {e}")
            raise RepositoryError(f"Failed to find speeches by speaker name: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error finding speeches by speaker name {speaker_name}: {e}")
            raise RepositoryError(f"Unexpected error finding speeches by speaker name: {e}")
    
    async def find_by_institution(self, institution: Institution, limit: Optional[int] = None) -> List[CentralBankSpeech]:
        """Find speeches by institution."""
        try:
            stmt = (
                select(SpeechModel)
                .options(
                    selectinload(SpeechModel.institution),
                    selectinload(SpeechModel.speaker).selectinload(SpeakerModel.institution)
                )
                .where(SpeechModel.institution_code == institution.code)
                .order_by(desc(SpeechModel.speech_date))
            )
            
            if limit:
                stmt = stmt.limit(limit)
            
            result = await self.session.execute(stmt)
            models = result.scalars().all()
            
            speeches = []
            for model in models:
                speech = await self._model_to_speech(model)
                speeches.append(speech)
            
            return speeches
            
        except SQLAlchemyError as e:
            self.logger.error(f"Database error finding speeches by institution {institution.code}: {e}")
            raise RepositoryError(f"Failed to find speeches by institution: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error finding speeches by institution {institution.code}: {e}")
            raise RepositoryError(f"Unexpected error finding speeches by institution: {e}")
    
    async def find_by_institution_code(self, institution_code: str, limit: Optional[int] = None) -> List[CentralBankSpeech]:
        """Find speeches by institution code."""
        try:
            stmt = (
                select(SpeechModel)
                .options(
                    selectinload(SpeechModel.institution),
                    selectinload(SpeechModel.speaker).selectinload(SpeakerModel.institution)
                )
                .where(SpeechModel.institution_code == institution_code)
                .order_by(desc(SpeechModel.speech_date))
            )
            
            if limit:
                stmt = stmt.limit(limit)
            
            result = await self.session.execute(stmt)
            models = result.scalars().all()
            
            speeches = []
            for model in models:
                speech = await self._model_to_speech(model)
                speeches.append(speech)
            
            return speeches
            
        except SQLAlchemyError as e:
            self.logger.error(f"Database error finding speeches by institution code {institution_code}: {e}")
            raise RepositoryError(f"Failed to find speeches by institution code: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error finding speeches by institution code {institution_code}: {e}")
            raise RepositoryError(f"Unexpected error finding speeches by institution code: {e}")
    
    async def find_by_date_range(self, date_range: DateRange, institution: Optional[Institution] = None) -> List[CentralBankSpeech]:
        """Find speeches within a date range."""
        try:
            stmt = (
                select(SpeechModel)
                .options(
                    selectinload(SpeechModel.institution),
                    selectinload(SpeechModel.speaker).selectinload(SpeakerModel.institution)
                )
                .where(
                    and_(
                        SpeechModel.speech_date >= date_range.start_date,
                        SpeechModel.speech_date <= date_range.end_date
                    )
                )
                .order_by(desc(SpeechModel.speech_date))
            )
            
            if institution:
                stmt = stmt.where(SpeechModel.institution_code == institution.code)
            
            result = await self.session.execute(stmt)
            models = result.scalars().all()
            
            speeches = []
            for model in models:
                speech = await self._model_to_speech(model)
                speeches.append(speech)
            
            return speeches
            
        except SQLAlchemyError as e:
            self.logger.error(f"Database error finding speeches by date range: {e}")
            raise RepositoryError(f"Failed to find speeches by date range: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error finding speeches by date range: {e}")
            raise RepositoryError(f"Unexpected error finding speeches by date range: {e}")
    
    async def find_by_status(self, status: SpeechStatus, limit: Optional[int] = None) -> List[CentralBankSpeech]:
        """Find speeches by status."""
        try:
            stmt = (
                select(SpeechModel)
                .options(
                    selectinload(SpeechModel.institution),
                    selectinload(SpeechModel.speaker).selectinload(SpeakerModel.institution)
                )
                .where(SpeechModel.status == status.value)
                .order_by(desc(SpeechModel.created_at))
            )
            
            if limit:
                stmt = stmt.limit(limit)
            
            result = await self.session.execute(stmt)
            models = result.scalars().all()
            
            speeches = []
            for model in models:
                speech = await self._model_to_speech(model)
                speeches.append(speech)
            
            return speeches
            
        except SQLAlchemyError as e:
            self.logger.error(f"Database error finding speeches by status {status}: {e}")
            raise RepositoryError(f"Failed to find speeches by status: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error finding speeches by status {status}: {e}")
            raise RepositoryError(f"Unexpected error finding speeches by status: {e}")
    
    async def find_by_policy_stance(self, stance: PolicyStance, date_range: Optional[DateRange] = None) -> List[CentralBankSpeech]:
        """Find speeches by policy stance."""
        try:
            stmt = (
                select(SpeechModel)
                .options(
                    selectinload(SpeechModel.institution),
                    selectinload(SpeechModel.speaker).selectinload(SpeakerModel.institution)
                )
                .where(SpeechModel.policy_stance == stance.value)
                .order_by(desc(SpeechModel.speech_date))
            )
            
            if date_range:
                stmt = stmt.where(
                    and_(
                        SpeechModel.speech_date >= date_range.start_date,
                        SpeechModel.speech_date <= date_range.end_date
                    )
                )
            
            result = await self.session.execute(stmt)
            models = result.scalars().all()
            
            speeches = []
            for model in models:
                speech = await self._model_to_speech(model)
                speeches.append(speech)
            
            return speeches
            
        except SQLAlchemyError as e:
            self.logger.error(f"Database error finding speeches by policy stance {stance}: {e}")
            raise RepositoryError(f"Failed to find speeches by policy stance: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error finding speeches by policy stance {stance}: {e}")
            raise RepositoryError(f"Unexpected error finding speeches by policy stance: {e}")
    
    async def find_by_sentiment_range(self, min_score: float, max_score: float, 
                                     institution: Optional[Institution] = None) -> List[CentralBankSpeech]:
        """Find speeches by sentiment score range."""
        try:
            stmt = (
                select(SpeechModel)
                .options(
                    selectinload(SpeechModel.institution),
                    selectinload(SpeechModel.speaker).selectinload(SpeakerModel.institution)
                )
                .where(
                    and_(
                        SpeechModel.hawkish_dovish_score >= min_score,
                        SpeechModel.hawkish_dovish_score <= max_score,
                        SpeechModel.hawkish_dovish_score.isnot(None)
                    )
                )
                .order_by(desc(SpeechModel.speech_date))
            )
            
            if institution:
                stmt = stmt.where(SpeechModel.institution_code == institution.code)
            
            result = await self.session.execute(stmt)
            models = result.scalars().all()
            
            speeches = []
            for model in models:
                speech = await self._model_to_speech(model)
                speeches.append(speech)
            
            return speeches
            
        except SQLAlchemyError as e:
            self.logger.error(f"Database error finding speeches by sentiment range: {e}")
            raise RepositoryError(f"Failed to find speeches by sentiment range: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error finding speeches by sentiment range: {e}")
            raise RepositoryError(f"Unexpected error finding speeches by sentiment range: {e}")
    
    async def find_by_specification(self, specification: QuerySpecification, 
                                   limit: Optional[int] = None) -> List[CentralBankSpeech]:
        """Find speeches by specification (in-memory filtering)."""
        try:
            # Load speeches and filter in memory for complex specifications
            # This is not optimal for large datasets but provides flexibility
            stmt = (
                select(SpeechModel)
                .options(
                    selectinload(SpeechModel.institution),
                    selectinload(SpeechModel.speaker).selectinload(SpeakerModel.institution)
                )
                .order_by(desc(SpeechModel.speech_date))
            )
            
            # Load extra rows to account for filtering
            if limit:
                stmt = stmt.limit(limit * 3)
            else:
                stmt = stmt.limit(1000)  # Reasonable default limit
            
            result = await self.session.execute(stmt)
            models = result.scalars().all()
            
            speeches = []
            for model in models:
                speech = await self._model_to_speech(model)
                if specification.is_satisfied_by(speech):
                    speeches.append(speech)
                    if limit and len(speeches) >= limit:
                        break
            
            return speeches
            
        except SQLAlchemyError as e:
            self.logger.error(f"Database error finding speeches by specification: {e}")
            raise RepositoryError(f"Failed to find speeches by specification: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error finding speeches by specification: {e}")
            raise RepositoryError(f"Unexpected error finding speeches by specification: {e}")
    
    async def stream_by_institution(self, institution: Institution) -> AsyncIterator[CentralBankSpeech]:
        """Stream speeches from an institution for memory-efficient processing."""
        try:
            stmt = (
                select(SpeechModel)
                .options(
                    selectinload(SpeechModel.institution),
                    selectinload(SpeechModel.speaker).selectinload(SpeakerModel.institution)
                )
                .where(SpeechModel.institution_code == institution.code)
                .order_by(SpeechModel.speech_date)
            )
            
            # Stream results in batches
            batch_size = 100
            offset = 0
            
            while True:
                batch_stmt = stmt.offset(offset).limit(batch_size)
                result = await self.session.execute(batch_stmt)
                models = result.scalars().all()
                
                if not models:
                    break
                
                for model in models:
                    speech = await self._model_to_speech(model)
                    yield speech
                
                offset += batch_size
                
                # Prevent infinite loops
                if len(models) < batch_size:
                    break
                    
        except SQLAlchemyError as e:
            self.logger.error(f"Database error streaming speeches by institution {institution.code}: {e}")
            raise RepositoryError(f"Failed to stream speeches by institution: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error streaming speeches by institution {institution.code}: {e}")
            raise RepositoryError(f"Unexpected error streaming speeches by institution: {e}")
    
    async def get_unprocessed_speeches(self, limit: Optional[int] = None) -> List[CentralBankSpeech]:
        """Get speeches that haven't been analyzed yet."""
        try:
            stmt = (
                select(SpeechModel)
                .options(
                    selectinload(SpeechModel.institution),
                    selectinload(SpeechModel.speaker).selectinload(SpeakerModel.institution)
                )
                .where(
                    and_(
                        SpeechModel.hawkish_dovish_score.is_(None),
                        SpeechModel.cleaned_text.isnot(None),
                        SpeechModel.cleaned_text != ""
                    )
                )
                .order_by(SpeechModel.created_at)
            )
            
            if limit:
                stmt = stmt.limit(limit)
            
            result = await self.session.execute(stmt)
            models = result.scalars().all()
            
            speeches = []
            for model in models:
                speech = await self._model_to_speech(model)
                speeches.append(speech)
            
            return speeches
            
        except SQLAlchemyError as e:
            self.logger.error(f"Database error getting unprocessed speeches: {e}")
            raise RepositoryError(f"Failed to get unprocessed speeches: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error getting unprocessed speeches: {e}")
            raise RepositoryError(f"Unexpected error getting unprocessed speeches: {e}")
    
    async def count_by_institution(self, institution: Institution) -> int:
        """Count speeches by institution."""
        try:
            stmt = select(func.count(SpeechModel.id)).where(SpeechModel.institution_code == institution.code)
            result = await self.session.execute(stmt)
            return result.scalar() or 0
            
        except SQLAlchemyError as e:
            self.logger.error(f"Database error counting speeches by institution {institution.code}: {e}")
            raise RepositoryError(f"Failed to count speeches by institution: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error counting speeches by institution {institution.code}: {e}")
            raise RepositoryError(f"Unexpected error counting speeches by institution: {e}")
    
    async def count_by_status(self, status: SpeechStatus) -> int:
        """Count speeches by status."""
        try:
            stmt = select(func.count(SpeechModel.id)).where(SpeechModel.status == status.value)
            result = await self.session.execute(stmt)
            return result.scalar() or 0
            
        except SQLAlchemyError as e:
            self.logger.error(f"Database error counting speeches by status {status}: {e}")
            raise RepositoryError(f"Failed to count speeches by status: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error counting speeches by status {status}: {e}")
            raise RepositoryError(f"Unexpected error counting speeches by status: {e}")
    
    async def count_by_date_range(self, date_range: DateRange, institution: Optional[Institution] = None) -> int:
        """Count speeches within a date range."""
        try:
            stmt = select(func.count(SpeechModel.id)).where(
                and_(
                    SpeechModel.speech_date >= date_range.start_date,
                    SpeechModel.speech_date <= date_range.end_date
                )
            )
            
            if institution:
                stmt = stmt.where(SpeechModel.institution_code == institution.code)
            
            result = await self.session.execute(stmt)
            return result.scalar() or 0
            
        except SQLAlchemyError as e:
            self.logger.error(f"Database error counting speeches by date range: {e}")
            raise RepositoryError(f"Failed to count speeches by date range: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error counting speeches by date range: {e}")
            raise RepositoryError(f"Unexpected error counting speeches by date range: {e}")
    
    async def get_all_institutions(self) -> List[Institution]:
        """Get all institutions that have speeches."""
        try:
            stmt = (
                select(InstitutionModel)
                .join(SpeechModel, InstitutionModel.code == SpeechModel.institution_code)
                .distinct()
            )
            
            result = await self.session.execute(stmt)
            models = result.scalars().all()
            
            institutions = []
            for model in models:
                institution = self._institution_model_to_entity(model)
                institutions.append(institution)
            
            return institutions
            
        except SQLAlchemyError as e:
            self.logger.error(f"Database error getting all institutions: {e}")
            raise RepositoryError(f"Failed to get all institutions: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error getting all institutions: {e}")
            raise RepositoryError(f"Unexpected error getting all institutions: {e}")
    
    async def get_date_range_for_institution(self, institution: Institution) -> Optional[DateRange]:
        """Get date range for institution's speeches."""
        try:
            stmt = select(
                func.min(SpeechModel.speech_date),
                func.max(SpeechModel.speech_date)
            ).where(SpeechModel.institution_code == institution.code)
            
            result = await self.session.execute(stmt)
            min_date, max_date = result.one()
            
            if min_date and max_date:
                return DateRange(start_date=min_date, end_date=max_date)
            return None
            
        except SQLAlchemyError as e:
            self.logger.error(f"Database error getting date range for institution {institution.code}: {e}")
            raise RepositoryError(f"Failed to get date range for institution: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error getting date range for institution {institution.code}: {e}")
            raise RepositoryError(f"Unexpected error getting date range for institution: {e}")
    
    async def delete(self, speech: CentralBankSpeech) -> None:
        """Delete a speech."""
        try:
            stmt = delete(SpeechModel).where(SpeechModel.id == speech.id)
            result = await self.session.execute(stmt)
            
            if result.rowcount == 0:
                raise EntityNotFoundError(f"Speech {speech.id} not found")
            
            self.logger.debug(f"Deleted speech {speech.id}")
            
        except EntityNotFoundError:
            raise
        except SQLAlchemyError as e:
            self.logger.error(f"Database error deleting speech {speech.id}: {e}")
            raise RepositoryError(f"Failed to delete speech: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error deleting speech {speech.id}: {e}")
            raise RepositoryError(f"Unexpected error deleting speech: {e}")
    
    async def exists(self, speech_id: UUID) -> bool:
        """Check if speech exists."""
        try:
            stmt = select(func.count(SpeechModel.id)).where(SpeechModel.id == speech_id)
            result = await self.session.execute(stmt)
            return (result.scalar() or 0) > 0
            
        except SQLAlchemyError as e:
            self.logger.error(f"Database error checking speech existence {speech_id}: {e}")
            raise RepositoryError(f"Failed to check speech existence: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error checking speech existence {speech_id}: {e}")
            raise RepositoryError(f"Unexpected error checking speech existence: {e}")
    
    async def exists_by_url(self, url: str) -> bool:
        """Check if a speech exists by URL."""
        try:
            stmt = select(func.count(SpeechModel.id)).where(SpeechModel.url == url)
            result = await self.session.execute(stmt)
            return (result.scalar() or 0) > 0
            
        except SQLAlchemyError as e:
            self.logger.error(f"Database error checking speech existence by URL {url}: {e}")
            raise RepositoryError(f"Failed to check speech existence by URL: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error checking speech existence by URL {url}: {e}")
            raise RepositoryError(f"Unexpected error checking speech existence by URL: {e}")
    
    async def get_processing_metrics(self, date_range: Optional[DateRange] = None) -> Dict[str, Any]:
        """Get processing metrics for speeches."""
        try:
            # Base condition
            conditions = []
            if date_range:
                conditions.extend([
                    SpeechModel.speech_date >= date_range.start_date,
                    SpeechModel.speech_date <= date_range.end_date
                ])
            
            base_where = and_(*conditions) if conditions else text("1=1")
            
            # Total count
            total_stmt = select(func.count(SpeechModel.id)).where(base_where)
            total_result = await self.session.execute(total_stmt)
            total_count = total_result.scalar() or 0
            
            # Status distribution
            status_stmt = (
                select(SpeechModel.status, func.count(SpeechModel.id))
                .where(base_where)
                .group_by(SpeechModel.status)
            )
            status_result = await self.session.execute(status_stmt)
            status_distribution = dict(status_result.all())
            
            # Institution distribution
            institution_stmt = (
                select(SpeechModel.institution_code, func.count(SpeechModel.id))
                .where(base_where)
                .group_by(SpeechModel.institution_code)
            )
            institution_result = await self.session.execute(institution_stmt)
            institution_distribution = dict(institution_result.all())
            
            # Analysis metrics
            analyzed_stmt = select(func.count(SpeechModel.id)).where(
                and_(base_where, SpeechModel.hawkish_dovish_score.isnot(None))
            )
            analyzed_result = await self.session.execute(analyzed_stmt)
            analyzed_count = analyzed_result.scalar() or 0
            
            return {
                'total_speeches': total_count,
                'analyzed_speeches': analyzed_count,
                'analysis_percentage': (analyzed_count / total_count * 100) if total_count > 0 else 0,
                'status_distribution': status_distribution,
                'institution_distribution': institution_distribution,
                'date_range': {
                    'start_date': date_range.start_date.isoformat() if date_range else None,
                    'end_date': date_range.end_date.isoformat() if date_range else None
                }
            }
            
        except SQLAlchemyError as e:
            self.logger.error(f"Database error getting processing metrics: {e}")
            raise RepositoryError(f"Failed to get processing metrics: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error getting processing metrics: {e}")
            raise RepositoryError(f"Unexpected error getting processing metrics: {e}")
    
    # Helper methods for model conversion
    
    async def _speech_to_model(self, speech: CentralBankSpeech) -> SpeechModel:
        """Convert speech entity to database model."""
        model = SpeechModel(
            id=speech.id,
            status=speech.status.value,
            tags=list(speech.tags) if speech.tags else None,
            processing_history=speech.processing_history,
            created_at=speech.created_at,
            updated_at=speech.updated_at
        )
        
        # Set metadata fields if available
        if speech.metadata:
            model.title = speech.metadata.title
            model.url = speech.metadata.url
            model.speech_date = speech.metadata.date
            model.location = speech.metadata.location
            model.language = speech.metadata.language
            model.speech_type = speech.metadata.speech_type.value if speech.metadata.speech_type else None
        
        # Set institutional context
        if speech.institution:
            model.institution_code = speech.institution.code
        elif speech.metadata:
            model.institution_code = speech.metadata.institution_code
        
        if speech.speaker:
            model.speaker_id = speech.speaker.id
        
        # Set content fields if available
        if speech.content:
            model.raw_text = speech.content.raw_text
            model.cleaned_text = speech.content.cleaned_text
            model.word_count = speech.content.word_count
            model.extraction_method = speech.content.extraction_method
            model.extraction_confidence = speech.content.confidence_score
            
            # Generate content hash if not present
            if speech.content.cleaned_text:
                content_hash = ContentHash.from_content(speech.content.cleaned_text)
                model.content_hash_sha256 = content_hash.sha256
                model.content_hash_md5 = content_hash.md5
        
        # Set sentiment analysis fields if available
        if speech.sentiment_analysis:
            model.hawkish_dovish_score = speech.sentiment_analysis.hawkish_dovish_score
            model.policy_stance = speech.sentiment_analysis.policy_stance.value
            model.uncertainty_score = speech.sentiment_analysis.uncertainty_score
            model.confidence_score = speech.sentiment_analysis.confidence_score
            model.analysis_timestamp = speech.sentiment_analysis.analysis_timestamp
            model.analyzer_version = speech.sentiment_analysis.analyzer_version
            model.topic_classifications = speech.sentiment_analysis.topic_classifications
        
        # Set validation fields if available
        if speech.validation_result:
            model.validation_status = speech.validation_result.status.value
            model.validation_confidence = speech.validation_result.confidence
            model.validation_issues = speech.validation_result.issues
        
        return model
    
    async def _update_speech_model(self, model: SpeechModel, speech: CentralBankSpeech) -> None:
        """Update existing speech model with entity data."""
        # Update basic fields
        model.status = speech.status.value
        model.tags = list(speech.tags) if speech.tags else None
        model.processing_history = speech.processing_history
        model.updated_at = speech.updated_at
        
        # Update metadata fields if available
        if speech.metadata:
            model.title = speech.metadata.title
            model.url = speech.metadata.url
            model.speech_date = speech.metadata.date
            model.location = speech.metadata.location
            model.language = speech.metadata.language
            model.speech_type = speech.metadata.speech_type.value if speech.metadata.speech_type else None
        
        # Update institutional context
        if speech.institution:
            model.institution_code = speech.institution.code
        elif speech.metadata:
            model.institution_code = speech.metadata.institution_code
        
        if speech.speaker:
            model.speaker_id = speech.speaker.id
        
        # Update content fields if available
        if speech.content:
            model.raw_text = speech.content.raw_text
            model.cleaned_text = speech.content.cleaned_text
            model.word_count = speech.content.word_count
            model.extraction_method = speech.content.extraction_method
            model.extraction_confidence = speech.content.confidence_score
            
            # Update content hash
            if speech.content.cleaned_text:
                content_hash = ContentHash.from_content(speech.content.cleaned_text)
                model.content_hash_sha256 = content_hash.sha256
                model.content_hash_md5 = content_hash.md5
        
        # Update sentiment analysis fields if available
        if speech.sentiment_analysis:
            model.hawkish_dovish_score = speech.sentiment_analysis.hawkish_dovish_score
            model.policy_stance = speech.sentiment_analysis.policy_stance.value
            model.uncertainty_score = speech.sentiment_analysis.uncertainty_score
            model.confidence_score = speech.sentiment_analysis.confidence_score
            model.analysis_timestamp = speech.sentiment_analysis.analysis_timestamp
            model.analyzer_version = speech.sentiment_analysis.analyzer_version
            model.topic_classifications = speech.sentiment_analysis.topic_classifications
        
        # Update validation fields if available
        if speech.validation_result:
            model.validation_status = speech.validation_result.status.value
            model.validation_confidence = speech.validation_result.confidence
            model.validation_issues = speech.validation_result.issues
    
    async def _model_to_speech(self, model: SpeechModel) -> CentralBankSpeech:
        """Convert database model to speech entity."""
        speech = CentralBankSpeech(
            id=model.id,
            status=SpeechStatus(model.status),
            tags=set(model.tags) if model.tags else set(),
            processing_history=model.processing_history or [],
            created_at=model.created_at,
            updated_at=model.updated_at
        )
        
        # Set metadata if available
        if model.title or model.url or model.speech_date:
            from interfaces.plugin_interfaces import SpeechType
            speech_type = SpeechType.FORMAL_SPEECH
            if model.speech_type:
                try:
                    speech_type = SpeechType(model.speech_type)
                except ValueError:
                    speech_type = SpeechType.FORMAL_SPEECH
            
            speech.metadata = SpeechMetadata(
                url=model.url or "",
                title=model.title or "",
                speaker_name=model.speaker.name if model.speaker else "",
                date=model.speech_date or date.today(),
                institution_code=model.institution_code or "",
                speech_type=speech_type,
                location=model.location,
                language=model.language or "en"
            )
        
        # Set content if available
        if model.raw_text or model.cleaned_text:
            speech.content = SpeechContent(
                raw_text=model.raw_text or "",
                cleaned_text=model.cleaned_text or "",
                extraction_method=model.extraction_method or "",
                confidence_score=model.extraction_confidence or 0.0,
                word_count=model.word_count or 0,
                extraction_timestamp=model.created_at
            )
        
        # Set institution if available
        if model.institution:
            speech.institution = self._institution_model_to_entity(model.institution)
        
        # Set speaker if available
        if model.speaker:
            speech.speaker = self._speaker_model_to_entity(model.speaker)
        
        # Set sentiment analysis if available
        if model.hawkish_dovish_score is not None:
            speech.sentiment_analysis = SentimentAnalysis(
                hawkish_dovish_score=model.hawkish_dovish_score,
                policy_stance=PolicyStance(model.policy_stance) if model.policy_stance else PolicyStance.NEUTRAL,
                uncertainty_score=model.uncertainty_score or 0.0,
                confidence_score=model.confidence_score or 0.0,
                analysis_timestamp=model.analysis_timestamp or datetime.utcnow(),
                analyzer_version=model.analyzer_version or "unknown",
                topic_classifications=model.topic_classifications or []
            )
        
        # Set validation result if available
        if model.validation_status:
            from interfaces.plugin_interfaces import ValidationStatus, ValidationResult
            speech.validation_result = ValidationResult(
                status=ValidationStatus(model.validation_status),
                confidence=model.validation_confidence or 0.0,
                issues=model.validation_issues or [],
                metadata={}
            )
        
        return speech
    
    def _institution_model_to_entity(self, model: InstitutionModel) -> Institution:
        """Convert institution model to entity."""
        return Institution(
            code=model.code,
            name=model.name,
            country=model.country,
            institution_type=InstitutionType(model.institution_type),
            established_date=model.established_date,
            website_url=model.website_url,
            description=model.description
        )
    
    def _speaker_model_to_entity(self, model: SpeakerModel) -> CentralBankSpeaker:
        """Convert speaker model to entity."""
        speaker = CentralBankSpeaker(
            id=model.id,
            name=model.name,
            role=model.role,
            start_date=model.start_date,
            end_date=model.end_date,
            voting_member=model.voting_member,
            biographical_notes=model.biography
        )
        
        # Set alternate names if available
        if model.alternate_names:
            speaker.alternate_names = set(model.alternate_names)
        
        # Set institution if available
        if model.institution:
            speaker.institution = self._institution_model_to_entity(model.institution)
        
        return speaker


class SqlAlchemySpeakerRepository(SpeakerRepository):
    """
    SQLAlchemy implementation of SpeakerRepository.
    
    Provides comprehensive speaker management with fuzzy matching,
    analytics, and relationship management.
    """
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def save(self, speaker: CentralBankSpeaker) -> None:
        """Save a speaker to the database."""
        try:
            # Check if speaker already exists
            existing = await self.session.get(SpeakerModel, speaker.id)
            
            if existing:
                # Update existing speaker
                await self._update_speaker_model(existing, speaker)
                self.logger.debug(f"Updated existing speaker {speaker.id}")
            else:
                # Create new speaker
                model = self._speaker_to_model(speaker)
                self.session.add(model)
                self.logger.debug(f"Added new speaker {speaker.id}")
            
            await self.session.flush()
            
        except IntegrityError as e:
            self.logger.error(f"Integrity error saving speaker {speaker.id}: {e}")
            raise DuplicateEntityError(f"Speaker with ID {speaker.id} violates constraints")
        except SQLAlchemyError as e:
            self.logger.error(f"Database error saving speaker {speaker.id}: {e}")
            raise RepositoryError(f"Failed to save speaker: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error saving speaker {speaker.id}: {e}")
            raise RepositoryError(f"Unexpected error saving speaker: {e}")
    
    async def save_batch(self, speakers: List[CentralBankSpeaker]) -> None:
        """Save multiple speakers in a single batch operation."""
        if not speakers:
            return
        
        try:
            # Convert all speakers to models
            models = []
            for speaker in speakers:
                model = self._speaker_to_model(speaker)
                models.append(model)
            
            # Use bulk insert for better performance
            self.session.add_all(models)
            await self.session.flush()
            
            self.logger.info(f"Successfully saved batch of {len(speakers)} speakers")
            
        except IntegrityError as e:
            self.logger.error(f"Integrity error in speaker batch save: {e}")
            raise DuplicateEntityError(f"One or more speakers violate constraints")
        except SQLAlchemyError as e:
            self.logger.error(f"Database error in speaker batch save: {e}")
            raise RepositoryError(f"Failed to save speaker batch: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error in speaker batch save: {e}")
            raise RepositoryError(f"Unexpected error saving speaker batch: {e}")
    
    async def get_by_id(self, speaker_id: UUID) -> Optional[CentralBankSpeaker]:
        """Retrieve a speaker by ID."""
        try:
            stmt = (
                select(SpeakerModel)
                .options(selectinload(SpeakerModel.institution))
                .where(SpeakerModel.id == speaker_id)
            )
            
            result = await self.session.execute(stmt)
            model = result.scalar_one_or_none()
            
            if model:
                return self._speaker_model_to_entity(model)
            return None
            
        except SQLAlchemyError as e:
            self.logger.error(f"Database error getting speaker {speaker_id}: {e}")
            raise RepositoryError(f"Failed to get speaker: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error getting speaker {speaker_id}: {e}")
            raise RepositoryError(f"Unexpected error getting speaker: {e}")
    
    async def find_by_name(self, name: str) -> List[CentralBankSpeaker]:
        """Find speakers by name."""
        try:
            stmt = (
                select(SpeakerModel)
                .options(selectinload(SpeakerModel.institution))
                .where(SpeakerModel.name.ilike(f"%{name}%"))
                .order_by(SpeakerModel.name)
            )
            
            result = await self.session.execute(stmt)
            models = result.scalars().all()
            
            speakers = []
            for model in models:
                speaker = self._speaker_model_to_entity(model)
                speakers.append(speaker)
            
            return speakers
            
        except SQLAlchemyError as e:
            self.logger.error(f"Database error finding speakers by name {name}: {e}")
            raise RepositoryError(f"Failed to find speakers by name: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error finding speakers by name {name}: {e}")
            raise RepositoryError(f"Unexpected error finding speakers by name: {e}")
    
    async def find_by_name_fuzzy(self, name: str, threshold: float = 0.8) -> List[CentralBankSpeaker]:
        """Find speakers using fuzzy name matching."""
        try:
            # Use PostgreSQL similarity functions for fuzzy matching
            stmt = (
                select(SpeakerModel)
                .options(selectinload(SpeakerModel.institution))
                .where(
                    or_(
                        SpeakerModel.name.ilike(f"%{name}%"),
                        func.similarity(SpeakerModel.name, name) > threshold
                    )
                )
                .order_by(func.similarity(SpeakerModel.name, name).desc())
            )
            
            result = await self.session.execute(stmt)
            models = result.scalars().all()
            
            speakers = []
            for model in models:
                speaker = self._speaker_model_to_entity(model)
                speakers.append(speaker)
            
            return speakers
            
        except SQLAlchemyError as e:
            self.logger.error(f"Database error in fuzzy speaker search for {name}: {e}")
            # Fallback to simple ILIKE search
            return await self.find_by_name(name)
        except Exception as e:
            self.logger.error(f"Unexpected error in fuzzy speaker search for {name}: {e}")
            raise RepositoryError(f"Unexpected error in fuzzy speaker search: {e}")
    
    async def find_by_institution(self, institution: Institution) -> List[CentralBankSpeaker]:
        """Find speakers by institution."""
        try:
            stmt = (
                select(SpeakerModel)
                .options(selectinload(SpeakerModel.institution))
                .where(SpeakerModel.institution_code == institution.code)
                .order_by(SpeakerModel.name)
            )
            
            result = await self.session.execute(stmt)
            models = result.scalars().all()
            
            speakers = []
            for model in models:
                speaker = self._speaker_model_to_entity(model)
                speakers.append(speaker)
            
            return speakers
            
        except SQLAlchemyError as e:
            self.logger.error(f"Database error finding speakers by institution {institution.code}: {e}")
            raise RepositoryError(f"Failed to find speakers by institution: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error finding speakers by institution {institution.code}: {e}")
            raise RepositoryError(f"Unexpected error finding speakers by institution: {e}")
    
    async def find_current_speakers(self, institution: Optional[Institution] = None) -> List[CentralBankSpeaker]:
        """Find currently active speakers."""
        try:
            stmt = (
                select(SpeakerModel)
                .options(selectinload(SpeakerModel.institution))
                .where(
                    or_(
                        SpeakerModel.end_date.is_(None),
                        SpeakerModel.end_date >= date.today()
                    )
                )
                .order_by(SpeakerModel.name)
            )
            
            if institution:
                stmt = stmt.where(SpeakerModel.institution_code == institution.code)
            
            result = await self.session.execute(stmt)
            models = result.scalars().all()
            
            speakers = []
            for model in models:
                speaker = self._speaker_model_to_entity(model)
                speakers.append(speaker)
            
            return speakers
            
        except SQLAlchemyError as e:
            self.logger.error(f"Database error finding current speakers: {e}")
            raise RepositoryError(f"Failed to find current speakers: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error finding current speakers: {e}")
            raise RepositoryError(f"Unexpected error finding current speakers: {e}")
    
    async def find_by_role(self, role: str, institution: Optional[Institution] = None) -> List[CentralBankSpeaker]:
        """Find speakers by role."""
        try:
            stmt = (
                select(SpeakerModel)
                .options(selectinload(SpeakerModel.institution))
                .where(SpeakerModel.role.ilike(f"%{role}%"))
                .order_by(SpeakerModel.name)
            )
            
            if institution:
                stmt = stmt.where(SpeakerModel.institution_code == institution.code)
            
            result = await self.session.execute(stmt)
            models = result.scalars().all()
            
            speakers = []
            for model in models:
                speaker = self._speaker_model_to_entity(model)
                speakers.append(speaker)
            
            return speakers
            
        except SQLAlchemyError as e:
            self.logger.error(f"Database error finding speakers by role {role}: {e}")
            raise RepositoryError(f"Failed to find speakers by role: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error finding speakers by role {role}: {e}")
            raise RepositoryError(f"Unexpected error finding speakers by role: {e}")
    
    async def find_voting_members(self, institution: Institution) -> List[CentralBankSpeaker]:
        """Find voting members at an institution."""
        try:
            stmt = (
                select(SpeakerModel)
                .options(selectinload(SpeakerModel.institution))
                .where(
                    and_(
                        SpeakerModel.institution_code == institution.code,
                        SpeakerModel.voting_member == True
                    )
                )
                .order_by(SpeakerModel.name)
            )
            
            result = await self.session.execute(stmt)
            models = result.scalars().all()
            
            speakers = []
            for model in models:
                speaker = self._speaker_model_to_entity(model)
                speakers.append(speaker)
            
            return speakers
            
        except SQLAlchemyError as e:
            self.logger.error(f"Database error finding voting members for {institution.code}: {e}")
            raise RepositoryError(f"Failed to find voting members: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error finding voting members for {institution.code}: {e}")
            raise RepositoryError(f"Unexpected error finding voting members: {e}")
    
    async def search_by_alternate_names(self, name: str) -> List[CentralBankSpeaker]:
        """Search speakers by alternate names."""
        try:
            # Use PostgreSQL array operations to search alternate names
            stmt = (
                select(SpeakerModel)
                .options(selectinload(SpeakerModel.institution))
                .where(
                    or_(
                        SpeakerModel.alternate_names.any(name),
                        func.array_to_string(SpeakerModel.alternate_names, ' ').ilike(f"%{name}%")
                    )
                )
                .order_by(SpeakerModel.name)
            )
            
            result = await self.session.execute(stmt)
            models = result.scalars().all()
            
            speakers = []
            for model in models:
                speaker = self._speaker_model_to_entity(model)
                speakers.append(speaker)
            
            return speakers
            
        except SQLAlchemyError as e:
            self.logger.error(f"Database error searching speakers by alternate names {name}: {e}")
            raise RepositoryError(f"Failed to search speakers by alternate names: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error searching speakers by alternate names {name}: {e}")
            raise RepositoryError(f"Unexpected error searching speakers by alternate names: {e}")
    
    async def get_speaker_statistics(self, speaker: CentralBankSpeaker) -> Dict[str, Any]:
        """Get statistics for a speaker."""
        try:
            # Get speech count and date range
            stmt = (
                select(
                    func.count(SpeechModel.id),
                    func.min(SpeechModel.speech_date),
                    func.max(SpeechModel.speech_date),
                    func.avg(SpeechModel.word_count),
                    func.avg(SpeechModel.hawkish_dovish_score)
                )
                .where(SpeechModel.speaker_id == speaker.id)
            )
            
            result = await self.session.execute(stmt)
            speech_count, first_date, last_date, avg_words, avg_sentiment = result.one()
            
            # Get stance distribution
            stance_stmt = (
                select(SpeechModel.policy_stance, func.count(SpeechModel.id))
                .where(
                    and_(
                        SpeechModel.speaker_id == speaker.id,
                        SpeechModel.policy_stance.isnot(None)
                    )
                )
                .group_by(SpeechModel.policy_stance)
            )
            stance_result = await self.session.execute(stance_stmt)
            stance_distribution = dict(stance_result.all())
            
            return {
                'total_speeches': speech_count or 0,
                'first_speech_date': first_date.isoformat() if first_date else None,
                'last_speech_date': last_date.isoformat() if last_date else None,
                'average_word_count': round(avg_words, 2) if avg_words else 0,
                'average_sentiment_score': round(avg_sentiment, 3) if avg_sentiment else None,
                'stance_distribution': stance_distribution
            }
            
        except SQLAlchemyError as e:
            self.logger.error(f"Database error getting speaker statistics for {speaker.id}: {e}")
            raise RepositoryError(f"Failed to get speaker statistics: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error getting speaker statistics for {speaker.id}: {e}")
            raise RepositoryError(f"Unexpected error getting speaker statistics: {e}")
    
    async def get_speakers_by_speech_count(self, min_speeches: int = 1) -> List[Tuple[CentralBankSpeaker, int]]:
        """Get speakers with their speech counts."""
        try:
            stmt = (
                select(SpeakerModel, func.count(SpeechModel.id).label('speech_count'))
                .outerjoin(SpeechModel, SpeakerModel.id == SpeechModel.speaker_id)
                .options(selectinload(SpeakerModel.institution))
                .group_by(SpeakerModel.id)
                .having(func.count(SpeechModel.id) >= min_speeches)
                .order_by(func.count(SpeechModel.id).desc())
            )
            
            result = await self.session.execute(stmt)
            rows = result.all()
            
            speakers_with_counts = []
            for model, count in rows:
                speaker = self._speaker_model_to_entity(model)
                speakers_with_counts.append((
                    speaker, count
                ))
            
            return speakers_with_counts
            
        except SQLAlchemyError as e:
            self.logger.error(f"Database error getting speakers by speech count: {e}")
            raise RepositoryError(f"Failed to get speakers by speech count: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error getting speakers by speech count: {e}")
            raise RepositoryError(f"Unexpected error getting speakers by speech count: {e}")
    
    async def delete(self, speaker: CentralBankSpeaker) -> None:
        """Delete a speaker."""
        try:
            stmt = delete(SpeakerModel).where(SpeakerModel.id == speaker.id)
            result = await self.session.execute(stmt)
            
            if result.rowcount == 0:
                raise EntityNotFoundError(f"Speaker {speaker.id} not found")
            
            self.logger.debug(f"Deleted speaker {speaker.id}")
            
        except EntityNotFoundError:
            raise
        except SQLAlchemyError as e:
            self.logger.error(f"Database error deleting speaker {speaker.id}: {e}")
            raise RepositoryError(f"Failed to delete speaker: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error deleting speaker {speaker.id}: {e}")
            raise RepositoryError(f"Unexpected error deleting speaker: {e}")
    
    async def exists(self, speaker_id: UUID) -> bool:
        """Check if speaker exists."""
        try:
            stmt = select(func.count(SpeakerModel.id)).where(SpeakerModel.id == speaker_id)
            result = await self.session.execute(stmt)
            return (result.scalar() or 0) > 0
            
        except SQLAlchemyError as e:
            self.logger.error(f"Database error checking speaker existence {speaker_id}: {e}")
            raise RepositoryError(f"Failed to check speaker existence: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error checking speaker existence {speaker_id}: {e}")
            raise RepositoryError(f"Unexpected error checking speaker existence: {e}")
    
    # Helper methods for model conversion
    
    def _speaker_to_model(self, speaker: CentralBankSpeaker) -> SpeakerModel:
        """Convert speaker entity to database model."""
        model = SpeakerModel(
            id=speaker.id,
            name=speaker.name,
            role=speaker.role,
            start_date=speaker.start_date,
            end_date=speaker.end_date,
            voting_member=speaker.voting_member,
            biography=speaker.biographical_notes,
            alternate_names=list(speaker.alternate_names) if speaker.alternate_names else None
        )
        
        # Set institution if available
        if speaker.institution:
            model.institution_code = speaker.institution.code
        
        return model
    
    async def _update_speaker_model(self, model: SpeakerModel, speaker: CentralBankSpeaker) -> None:
        """Update existing speaker model with entity data."""
        model.name = speaker.name
        model.role = speaker.role
        model.start_date = speaker.start_date
        model.end_date = speaker.end_date
        model.voting_member = speaker.voting_member
        model.biography = speaker.biographical_notes
        model.alternate_names = list(speaker.alternate_names) if speaker.alternate_names else None
        
        # Update institution if available
        if speaker.institution:
            model.institution_code = speaker.institution.code


class SqlAlchemyInstitutionRepository(InstitutionRepository):
    """
    SQLAlchemy implementation of InstitutionRepository.
    
    Provides institution management with comprehensive analytics
    and relationship tracking.
    """
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def save(self, institution: Institution) -> None:
        """Save an institution to the database."""
        try:
            # Check if institution already exists
            existing = await self.session.get(InstitutionModel, institution.code)
            
            if existing:
                # Update existing institution
                self._update_institution_model(existing, institution)
                self.logger.debug(f"Updated existing institution {institution.code}")
            else:
                # Create new institution
                model = self._institution_to_model(institution)
                self.session.add(model)
                self.logger.debug(f"Added new institution {institution.code}")
            
            await self.session.flush()
            
        except IntegrityError as e:
            self.logger.error(f"Integrity error saving institution {institution.code}: {e}")
            raise DuplicateEntityError(f"Institution with code {institution.code} violates constraints")
        except SQLAlchemyError as e:
            self.logger.error(f"Database error saving institution {institution.code}: {e}")
            raise RepositoryError(f"Failed to save institution: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error saving institution {institution.code}: {e}")
            raise RepositoryError(f"Unexpected error saving institution: {e}")
    
    async def get_by_code(self, code: str) -> Optional[Institution]:
        """Retrieve an institution by code."""
        try:
            result = await self.session.get(InstitutionModel, code)
            
            if result:
                return self._institution_model_to_entity(result)
            return None
            
        except SQLAlchemyError as e:
            self.logger.error(f"Database error getting institution {code}: {e}")
            raise RepositoryError(f"Failed to get institution: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error getting institution {code}: {e}")
            raise RepositoryError(f"Unexpected error getting institution: {e}")
    
    async def get_all(self) -> List[Institution]:
        """Get all institutions."""
        try:
            stmt = select(InstitutionModel).order_by(InstitutionModel.name)
            result = await self.session.execute(stmt)
            models = result.scalars().all()
            
            institutions = []
            for model in models:
                institution = self._institution_model_to_entity(model)
                institutions.append(institution)
            
            return institutions
            
        except SQLAlchemyError as e:
            self.logger.error(f"Database error getting all institutions: {e}")
            raise RepositoryError(f"Failed to get all institutions: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error getting all institutions: {e}")
            raise RepositoryError(f"Unexpected error getting all institutions: {e}")
    
    async def find_by_country(self, country: str) -> List[Institution]:
        """Find institutions by country."""
        try:
            stmt = (
                select(InstitutionModel)
                .where(InstitutionModel.country.ilike(f"%{country}%"))
                .order_by(InstitutionModel.name)
            )
            
            result = await self.session.execute(stmt)
            models = result.scalars().all()
            
            institutions = []
            for model in models:
                institution = self._institution_model_to_entity(model)
                institutions.append(institution)
            
            return institutions
            
        except SQLAlchemyError as e:
            self.logger.error(f"Database error finding institutions by country {country}: {e}")
            raise RepositoryError(f"Failed to find institutions by country: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error finding institutions by country {country}: {e}")
            raise RepositoryError(f"Unexpected error finding institutions by country: {e}")
    
    async def find_by_type(self, institution_type: InstitutionType) -> List[Institution]:
        """Find institutions by type."""
        try:
            stmt = (
                select(InstitutionModel)
                .where(InstitutionModel.institution_type == institution_type.value)
                .order_by(InstitutionModel.name)
            )
            
            result = await self.session.execute(stmt)
            models = result.scalars().all()
            
            institutions = []
            for model in models:
                institution = self._institution_model_to_entity(model)
                institutions.append(institution)
            
            return institutions
            
        except SQLAlchemyError as e:
            self.logger.error(f"Database error finding institutions by type {institution_type}: {e}")
            raise RepositoryError(f"Failed to find institutions by type: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error finding institutions by type {institution_type}: {e}")
            raise RepositoryError(f"Unexpected error finding institutions by type: {e}")
    
    async def delete(self, institution: Institution) -> None:
        """Delete an institution."""
        try:
            stmt = delete(InstitutionModel).where(InstitutionModel.code == institution.code)
            result = await self.session.execute(stmt)
            
            if result.rowcount == 0:
                raise EntityNotFoundError(f"Institution {institution.code} not found")
            
            self.logger.debug(f"Deleted institution {institution.code}")
            
        except EntityNotFoundError:
            raise
        except SQLAlchemyError as e:
            self.logger.error(f"Database error deleting institution {institution.code}: {e}")
            raise RepositoryError(f"Failed to delete institution: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error deleting institution {institution.code}: {e}")
            raise RepositoryError(f"Unexpected error deleting institution: {e}")
    
    async def exists(self, code: str) -> bool:
        """Check if institution exists."""
        try:
            stmt = select(func.count(InstitutionModel.code)).where(InstitutionModel.code == code)
            result = await self.session.execute(stmt)
            return (result.scalar() or 0) > 0
            
        except SQLAlchemyError as e:
            self.logger.error(f"Database error checking institution existence {code}: {e}")
            raise RepositoryError(f"Failed to check institution existence: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error checking institution existence {code}: {e}")
            raise RepositoryError(f"Unexpected error checking institution existence: {e}")
    
    # Helper methods for model conversion
    
    def _institution_to_model(self, institution: Institution) -> InstitutionModel:
        """Convert institution entity to database model."""
        return InstitutionModel(
            code=institution.code,
            name=institution.name,
            country=institution.country,
            institution_type=institution.institution_type.value,
            established_date=institution.established_date,
            website_url=institution.website_url,
            description=institution.description
        )
    
    def _update_institution_model(self, model: InstitutionModel, institution: Institution) -> None:
        """Update existing institution model with entity data."""
        model.name = institution.name
        model.country = institution.country
        model.institution_type = institution.institution_type.value
        model.established_date = institution.established_date
        model.website_url = institution.website_url
        model.description = institution.description


class SqlAlchemyAnalysisRepository(AnalysisRepository):
    """
    SQLAlchemy implementation of AnalysisRepository.
    
    Handles storage and retrieval of NLP analysis results with
    comprehensive querying capabilities.
    """
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def save(self, analysis: SentimentAnalysis) -> None:
        """Save sentiment analysis results."""
        try:
            # Check if analysis already exists for this speech
            existing = await self.session.get(SentimentAnalysisModel, analysis.speech_id)
            
            if existing:
                # Update existing analysis
                self._update_analysis_model(existing, analysis)
                self.logger.debug(f"Updated existing analysis for speech {analysis.speech_id}")
            else:
                # Create new analysis
                model = self._analysis_to_model(analysis)
                self.session.add(model)
                self.logger.debug(f"Added new analysis for speech {analysis.speech_id}")
            
            await self.session.flush()
            
        except IntegrityError as e:
            self.logger.error(f"Integrity error saving analysis for speech {analysis.speech_id}: {e}")
            raise DuplicateEntityError(f"Analysis for speech {analysis.speech_id} violates constraints")
        except SQLAlchemyError as e:
            self.logger.error(f"Database error saving analysis for speech {analysis.speech_id}: {e}")
            raise RepositoryError(f"Failed to save analysis: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error saving analysis for speech {analysis.speech_id}: {e}")
            raise RepositoryError(f"Unexpected error saving analysis: {e}")
    
    async def get_by_speech_id(self, speech_id: UUID) -> Optional[SentimentAnalysis]:
        """Get analysis by speech ID."""
        try:
            result = await self.session.get(SentimentAnalysisModel, speech_id)
            
            if result:
                return self._analysis_model_to_entity(result)
            return None
            
        except SQLAlchemyError as e:
            self.logger.error(f"Database error getting analysis for speech {speech_id}: {e}")
            raise RepositoryError(f"Failed to get analysis: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error getting analysis for speech {speech_id}: {e}")
            raise RepositoryError(f"Unexpected error getting analysis: {e}")
    
    async def find_by_score_range(self, min_score: float, max_score: float, 
                                 limit: Optional[int] = None) -> List[SentimentAnalysis]:
        """Find analyses by score range."""
        try:
            stmt = (
                select(SentimentAnalysisModel)
                .where(
                    and_(
                        SentimentAnalysisModel.hawkish_dovish_score >= min_score,
                        SentimentAnalysisModel.hawkish_dovish_score <= max_score
                    )
                )
                .order_by(desc(SentimentAnalysisModel.hawkish_dovish_score))
            )
            
            if limit:
                stmt = stmt.limit(limit)
            
            result = await self.session.execute(stmt)
            models = result.scalars().all()
            
            analyses = []
            for model in models:
                analysis = self._analysis_model_to_entity(model)
                analyses.append(analysis)
            
            return analyses
            
        except SQLAlchemyError as e:
            self.logger.error(f"Database error finding analyses by score range: {e}")
            raise RepositoryError(f"Failed to find analyses by score range: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error finding analyses by score range: {e}")
            raise RepositoryError(f"Unexpected error finding analyses by score range: {e}")
    
    async def find_by_policy_stance(self, stance: PolicyStance, 
                                   limit: Optional[int] = None) -> List[SentimentAnalysis]:
        """Find analyses by policy stance."""
        try:
            stmt = (
                select(SentimentAnalysisModel)
                .where(SentimentAnalysisModel.policy_stance == stance.value)
                .order_by(desc(SentimentAnalysisModel.analysis_timestamp))
            )
            
            if limit:
                stmt = stmt.limit(limit)
            
            result = await self.session.execute(stmt)
            models = result.scalars().all()
            
            analyses = []
            for model in models:
                analysis = self._analysis_model_to_entity(model)
                analyses.append(analysis)
            
            return analyses
            
        except SQLAlchemyError as e:
            self.logger.error(f"Database error finding analyses by policy stance {stance}: {e}")
            raise RepositoryError(f"Failed to find analyses by policy stance: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error finding analyses by policy stance {stance}: {e}")
            raise RepositoryError(f"Unexpected error finding analyses by policy stance: {e}")
    
    async def delete(self, speech_id: UUID) -> None:
        """Delete analysis by speech ID."""
        try:
            stmt = delete(SentimentAnalysisModel).where(SentimentAnalysisModel.speech_id == speech_id)
            result = await self.session.execute(stmt)
            
            if result.rowcount == 0:
                raise EntityNotFoundError(f"Analysis for speech {speech_id} not found")
            
            self.logger.debug(f"Deleted analysis for speech {speech_id}")
            
        except EntityNotFoundError:
            raise
        except SQLAlchemyError as e:
            self.logger.error(f"Database error deleting analysis for speech {speech_id}: {e}")
            raise RepositoryError(f"Failed to delete analysis: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error deleting analysis for speech {speech_id}: {e}")
            raise RepositoryError(f"Unexpected error deleting analysis: {e}")
    
    async def get_analysis_summary(self, date_range: Optional[DateRange] = None) -> Dict[str, Any]:
        """Get summary statistics for analyses."""
        try:
            # Base query conditions
            conditions = []
            if date_range:
                conditions.extend([
                    SentimentAnalysisModel.analysis_timestamp >= datetime.combine(date_range.start_date, datetime.min.time()),
                    SentimentAnalysisModel.analysis_timestamp <= datetime.combine(date_range.end_date, datetime.max.time())
                ])
            
            base_where = and_(*conditions) if conditions else text("1=1")
            
            # Get basic statistics
            stats_stmt = select(
                func.count(SentimentAnalysisModel.speech_id),
                func.avg(SentimentAnalysisModel.hawkish_dovish_score),
                func.stddev(SentimentAnalysisModel.hawkish_dovish_score),
                func.min(SentimentAnalysisModel.hawkish_dovish_score),
                func.max(SentimentAnalysisModel.hawkish_dovish_score)
            ).where(base_where)
            
            stats_result = await self.session.execute(stats_stmt)
            count, avg_score, std_dev, min_score, max_score = stats_result.one()
            
            # Get stance distribution
            stance_stmt = (
                select(SentimentAnalysisModel.policy_stance, func.count(SentimentAnalysisModel.speech_id))
                .where(base_where)
                .group_by(SentimentAnalysisModel.policy_stance)
            )
            stance_result = await self.session.execute(stance_stmt)
            stance_distribution = dict(stance_result.all())
            
            return {
                'total_analyses': count or 0,
                'average_score': round(avg_score, 3) if avg_score else None,
                'score_std_dev': round(std_dev, 3) if std_dev else None,
                'min_score': round(min_score, 3) if min_score else None,
                'max_score': round(max_score, 3) if max_score else None,
                'stance_distribution': stance_distribution,
                'date_range': {
                    'start_date': date_range.start_date.isoformat() if date_range else None,
                    'end_date': date_range.end_date.isoformat() if date_range else None
                }
            }
            
        except SQLAlchemyError as e:
            self.logger.error(f"Database error getting analysis summary: {e}")
            raise RepositoryError(f"Failed to get analysis summary: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error getting analysis summary: {e}")
            raise RepositoryError(f"Unexpected error getting analysis summary: {e}")
    
    # Helper methods for model conversion
    
    def _analysis_to_model(self, analysis: SentimentAnalysis) -> SentimentAnalysisModel:
        """Convert analysis entity to database model."""
        return SentimentAnalysisModel(
            speech_id=analysis.speech_id,
            hawkish_dovish_score=analysis.hawkish_dovish_score,
            policy_stance=analysis.policy_stance.value,
            uncertainty_score=analysis.uncertainty_score,
            confidence_score=analysis.confidence_score,
            analysis_timestamp=analysis.analysis_timestamp,
            analyzer_version=analysis.analyzer_version,
            topic_classifications=analysis.topic_classifications
        )
    
    def _update_analysis_model(self, model: SentimentAnalysisModel, analysis: SentimentAnalysis) -> None:
        """Update existing analysis model with entity data."""
        model.hawkish_dovish_score = analysis.hawkish_dovish_score
        model.policy_stance = analysis.policy_stance.value
        model.uncertainty_score = analysis.uncertainty_score
        model.confidence_score = analysis.confidence_score
        model.analysis_timestamp = analysis.analysis_timestamp
        model.analyzer_version = analysis.analyzer_version
        model.topic_classifications = analysis.topic_classifications
    
    def _analysis_model_to_entity(self, model: SentimentAnalysisModel) -> SentimentAnalysis:
        """Convert analysis model to entity."""
        return SentimentAnalysis(
            speech_id=model.speech_id,
            hawkish_dovish_score=model.hawkish_dovish_score,
            policy_stance=PolicyStance(model.policy_stance),
            uncertainty_score=model.uncertainty_score,
            confidence_score=model.confidence_score,
            analysis_timestamp=model.analysis_timestamp,
            analyzer_version=model.analyzer_version,
            topic_classifications=model.topic_classifications or []
        )


class SqlAlchemyUnitOfWork(UnitOfWork):
    """
    SQLAlchemy implementation of Unit of Work pattern.
    
    Coordinates transactional operations across multiple repositories
    ensuring ACID properties and consistent state management.
    """
    
    def __init__(self, session: AsyncSession):
        """
        Initialize Unit of Work with async session.
        
        Args:
            session: SQLAlchemy async session
        """
        self.session = session
        self._speeches: Optional[SqlAlchemySpeechRepository] = None
        self._speakers: Optional[SqlAlchemySpeakerRepository] = None
        self._institutions: Optional[SqlAlchemyInstitutionRepository] = None
        self._analyses: Optional[SqlAlchemyAnalysisRepository] = None
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @property
    def speeches(self) -> SqlAlchemySpeechRepository:
        """Get speech repository."""
        if self._speeches is None:
            self._speeches = SqlAlchemySpeechRepository(self.session)
        return self._speeches
    
    @property
    def speakers(self) -> SqlAlchemySpeakerRepository:
        """Get speaker repository."""
        if self._speakers is None:
            self._speakers = SqlAlchemySpeakerRepository(self.session)
        return self._speakers
    
    @property
    def institutions(self) -> SqlAlchemyInstitutionRepository:
        """Get institution repository."""
        if self._institutions is None:
            self._institutions = SqlAlchemyInstitutionRepository(self.session)
        return self._institutions
    
    @property
    def analyses(self) -> SqlAlchemyAnalysisRepository:
        """Get analysis repository."""
        if self._analyses is None:
            self._analyses = SqlAlchemyAnalysisRepository(self.session)
        return self._analyses
    
    async def __aenter__(self):
        """Enter async context manager."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context manager with proper cleanup."""
        try:
            if exc_type is not None:
                await self.rollback()
                self.logger.error(f"Transaction rolled back due to exception: {exc_type.__name__}")
            else:
                await self.commit()
                self.logger.debug("Transaction committed successfully")
        except Exception as e:
            self.logger.error(f"Error during transaction cleanup: {e}")
            await self.rollback()
            raise
        finally:
            await self.session.close()
    
    async def commit(self) -> None:
        """Commit the current transaction."""
        try:
            await self.session.commit()
            self.logger.debug("Transaction committed")
        except SQLAlchemyError as e:
            self.logger.error(f"Error committing transaction: {e}")
            await self.rollback()
            raise RepositoryError(f"Failed to commit transaction: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error committing transaction: {e}")
            await self.rollback()
            raise RepositoryError(f"Unexpected error committing transaction: {e}")
    
    async def rollback(self) -> None:
        """Rollback the current transaction."""
        try:
            await self.session.rollback()
            self.logger.debug("Transaction rolled back")
        except SQLAlchemyError as e:
            self.logger.error(f"Error rolling back transaction: {e}")
            raise RepositoryError(f"Failed to rollback transaction: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error rolling back transaction: {e}")
            raise RepositoryError(f"Unexpected error rolling back transaction: {e}")
    
    async def flush(self) -> None:
        """Flush pending changes to the database without committing."""
        try:
            await self.session.flush()
            self.logger.debug("Session flushed")
        except SQLAlchemyError as e:
            self.logger.error(f"Error flushing session: {e}")
            raise RepositoryError(f"Failed to flush session: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error flushing session: {e}")
            raise RepositoryError(f"Unexpected error flushing session: {e}")
    
    async def refresh(self, entity) -> None:
        """Refresh entity from database."""
        try:
            await self.session.refresh(entity)
            self.logger.debug(f"Entity refreshed: {type(entity).__name__}")
        except SQLAlchemyError as e:
            self.logger.error(f"Error refreshing entity: {e}")
            raise RepositoryError(f"Failed to refresh entity: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error refreshing entity: {e}")
            raise RepositoryError(f"Unexpected error refreshing entity: {e}")


# Factory functions for creating repositories and UoW

async def create_async_engine_from_url(database_url: str, **kwargs) -> AsyncEngine:
    """
    Create async SQLAlchemy engine from database URL.
    
    Args:
        database_url: Database connection URL
        **kwargs: Additional engine arguments
        
    Returns:
        Configured async engine
    """
    default_kwargs = {
        'echo': False,
        'future': True,
        'pool_pre_ping': True,
        'pool_recycle': 3600,
        'pool_size': 20,
        'max_overflow': 30
    }
    default_kwargs.update(kwargs)
    
    return create_async_engine(database_url, **default_kwargs)


def create_unit_of_work(session: AsyncSession) -> SqlAlchemyUnitOfWork:
    """
    Create Unit of Work instance.
    
    Args:
        session: SQLAlchemy async session
        
    Returns:
        Configured Unit of Work instance
    """
    return SqlAlchemyUnitOfWork(session)


# Health check and diagnostic functions

async def check_database_health(engine: AsyncEngine) -> Dict[str, Any]:
    """
    Check database connectivity and basic health metrics.
    
    Args:
        engine: SQLAlchemy async engine
        
    Returns:
        Dictionary with health check results
    """
    try:
        async with engine.begin() as conn:
            # Test basic connectivity
            result = await conn.execute(text("SELECT 1"))
            connectivity = result.scalar() == 1
            
            # Check if tables exist
            tables_result = await conn.execute(
                text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
            )
            tables = [row[0] for row in tables_result.fetchall()]
            
            # Basic statistics
            if 'speeches' in tables:
                count_result = await conn.execute(text("SELECT COUNT(*) FROM speeches"))
                speech_count = count_result.scalar()
            else:
                speech_count = 0
            
            return {
                'status': 'healthy' if connectivity else 'unhealthy',
                'connectivity': connectivity,
                'tables_exist': len(tables) > 0,
                'table_count': len(tables),
                'speech_count': speech_count,
                'timestamp': datetime.utcnow().isoformat()
            }
            
    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }


async def initialize_database_schema(engine: AsyncEngine) -> None:
    """
    Initialize database schema by creating all tables.
    
    Args:
        engine: SQLAlchemy async engine
    """
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            logger.info("Database schema initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database schema: {e}")
        raise RepositoryError(f"Failed to initialize database schema: {e}")


# Export all public classes and functions
__all__ = [
    'SqlAlchemySpeechRepository',
    'SqlAlchemySpeakerRepository', 
    'SqlAlchemyInstitutionRepository',
    'SqlAlchemyAnalysisRepository',
    'SqlAlchemyUnitOfWork',
    'create_async_engine_from_url',
    'create_unit_of_work',
    'check_database_health',
    'initialize_database_schema'
]
                    