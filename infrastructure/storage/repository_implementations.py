"""
Repository Implementations for Central Bank Speech Analysis Platform

This module provides concrete implementations of the repository interfaces using
SQLAlchemy and PostgreSQL. These implementations handle the technical details of
data persistence while maintaining the contracts defined in the domain layer.

Key Features:
- SQLAlchemy ORM for database operations
- PostgreSQL for robust data storage
- Async operations for scalability
- Connection pooling and transaction management
- Full-text search capabilities
- Efficient indexing for common queries
"""

import asyncio
import logging
from datetime import date, datetime
from typing import List, Optional, Dict, Any, Set
from uuid import UUID
from sqlalchemy import (
    Column, String, Integer, Float, DateTime, Date, Boolean, Text, JSON,
    ForeignKey, Index, UniqueConstraint, create_engine, select, delete,
    update, and_, or_, func, desc, asc
)
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, AsyncEngine
from sqlalchemy.orm import declarative_base, relationship, selectinload
from sqlalchemy.sql import text

from domain.entities import (
    CentralBankSpeech, CentralBankSpeaker, Institution, SpeechCollection,
    SentimentAnalysis, SpeechStatus, PolicyStance, InstitutionType
)
from domain.value_objects import DateRange, Url, ContentHash
from domain.repositories import (
    SpeechRepository, SpeakerRepository, InstitutionRepository,
    SpeechCollectionRepository, AnalysisRepository, UnitOfWork,
    QuerySpecification, RepositoryError, EntityNotFoundError,
    DuplicateEntityError, ConcurrencyError
)
from interfaces.plugin_interfaces import SpeechMetadata, SpeechContent

logger = logging.getLogger(__name__)

# SQLAlchemy Base
Base = declarative_base()


# Database Models

class InstitutionModel(Base):
    """SQLAlchemy model for Institution entity."""
    __tablename__ = 'institutions'
    
    code = Column(String(10), primary_key=True)
    name = Column(String(255), nullable=False)
    country = Column(String(100), nullable=False)
    institution_type = Column(String(50), nullable=False)
    established_date = Column(Date, nullable=True)
    website_url = Column(String(500), nullable=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    speakers = relationship("SpeakerModel", back_populates="institution")
    speeches = relationship("SpeechModel", back_populates="institution")
    
    # Indexes
    __table_args__ = (
        Index('idx_institution_country', 'country'),
        Index('idx_institution_type', 'institution_type'),
    )


class SpeakerModel(Base):
    """SQLAlchemy model for CentralBankSpeaker entity."""
    __tablename__ = 'speakers'
    
    id = Column(PostgresUUID(as_uuid=True), primary_key=True)
    name = Column(String(255), nullable=False)
    role = Column(String(100), nullable=False)
    institution_code = Column(String(10), ForeignKey('institutions.code'), nullable=False)
    start_date = Column(Date, nullable=True)
    end_date = Column(Date, nullable=True)
    voting_member = Column(Boolean, default=False)
    biographical_notes = Column(Text, nullable=True)
    alternate_names = Column(JSON, nullable=True)  # Store as JSON array
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    institution = relationship("InstitutionModel", back_populates="speakers")
    speeches = relationship("SpeechModel", back_populates="speaker")
    
    # Indexes
    __table_args__ = (
        Index('idx_speaker_name', 'name'),
        Index('idx_speaker_institution', 'institution_code'),
        Index('idx_speaker_role', 'role'),
        Index('idx_speaker_voting', 'voting_member'),
        Index('idx_speaker_dates', 'start_date', 'end_date'),
    )


class SpeechModel(Base):
    """SQLAlchemy model for CentralBankSpeech entity."""
    __tablename__ = 'speeches'
    
    id = Column(PostgresUUID(as_uuid=True), primary_key=True)
    title = Column(String(500), nullable=True)
    url = Column(String(1000), nullable=True)
    speech_date = Column(Date, nullable=True)
    institution_code = Column(String(10), ForeignKey('institutions.code'), nullable=False)
    speaker_id = Column(PostgresUUID(as_uuid=True), ForeignKey('speakers.id'), nullable=True)
    
    # Content fields
    raw_text = Column(Text, nullable=True)
    cleaned_text = Column(Text, nullable=True)
    word_count = Column(Integer, nullable=True)
    content_hash_sha256 = Column(String(64), nullable=True)
    content_hash_md5 = Column(String(32), nullable=True)
    extraction_method = Column(String(50), nullable=True)
    extraction_confidence = Column(Float, nullable=True)
    
    # Metadata fields
    speech_type = Column(String(50), nullable=True)
    location = Column(String(255), nullable=True)
    language = Column(String(10), default='en')
    tags = Column(JSON, nullable=True)  # Store as JSON array
    
    # Processing fields
    status = Column(String(20), nullable=False, default='discovered')
    processing_history = Column(JSON, nullable=True)
    
    # Sentiment analysis fields
    hawkish_dovish_score = Column(Float, nullable=True)
    policy_stance = Column(String(20), nullable=True)
    uncertainty_score = Column(Float, nullable=True)
    confidence_score = Column(Float, nullable=True)
    analysis_timestamp = Column(DateTime, nullable=True)
    analyzer_version = Column(String(20), nullable=True)
    
    # Validation fields
    validation_status = Column(String(20), nullable=True)
    validation_confidence = Column(Float, nullable=True)
    validation_issues = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    institution = relationship("InstitutionModel", back_populates="speeches")
    speaker = relationship("SpeakerModel", back_populates="speeches")
    
    # Indexes
    __table_args__ = (
        Index('idx_speech_url', 'url'),
        Index('idx_speech_date', 'speech_date'),
        Index('idx_speech_institution', 'institution_code'),
        Index('idx_speech_speaker', 'speaker_id'),
        Index('idx_speech_status', 'status'),
        Index('idx_speech_content_hash', 'content_hash_sha256'),
        Index('idx_speech_sentiment', 'hawkish_dovish_score'),
        Index('idx_speech_stance', 'policy_stance'),
        Index('idx_speech_word_count', 'word_count'),
        # Composite indexes for common queries
        Index('idx_speech_institution_date', 'institution_code', 'speech_date'),
        Index('idx_speech_speaker_date', 'speaker_id', 'speech_date'),
        Index('idx_speech_status_date', 'status', 'speech_date'),
        # Full-text search index
        Index('idx_speech_fulltext', text('to_tsvector(\'english\', COALESCE(title, \'\') || \' \' || COALESCE(cleaned_text, \'\'))')),
        # Unique constraint for URL deduplication
        UniqueConstraint('url', name='uq_speech_url'),
    )


class SpeechCollectionModel(Base):
    """SQLAlchemy model for SpeechCollection entity."""
    __tablename__ = 'speech_collections'
    
    id = Column(PostgresUUID(as_uuid=True), primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    tags = Column(JSON, nullable=True)
    speech_ids = Column(JSON, nullable=True)  # Store speech IDs as JSON array
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Indexes
    __table_args__ = (
        Index('idx_collection_name', 'name'),
        UniqueConstraint('name', name='uq_collection_name'),
    )


class SentimentAnalysisModel(Base):
    """SQLAlchemy model for SentimentAnalysis results."""
    __tablename__ = 'sentiment_analyses'
    
    id = Column(PostgresUUID(as_uuid=True), primary_key=True)
    speech_id = Column(PostgresUUID(as_uuid=True), ForeignKey('speeches.id'), nullable=False)
    hawkish_dovish_score = Column(Float, nullable=False)
    policy_stance = Column(String(20), nullable=False)
    uncertainty_score = Column(Float, nullable=False)
    confidence_score = Column(Float, nullable=False)
    analyzer_version = Column(String(50), nullable=False)
    raw_scores = Column(JSON, nullable=True)
    topic_classifications = Column(JSON, nullable=True)
    analysis_timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Indexes
    __table_args__ = (
        Index('idx_analysis_speech', 'speech_id'),
        Index('idx_analysis_version', 'analyzer_version'),
        Index('idx_analysis_timestamp', 'analysis_timestamp'),
        Index('idx_analysis_stance', 'policy_stance'),
        # Composite index for latest analysis queries
        Index('idx_analysis_speech_timestamp', 'speech_id', 'analysis_timestamp'),
    )


# Repository Implementations

class SqlAlchemySpeechRepository(SpeechRepository):
    """SQLAlchemy implementation of SpeechRepository."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def save(self, speech: CentralBankSpeech) -> None:
        """Save a speech to the database."""
        try:
            # Convert domain entity to database model
            model = await self._speech_to_model(speech)
            
            # Check if speech already exists
            existing = await self.session.get(SpeechModel, speech.id)
            if existing:
                # Update existing speech
                await self._update_speech_model(existing, speech)
            else:
                # Add new speech
                self.session.add(model)
            
            await self.session.flush()
            
        except Exception as e:
            self.logger.error(f"Error saving speech {speech.id}: {e}")
            raise RepositoryError(f"Failed to save speech: {e}")
    
    async def get_by_id(self, speech_id: UUID) -> Optional[CentralBankSpeech]:
        """Retrieve a speech by its ID."""
        try:
            stmt = (
                select(SpeechModel)
                .options(selectinload(SpeechModel.institution))
                .options(selectinload(SpeechModel.speaker))
                .where(SpeechModel.id == speech_id)
            )
            
            result = await self.session.execute(stmt)
            model = result.scalar_one_or_none()
            
            return await self._model_to_speech(model) if model else None
            
        except Exception as e:
            self.logger.error(f"Error getting speech by ID {speech_id}: {e}")
            raise RepositoryError(f"Failed to get speech: {e}")
    
    async def get_by_url(self, url: Url) -> Optional[CentralBankSpeech]:
        """Retrieve a speech by its URL."""
        try:
            stmt = (
                select(SpeechModel)
                .options(selectinload(SpeechModel.institution))
                .options(selectinload(SpeechModel.speaker))
                .where(SpeechModel.url == url.value)
            )
            
            result = await self.session.execute(stmt)
            model = result.scalar_one_or_none()
            
            return await self._model_to_speech(model) if model else None
            
        except Exception as e:
            self.logger.error(f"Error getting speech by URL {url}: {e}")
            raise RepositoryError(f"Failed to get speech: {e}")
    
    async def get_by_content_hash(self, content_hash: ContentHash) -> Optional[CentralBankSpeech]:
        """Retrieve a speech by its content hash."""
        try:
            stmt = (
                select(SpeechModel)
                .options(selectinload(SpeechModel.institution))
                .options(selectinload(SpeechModel.speaker))
                .where(SpeechModel.content_hash_sha256 == content_hash.sha256)
            )
            
            result = await self.session.execute(stmt)
            model = result.scalar_one_or_none()
            
            return await self._model_to_speech(model) if model else None
            
        except Exception as e:
            self.logger.error(f"Error getting speech by content hash: {e}")
            raise RepositoryError(f"Failed to get speech: {e}")
    
    async def find_by_speaker(self, speaker: CentralBankSpeaker, limit: Optional[int] = None) -> List[CentralBankSpeech]:
        """Find speeches by speaker."""
        try:
            stmt = (
                select(SpeechModel)
                .options(selectinload(SpeechModel.institution))
                .options(selectinload(SpeechModel.speaker))
                .where(SpeechModel.speaker_id == speaker.id)
                .order_by(desc(SpeechModel.speech_date))
            )
            
            if limit:
                stmt = stmt.limit(limit)
            
            result = await self.session.execute(stmt)
            models = result.scalars().all()
            
            return [await self._model_to_speech(model) for model in models]
            
        except Exception as e:
            self.logger.error(f"Error finding speeches by speaker {speaker.id}: {e}")
            raise RepositoryError(f"Failed to find speeches: {e}")
    
    async def find_by_institution(self, institution: Institution, limit: Optional[int] = None) -> List[CentralBankSpeech]:
        """Find speeches by institution."""
        try:
            stmt = (
                select(SpeechModel)
                .options(selectinload(SpeechModel.institution))
                .options(selectinload(SpeechModel.speaker))
                .where(SpeechModel.institution_code == institution.code)
                .order_by(desc(SpeechModel.speech_date))
            )
            
            if limit:
                stmt = stmt.limit(limit)
            
            result = await self.session.execute(stmt)
            models = result.scalars().all()
            
            return [await self._model_to_speech(model) for model in models]
            
        except Exception as e:
            self.logger.error(f"Error finding speeches by institution {institution.code}: {e}")
            raise RepositoryError(f"Failed to find speeches: {e}")
    
    async def find_by_date_range(self, date_range: DateRange, institution: Optional[Institution] = None) -> List[CentralBankSpeech]:
        """Find speeches within a date range."""
        try:
            stmt = (
                select(SpeechModel)
                .options(selectinload(SpeechModel.institution))
                .options(selectinload(SpeechModel.speaker))
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
            
            return [await self._model_to_speech(model) for model in models]
            
        except Exception as e:
            self.logger.error(f"Error finding speeches by date range: {e}")
            raise RepositoryError(f"Failed to find speeches: {e}")
    
    async def find_by_status(self, status: SpeechStatus, limit: Optional[int] = None) -> List[CentralBankSpeech]:
        """Find speeches by status."""
        try:
            stmt = (
                select(SpeechModel)
                .options(selectinload(SpeechModel.institution))
                .options(selectinload(SpeechModel.speaker))
                .where(SpeechModel.status == status.value)
                .order_by(desc(SpeechModel.created_at))
            )
            
            if limit:
                stmt = stmt.limit(limit)
            
            result = await self.session.execute(stmt)
            models = result.scalars().all()
            
            return [await self._model_to_speech(model) for model in models]
            
        except Exception as e:
            self.logger.error(f"Error finding speeches by status {status}: {e}")
            raise RepositoryError(f"Failed to find speeches: {e}")
    
    async def find_by_policy_stance(self, stance: PolicyStance, date_range: Optional[DateRange] = None) -> List[CentralBankSpeech]:
        """Find speeches by policy stance."""
        try:
            stmt = (
                select(SpeechModel)
                .options(selectinload(SpeechModel.institution))
                .options(selectinload(SpeechModel.speaker))
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
            
            return [await self._model_to_speech(model) for model in models]
            
        except Exception as e:
            self.logger.error(f"Error finding speeches by policy stance {stance}: {e}")
            raise RepositoryError(f"Failed to find speeches: {e}")
    
    async def find_by_sentiment_range(self, min_score: float, max_score: float, 
                                     institution: Optional[Institution] = None) -> List[CentralBankSpeech]:
        """Find speeches by sentiment score range."""
        try:
            stmt = (
                select(SpeechModel)
                .options(selectinload(SpeechModel.institution))
                .options(selectinload(SpeechModel.speaker))
                .where(
                    and_(
                        SpeechModel.hawkish_dovish_score >= min_score,
                        SpeechModel.hawkish_dovish_score <= max_score
                    )
                )
                .order_by(desc(SpeechModel.speech_date))
            )
            
            if institution:
                stmt = stmt.where(SpeechModel.institution_code == institution.code)
            
            result = await self.session.execute(stmt)
            models = result.scalars().all()
            
            return [await self._model_to_speech(model) for model in models]
            
        except Exception as e:
            self.logger.error(f"Error finding speeches by sentiment range: {e}")
            raise RepositoryError(f"Failed to find speeches: {e}")
    
    async def find_by_specification(self, specification: QuerySpecification, 
                                   limit: Optional[int] = None) -> List[CentralBankSpeech]:
        """Find speeches by specification (in-memory filtering for now)."""
        try:
            # For complex specifications, we'll load speeches and filter in memory
            # This is not optimal for large datasets, but provides flexibility
            stmt = (
                select(SpeechModel)
                .options(selectinload(SpeechModel.institution))
                .options(selectinload(SpeechModel.speaker))
                .order_by(desc(SpeechModel.speech_date))
            )
            
            if limit:
                stmt = stmt.limit(limit * 2)  # Load extra to account for filtering
            
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
            
        except Exception as e:
            self.logger.error(f"Error finding speeches by specification: {e}")
            raise RepositoryError(f"Failed to find speeches: {e}")
    
    async def count_by_institution(self, institution: Institution) -> int:
        """Count speeches by institution."""
        try:
            stmt = (
                select(func.count(SpeechModel.id))
                .where(SpeechModel.institution_code == institution.code)
            )
            
            result = await self.session.execute(stmt)
            return result.scalar()
            
        except Exception as e:
            self.logger.error(f"Error counting speeches by institution {institution.code}: {e}")
            raise RepositoryError(f"Failed to count speeches: {e}")
    
    async def count_by_status(self, status: SpeechStatus) -> int:
        """Count speeches by status."""
        try:
            stmt = (
                select(func.count(SpeechModel.id))
                .where(SpeechModel.status == status.value)
            )
            
            result = await self.session.execute(stmt)
            return result.scalar()
            
        except Exception as e:
            self.logger.error(f"Error counting speeches by status {status}: {e}")
            raise RepositoryError(f"Failed to count speeches: {e}")
    
    async def get_all_institutions(self) -> List[Institution]:
        """Get all institutions that have speeches."""
        try:
            stmt = (
                select(InstitutionModel)
                .join(SpeechModel)
                .distinct()
            )
            
            result = await self.session.execute(stmt)
            models = result.scalars().all()
            
            return [self._institution_model_to_entity(model) for model in models]
            
        except Exception as e:
            self.logger.error(f"Error getting all institutions: {e}")
            raise RepositoryError(f"Failed to get institutions: {e}")
    
    async def get_date_range_for_institution(self, institution: Institution) -> Optional[DateRange]:
        """Get date range for institution's speeches."""
        try:
            stmt = (
                select(
                    func.min(SpeechModel.speech_date),
                    func.max(SpeechModel.speech_date)
                )
                .where(SpeechModel.institution_code == institution.code)
            )
            
            result = await self.session.execute(stmt)
            min_date, max_date = result.one()
            
            if min_date and max_date:
                return DateRange(start_date=min_date, end_date=max_date)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting date range for institution {institution.code}: {e}")
            raise RepositoryError(f"Failed to get date range: {e}")
    
    async def delete(self, speech: CentralBankSpeech) -> None:
        """Delete a speech."""
        try:
            stmt = delete(SpeechModel).where(SpeechModel.id == speech.id)
            result = await self.session.execute(stmt)
            
            if result.rowcount == 0:
                raise EntityNotFoundError(f"Speech {speech.id} not found")
            
        except EntityNotFoundError:
            raise
        except Exception as e:
            self.logger.error(f"Error deleting speech {speech.id}: {e}")
            raise RepositoryError(f"Failed to delete speech: {e}")
    
    async def exists(self, speech_id: UUID) -> bool:
        """Check if speech exists."""
        try:
            stmt = select(func.count(SpeechModel.id)).where(SpeechModel.id == speech_id)
            result = await self.session.execute(stmt)
            return result.scalar() > 0
            
        except Exception as e:
            self.logger.error(f"Error checking speech existence {speech_id}: {e}")
            raise RepositoryError(f"Failed to check speech existence: {e}")
    
    async def get_processing_metrics(self, date_range: Optional[DateRange] = None) -> Dict[str, Any]:
        """Get processing metrics."""
        try:
            # Base query for speech counts
            base_stmt = select(func.count(SpeechModel.id))
            
            if date_range:
                base_stmt = base_stmt.where(
                    and_(
                        SpeechModel.speech_date >= date_range.start_date,
                        SpeechModel.speech_date <= date_range.end_date
                    )
                )
            
            # Get total count
            total_result = await self.session.execute(base_stmt)
            total_count = total_result.scalar()
            
            # Get status distribution
            status_stmt = (
                select(SpeechModel.status, func.count(SpeechModel.id))
                .group_by(SpeechModel.status)
            )
            
            if date_range:
                status_stmt = status_stmt.where(
                    and_(
                        SpeechModel.speech_date >= date_range.start_date,
                        SpeechModel.speech_date <= date_range.end_date
                    )
                )
            
            status_result = await self.session.execute(status_stmt)
            status_distribution = dict(status_result.all())
            
            return {
                'total_speeches': total_count,
                'status_distribution': status_distribution,
                'date_range': {
                    'start_date': date_range.start_date.isoformat() if date_range else None,
                    'end_date': date_range.end_date.isoformat() if date_range else None
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting processing metrics: {e}")
            raise RepositoryError(f"Failed to get metrics: {e}")
    
    # Helper methods for model conversion
    
    async def _speech_to_model(self, speech: CentralBankSpeech) -> SpeechModel:
        """Convert speech entity to database model."""
        model = SpeechModel(
            id=speech.id,
            title=speech.title,
            url=speech.url,
            speech_date=speech.speech_date,
            institution_code=speech.institution.code if speech.institution else None,
            speaker_id=speech.speaker.id if speech.speaker else None,
            status=speech.status.value,
            processing_history=speech.processing_history,
            tags=list(speech.tags) if speech.tags else None,
            created_at=speech.created_at,
            updated_at=speech.updated_at
        )
        
        # Add content fields if available
        if speech.content:
            model.raw_text = speech.content.raw_text
            model.cleaned_text = speech.content.cleaned_text
            model.word_count = speech.content.word_count
            model.extraction_method = speech.content.extraction_method
            model.extraction_confidence = speech.content.confidence_score
            
            # Generate content hash if not present
            if not model.content_hash_sha256:
                content_hash = ContentHash.from_content(speech.content.cleaned_text)
                model.content_hash_sha256 = content_hash.sha256
                model.content_hash_md5 = content_hash.md5
        
        # Add sentiment analysis fields if available
        if speech.sentiment_analysis:
            model.hawkish_dovish_score = speech.sentiment_analysis.hawkish_dovish_score
            model.policy_stance = speech.sentiment_analysis.policy_stance.value
            model.uncertainty_score = speech.sentiment_analysis.uncertainty_score
            model.confidence_score = speech.sentiment_analysis.confidence_score
            model.analysis_timestamp = speech.sentiment_analysis.analysis_timestamp
            model.analyzer_version = speech.sentiment_analysis.analyzer_version
        
        # Add validation fields if available
        if speech.validation_result:
            model.validation_status = speech.validation_result.status.value
            model.validation_confidence = speech.validation_result.confidence
            model.validation_issues = speech.validation_result.issues
        
        return model
    
    async def _update_speech_model(self, model: SpeechModel, speech: CentralBankSpeech) -> None:
        """Update existing speech model with entity data."""
        # Update basic fields
        model.title = speech.title
        model.url = speech.url
        model.speech_date = speech.speech_date
        model.institution_code = speech.institution.code if speech.institution else None
        model.speaker_id = speech.speaker.id if speech.speaker else None
        model.status = speech.status.value
        model.processing_history = speech.processing_history
        model.tags = list(speech.tags) if speech.tags else None
        model.updated_at = speech.updated_at
        
        # Update content fields if available
        if speech.content:
            model.raw_text = speech.content.raw_text
            model.cleaned_text = speech.content.cleaned_text
            model.word_count = speech.content.word_count
            model.extraction_method = speech.content.extraction_method
            model.extraction_confidence = speech.content.confidence_score
            
            # Update content hash
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
            speech.metadata = SpeechMetadata(
                url=model.url or "",
                title=model.title or "",
                speaker_name=model.speaker.name if model.speaker else "",
                date=model.speech_date or date.today(),
                institution_code=model.institution_code or "",
                speech_type=SpeechType.FORMAL_SPEECH,
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
                policy_stance=PolicyStance(model.policy_stance),
                uncertainty_score=model.uncertainty_score or 0.0,
                confidence_score=model.confidence_score or 0.0,
                analysis_timestamp=model.analysis_timestamp or datetime.utcnow(),
                analyzer_version=model.analyzer_version or "unknown"
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
    
async   def _institution_model_to_entity(self, model: InstitutionModel) -> Institution:
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
    
async def _speaker_model_to_entity(self, model: SpeakerModel) -> CentralBankSpeaker:
        """Convert speaker model to entity."""
        speaker = CentralBankSpeaker(
            id=model.id,
            name=model.name,
            role=model.role,
            start_date=model.start_date,
            end_date=model.end_date,
            voting_member=model.voting_member,
            biographical_notes=model.biographical_notes
        )
        
        # Set alternate names if available
        if model.alternate_names:
            speaker.alternate_names = set(model.alternate_names)
        
        # Set institution if available
        if model.institution:
            speaker.institution = self._institution_model_to_entity(model.institution)
        
        return speaker


class SqlAlchemySpeakerRepository(SpeakerRepository):
    """SQLAlchemy implementation of SpeakerRepository."""
    
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
            else:
                # Add new speaker
                model = self._speaker_to_model(speaker)
                self.session.add(model)
            
            await self.session.flush()
            
        except Exception as e:
            self.logger.error(f"Error saving speaker {speaker.id}: {e}")
            raise RepositoryError(f"Failed to save speaker: {e}")
    
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
            
            return self._speaker_model_to_entity(model) if model else None
            
        except Exception as e:
            self.logger.error(f"Error getting speaker by ID {speaker_id}: {e}")
            raise RepositoryError(f"Failed to get speaker: {e}")
    
    async def find_by_name(self, name: str) -> List[CentralBankSpeaker]:
        """Find speakers by name."""
        try:
            stmt = (
                select(SpeakerModel)
                .options(selectinload(SpeakerModel.institution))
                .where(SpeakerModel.name.ilike(f"%{name}%"))
            )
            
            result = await self.session.execute(stmt)
            models = result.scalars().all()
            
            return [self._speaker_model_to_entity(model) for model in models]
            
        except Exception as e:
            self.logger.error(f"Error finding speakers by name {name}: {e}")
            raise RepositoryError(f"Failed to find speakers: {e}")
    
    async def find_by_institution(self, institution: Institution) -> List[CentralBankSpeaker]:
        """Find speakers by institution."""
        try:
            stmt = (
                select(SpeakerModel)
                .options(selectinload(SpeakerModel.institution))
                .where(SpeakerModel.institution_code == institution.code)
            )
            
            result = await self.session.execute(stmt)
            models = result.scalars().all()
            
            return [self._speaker_model_to_entity(model) for model in models]
            
        except Exception as e:
            self.logger.error(f"Error finding speakers by institution {institution.code}: {e}")
            raise RepositoryError(f"Failed to find speakers: {e}")
    
    async def find_current_speakers(self, institution: Optional[Institution] = None) -> List[CentralBankSpeaker]:
        """Find current speakers."""
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
            )
            
            if institution:
                stmt = stmt.where(SpeakerModel.institution_code == institution.code)
            
            result = await self.session.execute(stmt)
            models = result.scalars().all()
            
            return [self._speaker_model_to_entity(model) for model in models]
            
        except Exception as e:
            self.logger.error(f"Error finding current speakers: {e}")
            raise RepositoryError(f"Failed to find speakers: {e}")
    
    async def find_by_role(self, role: str, institution: Optional[Institution] = None) -> List[CentralBankSpeaker]:
        """Find speakers by role."""
        try:
            stmt = (
                select(SpeakerModel)
                .options(selectinload(SpeakerModel.institution))
                .where(SpeakerModel.role.ilike(f"%{role}%"))
            )
            
            if institution:
                stmt = stmt.where(SpeakerModel.institution_code == institution.code)
            
            result = await self.session.execute(stmt)
            models = result.scalars().all()
            
            return [self._speaker_model_to_entity(model) for model in models]
            
        except Exception as e:
            self.logger.error(f"Error finding speakers by role {role}: {e}")
            raise RepositoryError(f"Failed to find speakers: {e}")
    
    async def find_voting_members(self, institution: Institution) -> List[CentralBankSpeaker]:
        """Find voting members."""
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
            )
            
            result = await self.session.execute(stmt)
            models = result.scalars().all()
            
            return [self._speaker_model_to_entity(model) for model in models]
            
        except Exception as e:
            self.logger.error(f"Error finding voting members for {institution.code}: {e}")
            raise RepositoryError(f"Failed to find voting members: {e}")
    
    async def search_by_alternate_names(self, name: str) -> List[CentralBankSpeaker]:
        """Search speakers by alternate names."""
        try:
            # Use PostgreSQL JSON operations to search in alternate_names array
            stmt = (
                select(SpeakerModel)
                .options(selectinload(SpeakerModel.institution))
                .where(
                    SpeakerModel.alternate_names.op('?')(name)
                )
            )
            
            result = await self.session.execute(stmt)
            models = result.scalars().all()
            
            return [self._speaker_model_to_entity(model) for model in models]
            
        except Exception as e:
            self.logger.error(f"Error searching speakers by alternate names {name}: {e}")
            raise RepositoryError(f"Failed to search speakers: {e}")
    
    async def delete(self, speaker: CentralBankSpeaker) -> None:
        """Delete a speaker."""
        try:
            stmt = delete(SpeakerModel).where(SpeakerModel.id == speaker.id)
            result = await self.session.execute(stmt)
            
            if result.rowcount == 0:
                raise EntityNotFoundError(f"Speaker {speaker.id} not found")
            
        except EntityNotFoundError:
            raise
        except Exception as e:
            self.logger.error(f"Error deleting speaker {speaker.id}: {e}")
            raise RepositoryError(f"Failed to delete speaker: {e}")
    
    async def exists(self, speaker_id: UUID) -> bool:
        """Check if speaker exists."""
        try:
            stmt = select(func.count(SpeakerModel.id)).where(SpeakerModel.id == speaker_id)
            result = await self.session.execute(stmt)
            return result.scalar() > 0
            
        except Exception as e:
            self.logger.error(f"Error checking speaker existence {speaker_id}: {e}")
            raise RepositoryError(f"Failed to check speaker existence: {e}")
    
    def _speaker_to_model(self, speaker: CentralBankSpeaker) -> SpeakerModel:
        """Convert speaker entity to database model."""
        return SpeakerModel(
            id=speaker.id,
            name=speaker.name,
            role=speaker.role,
            institution_code=speaker.institution.code if speaker.institution else None,
            start_date=speaker.start_date,
            end_date=speaker.end_date,
            voting_member=speaker.voting_member,
            biographical_notes=speaker.biographical_notes,
            alternate_names=list(speaker.alternate_names) if speaker.alternate_names else None
        )
    
    async def _update_speaker_model(self, model: SpeakerModel, speaker: CentralBankSpeaker) -> None:
        """Update existing speaker model with entity data."""
        model.name = speaker.name
        model.role = speaker.role
        model.institution_code = speaker.institution.code if speaker.institution else None
        model.start_date = speaker.start_date
        model.end_date = speaker.end_date
        model.voting_member = speaker.voting_member
        model.biographical_notes = speaker.biographical_notes
        model.alternate_names = list(speaker.alternate_names) if speaker.alternate_names else None
        model.updated_at = datetime.utcnow()
    
    def _speaker_model_to_entity(self, model: SpeakerModel) -> CentralBankSpeaker:
        """Convert speaker model to entity."""
        speaker = CentralBankSpeaker(
            id=model.id,
            name=model.name,
            role=model.role,
            start_date=model.start_date,
            end_date=model.end_date,
            voting_member=model.voting_member,
            biographical_notes=model.biographical_notes
        )
        
        # Set alternate names if available
        if model.alternate_names:
            speaker.alternate_names = set(model.alternate_names)
        
        # Set institution if available
        if model.institution:
            speaker.institution = Institution(
                code=model.institution.code,
                name=model.institution.name,
                country=model.institution.country,
                institution_type=InstitutionType(model.institution.institution_type),
                established_date=model.institution.established_date,
                website_url=model.institution.website_url,
                description=model.institution.description
            )
        
        return speaker


class SqlAlchemyInstitutionRepository(InstitutionRepository):
    """SQLAlchemy implementation of InstitutionRepository."""
    
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
                await self._update_institution_model(existing, institution)
            else:
                # Add new institution
                model = self._institution_to_model(institution)
                self.session.add(model)
            
            await self.session.flush()
            
        except Exception as e:
            self.logger.error(f"Error saving institution {institution.code}: {e}")
            raise RepositoryError(f"Failed to save institution: {e}")
    
    async def get_by_code(self, code: str) -> Optional[Institution]:
        """Retrieve an institution by code."""
        try:
            model = await self.session.get(InstitutionModel, code)
            return self._institution_model_to_entity(model) if model else None
            
        except Exception as e:
            self.logger.error(f"Error getting institution by code {code}: {e}")
            raise RepositoryError(f"Failed to get institution: {e}")
    
    async def get_by_name(self, name: str) -> Optional[Institution]:
        """Retrieve an institution by name."""
        try:
            stmt = select(InstitutionModel).where(InstitutionModel.name == name)
            result = await self.session.execute(stmt)
            model = result.scalar_one_or_none()
            
            return self._institution_model_to_entity(model) if model else None
            
        except Exception as e:
            self.logger.error(f"Error getting institution by name {name}: {e}")
            raise RepositoryError(f"Failed to get institution: {e}")
    
    async def get_all(self) -> List[Institution]:
        """Get all institutions."""
        try:
            stmt = select(InstitutionModel)
            result = await self.session.execute(stmt)
            models = result.scalars().all()
            
            return [self._institution_model_to_entity(model) for model in models]
            
        except Exception as e:
            self.logger.error(f"Error getting all institutions: {e}")
            raise RepositoryError(f"Failed to get institutions: {e}")
    
    async def find_by_country(self, country: str) -> List[Institution]:
        """Find institutions by country."""
        try:
            stmt = select(InstitutionModel).where(InstitutionModel.country == country)
            result = await self.session.execute(stmt)
            models = result.scalars().all()
            
            return [self._institution_model_to_entity(model) for model in models]
            
        except Exception as e:
            self.logger.error(f"Error finding institutions by country {country}: {e}")
            raise RepositoryError(f"Failed to find institutions: {e}")
    
    async def delete(self, institution: Institution) -> None:
        """Delete an institution."""
        try:
            stmt = delete(InstitutionModel).where(InstitutionModel.code == institution.code)
            result = await self.session.execute(stmt)
            
            if result.rowcount == 0:
                raise EntityNotFoundError(f"Institution {institution.code} not found")
            
        except EntityNotFoundError:
            raise
        except Exception as e:
            self.logger.error(f"Error deleting institution {institution.code}: {e}")
            raise RepositoryError(f"Failed to delete institution: {e}")
    
    async def exists(self, code: str) -> bool:
        """Check if institution exists."""
        try:
            stmt = select(func.count(InstitutionModel.code)).where(InstitutionModel.code == code)
            result = await self.session.execute(stmt)
            return result.scalar() > 0
            
        except Exception as e:
            self.logger.error(f"Error checking institution existence {code}: {e}")
            raise RepositoryError(f"Failed to check institution existence: {e}")
    
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
    
    async def _update_institution_model(self, model: InstitutionModel, institution: Institution) -> None:
        """Update existing institution model with entity data."""
        model.name = institution.name
        model.country = institution.country
        model.institution_type = institution.institution_type.value
        model.established_date = institution.established_date
        model.website_url = institution.website_url
        model.description = institution.description
        model.updated_at = datetime.utcnow()
    
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


class SqlAlchemyUnitOfWork(UnitOfWork):
    """SQLAlchemy implementation of Unit of Work."""
    
    def __init__(self, session_factory):
        self.session_factory = session_factory
        self.session: Optional[AsyncSession] = None
        self._speeches: Optional[SpeechRepository] = None
        self._speakers: Optional[SpeakerRepository] = None
        self._institutions: Optional[InstitutionRepository] = None
        self._collections: Optional[SpeechCollectionRepository] = None
        self._analyses: Optional[AnalysisRepository] = None
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def __aenter__(self) -> 'SqlAlchemyUnitOfWork':
        """Start a new unit of work."""
        self.session = self.session_factory()
        return self
    
async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
    """Complete or rollback the unit of work."""
    try:
        if exc_type is not None:
            await self.rollback()
        else:
            try:
                await self.commit()
            except Exception:
                await self.rollback()
                raise
    finally:
        if self.session:
            await self.session.close()

    
    async def commit(self) -> None:
        """Commit all changes in this unit of work."""
        if self.session:
            try:
                await self.session.commit()
            except Exception as e:
                self.logger.error(f"Error committing transaction: {e}")
                await self.rollback()
                raise RepositoryError(f"Failed to commit transaction: {e}")
    
    async def rollback(self) -> None:
        """Rollback all changes in this unit of work."""
        if self.session:
            try:
                await self.session.rollback()
            except Exception as e:
                self.logger.error(f"Error rolling back transaction: {e}")
                raise RepositoryError(f"Failed to rollback transaction: {e}")
    
    @property
    def speeches(self) -> SpeechRepository:
        """Get the speech repository for this unit of work."""
        if self._speeches is None:
            self._speeches = SqlAlchemySpeechRepository(self.session)
        return self._speeches
    
    @property
    def speakers(self) -> SpeakerRepository:
        """Get the speaker repository for this unit of work."""
        if self._speakers is None:
            self._speakers = SqlAlchemySpeakerRepository(self.session)
        return self._speakers
    
    @property
    def institutions(self) -> InstitutionRepository:
        """Get the institution repository for this unit of work."""
        if self._institutions is None:
            self._institutions = SqlAlchemyInstitutionRepository(self.session)
        return self._institutions
    
    @property
    def collections(self) -> SpeechCollectionRepository:
        """Get the collection repository for this unit of work."""
        if self._collections is None:
            # TODO: Implement SqlAlchemySpeechCollectionRepository
            raise NotImplementedError("SpeechCollectionRepository not yet implemented")
        return self._collections
    
    @property
    def analyses(self) -> AnalysisRepository:
        """Get the analysis repository for this unit of work."""
        if self._analyses is None:
            # TODO: Implement SqlAlchemyAnalysisRepository
            raise NotImplementedError("AnalysisRepository not yet implemented")
        return self._analyses


# Database Configuration and Setup

class DatabaseConfig:
    """Configuration for database connection."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "central_bank_speeches",
        username: str = "postgres",
        password: str = "postgres",
        pool_size: int = 10,
        max_overflow: int = 20,
        pool_timeout: int = 30,
        pool_recycle: int = 3600
    ):
        self.host = host
        self.port = port
        self.database = database
        self.username = username
        self.password = password
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_timeout = pool_timeout
        self.pool_recycle = pool_recycle
    
    @property
    def database_url(self) -> str:
        """Get the database connection URL."""
        return (
            f"postgresql+asyncpg://{self.username}:{self.password}@"
            f"{self.host}:{self.port}/{self.database}"
        )


class DatabaseManager:
    """Manages database connections and schema."""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.engine: Optional[AsyncEngine] = None
        self.session_factory = None
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def initialize(self) -> None:
        """Initialize database engine and session factory."""
        try:
            self.engine = create_async_engine(
                self.config.database_url,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout,
                pool_recycle=self.config.pool_recycle,
                echo=False  # Set to True for SQL query logging
            )
            
            from sqlalchemy.orm import sessionmaker
            self.session_factory = sessionmaker(
                bind=self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            self.logger.info("Database connection initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            raise RepositoryError(f"Database initialization failed: {e}")
    
    async def create_tables(self) -> None:
        """Create database tables."""
        if not self.engine:
            raise RepositoryError("Database engine not initialized")
        
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            
            self.logger.info("Database tables created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create database tables: {e}")
            raise RepositoryError(f"Table creation failed: {e}")
    
    async def drop_tables(self) -> None:
        """Drop database tables."""
        if not self.engine:
            raise RepositoryError("Database engine not initialized")
        
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
            
            self.logger.info("Database tables dropped successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to drop database tables: {e}")
            raise RepositoryError(f"Table drop failed: {e}")
    
    def get_unit_of_work(self) -> UnitOfWork:
        """Get a new unit of work instance."""
        if not self.session_factory:
            raise RepositoryError("Database not initialized")
        
        return SqlAlchemyUnitOfWork(self.session_factory)
    
    async def close(self) -> None:
        """Close database engine."""
        if self.engine:
            await self.engine.dispose()
            self.logger.info("Database connection closed")


# Factory function for easy setup
def create_database_manager(
    host: str = "localhost",
    port: int = 5432,
    database: str = "central_bank_speeches",
    username: str = "postgres",
    password: str = "postgres"
) -> DatabaseManager:
    """
    Create a database manager with the specified configuration.
    
    Args:
        host: Database host
        port: Database port
        database: Database name
        username: Database username
        password: Database password
        
    Returns:
        Configured DatabaseManager instance
        
    Example:
        >>> db_manager = create_database_manager(
        ...     host="localhost",
        ...     database="speech_analysis",
        ...     username="app_user",
        ...     password="secure_password"
        ... )
        >>> await db_manager.initialize()
        >>> await db_manager.create_tables()
        >>> 
        >>> # Use the database
        >>> uow = db_manager.get_unit_of_work()
        >>> async with uow:
        ...     speeches = await uow.speeches.find_by_status(SpeechStatus.DISCOVERED)
        ...     print(f"Found {len(speeches)} speeches")
    """
    config = DatabaseConfig(
        host=host,
        port=port,
        database=database,
        username=username,
        password=password
    )
    
    return DatabaseManager(config)