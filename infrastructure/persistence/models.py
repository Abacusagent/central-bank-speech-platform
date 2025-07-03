# infrastructure/persistence/models.py

"""
SQLAlchemy ORM Models for Central Bank Speech Analysis Platform

Defines database models and mappings for all core domain entities,
enabling robust, type-safe data persistence.

Author: Central Bank Speech Analysis Platform
Date: 2025
"""

from sqlalchemy import (
    Column, String, Integer, Float, DateTime, Date, Boolean, Text, JSON,
    ForeignKey, Index, UniqueConstraint
)
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime
from uuid import uuid4

Base = declarative_base()

class InstitutionModel(Base):
    __tablename__ = "institutions"

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
    speakers = relationship("SpeakerModel", back_populates="institution", cascade="all, delete-orphan")
    speeches = relationship("SpeechModel", back_populates="institution", cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_institution_country', 'country'),
        Index('idx_institution_type', 'institution_type'),
    )

class SpeakerModel(Base):
    __tablename__ = "speakers"

    id = Column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(255), nullable=False)
    role = Column(String(100), nullable=False)
    institution_code = Column(String(10), ForeignKey('institutions.code'), nullable=False)
    start_date = Column(Date, nullable=True)
    end_date = Column(Date, nullable=True)
    voting_member = Column(Boolean, default=False)
    biographical_notes = Column(Text, nullable=True)
    alternate_names = Column(JSON, nullable=True)  # JSON array of names
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    institution = relationship("InstitutionModel", back_populates="speakers")
    speeches = relationship("SpeechModel", back_populates="speaker")

    __table_args__ = (
        Index('idx_speaker_name', 'name'),
        Index('idx_speaker_institution', 'institution_code'),
        Index('idx_speaker_role', 'role'),
        Index('idx_speaker_voting', 'voting_member'),
        Index('idx_speaker_dates', 'start_date', 'end_date'),
    )

class SpeechModel(Base):
    __tablename__ = "speeches"

    id = Column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
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
    tags = Column(JSON, nullable=True)  # JSON array of tags

    # Processing fields
    status = Column(String(20), nullable=False, default='discovered')
    processing_history = Column(JSON, nullable=True)

    # Sentiment/NLP fields
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
        Index('idx_speech_institution_date', 'institution_code', 'speech_date'),
        Index('idx_speech_speaker_date', 'speaker_id', 'speech_date'),
        Index('idx_speech_status_date', 'status', 'speech_date'),
        UniqueConstraint('url', name='uq_speech_url'),
    )

class SentimentAnalysisModel(Base):
    __tablename__ = "sentiment_analyses"

    id = Column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    speech_id = Column(PostgresUUID(as_uuid=True), ForeignKey('speeches.id'), nullable=False)
    hawkish_dovish_score = Column(Float, nullable=False)
    policy_stance = Column(String(20), nullable=False)
    uncertainty_score = Column(Float, nullable=False)
    confidence_score = Column(Float, nullable=False)
    analyzer_version = Column(String(50), nullable=False)
    raw_scores = Column(JSON, nullable=True)
    topic_classifications = Column(JSON, nullable=True)
    analysis_timestamp = Column(DateTime, default=datetime.utcnow)

    # No direct relationships for simplicity (speech object can retrieve by id)

    __table_args__ = (
        Index('idx_analysis_speech', 'speech_id'),
        Index('idx_analysis_version', 'analyzer_version'),
        Index('idx_analysis_timestamp', 'analysis_timestamp'),
        Index('idx_analysis_stance', 'policy_stance'),
        Index('idx_analysis_speech_timestamp', 'speech_id', 'analysis_timestamp'),
    )

class SpeechCollectionModel(Base):
    __tablename__ = "speech_collections"

    id = Column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    tags = Column(JSON, nullable=True)
    speech_ids = Column(JSON, nullable=True)  # List of UUIDs
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index('idx_collection_name', 'name'),
        UniqueConstraint('name', name='uq_collection_name'),
    )
