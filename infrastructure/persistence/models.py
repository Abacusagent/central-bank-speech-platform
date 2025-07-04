"""
SQLAlchemy ORM Models for Central Bank Speech Analysis Platform

This module defines comprehensive database models that map to all core domain entities,
enabling robust, type-safe data persistence with optimal performance characteristics.

Key Features:
- Full mapping of domain entities to database tables
- Optimized indexes for common query patterns
- Proper relationships and foreign key constraints
- Support for full-text search and analytics
- PostgreSQL-specific optimizations
- Audit trails and timestamp tracking

Database Design Principles:
- Normalized structure with appropriate denormalization for performance
- Comprehensive indexing strategy for production workloads
- JSON columns for flexible metadata storage
- Full-text search capabilities for speech content
- Optimized for both OLTP and analytical queries

Author: Central Bank Speech Analysis Platform
Date: 2025
Version: 2.0.0
"""

from sqlalchemy import (
    Column, String, Integer, Float, DateTime, Date, Boolean, Text, JSON,
    ForeignKey, Index, UniqueConstraint, CheckConstraint,
    text, func, Table
)
from sqlalchemy.dialects.postgresql import (
    UUID as PostgresUUID, ARRAY, TSVECTOR, JSONB
)
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import expression
from datetime import datetime
from uuid import uuid4

# Create the declarative base
Base = declarative_base()


class InstitutionModel(Base):
    """
    SQLAlchemy model for Institution entity.
    
    Represents central banking institutions with their core attributes
    and metadata. This is the root entity for organizing speeches and speakers.
    """
    __tablename__ = "institutions"

    # Primary key and core attributes
    code = Column(String(10), primary_key=True, comment="Unique institution identifier")
    name = Column(String(255), nullable=False, comment="Full institution name")
    country = Column(String(100), nullable=False, comment="Country where institution is located")
    institution_type = Column(String(50), nullable=False, comment="Type of central banking institution")
    
    # Optional metadata
    established_date = Column(Date, nullable=True, comment="Date when institution was established")
    website_url = Column(String(500), nullable=True, comment="Primary website URL")
    description = Column(Text, nullable=True, comment="Detailed description of institution")
    
    # Additional metadata for enhanced functionality
    languages = Column(ARRAY(String(5)), nullable=True, comment="Languages used by institution")
    timezone = Column(String(50), nullable=True, comment="Institution timezone")
    contact_info = Column(JSONB, nullable=True, comment="Contact information in JSON format")
    
    # Audit fields
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    speakers = relationship(
        "SpeakerModel", 
        back_populates="institution",
        cascade="all, delete-orphan",
        passive_deletes=True
    )
    speeches = relationship(
        "SpeechModel", 
        back_populates="institution",
        cascade="all, delete-orphan",
        passive_deletes=True
    )

    # Table constraints and indexes
    __table_args__ = (
        Index('idx_institution_country', 'country'),
        Index('idx_institution_type', 'institution_type'),
        Index('idx_institution_name', 'name'),
        Index('idx_institution_updated', 'updated_at'),
        CheckConstraint('length(code) >= 2', name='chk_institution_code_length'),
        CheckConstraint('length(name) >= 3', name='chk_institution_name_length'),
        {'comment': 'Central banking institutions'}
    )


class SpeakerModel(Base):
    """
    SQLAlchemy model for CentralBankSpeaker entity.
    
    Represents speakers at central banking institutions with their roles,
    tenure information, and alternate names for fuzzy matching.
    """
    __tablename__ = "speakers"

    # Primary key
    id = Column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Core speaker information
    name = Column(String(255), nullable=False, comment="Full name of the speaker")
    role = Column(String(100), nullable=False, comment="Role or title at institution")
    institution_code = Column(
        String(10), 
        ForeignKey('institutions.code', ondelete='CASCADE'), 
        nullable=False,
        comment="Institution where speaker works"
    )
    
    # Tenure information
    start_date = Column(Date, nullable=True, comment="Start date of current role")
    end_date = Column(Date, nullable=True, comment="End date of role (if no longer active)")
    voting_member = Column(Boolean, default=False, nullable=False, comment="Whether speaker is a voting member")
    
    # Enhanced speaker metadata
    alternate_names = Column(ARRAY(String(255)), nullable=True, comment="Alternative names/spellings")
    biography = Column(Text, nullable=True, comment="Brief biography")
    education = Column(JSONB, nullable=True, comment="Educational background")
    previous_roles = Column(JSONB, nullable=True, comment="Previous roles and positions")
    
    # Contact and social media
    email = Column(String(255), nullable=True, comment="Official email address")
    social_media = Column(JSONB, nullable=True, comment="Social media profiles")
    
    # Analytics fields
    total_speeches = Column(Integer, default=0, nullable=False, comment="Total number of speeches")
    first_speech_date = Column(Date, nullable=True, comment="Date of first recorded speech")
    last_speech_date = Column(Date, nullable=True, comment="Date of most recent speech")
    
    # Audit fields
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    institution = relationship("InstitutionModel", back_populates="speakers")
    speeches = relationship(
        "SpeechModel", 
        back_populates="speaker",
        cascade="all, delete-orphan",
        passive_deletes=True
    )

    # Table constraints and indexes
    __table_args__ = (
        Index('idx_speaker_name', 'name'),
        Index('idx_speaker_institution', 'institution_code'),
        Index('idx_speaker_role', 'role'),
        Index('idx_speaker_voting', 'voting_member'),
        Index('idx_speaker_active', 'end_date'),  # NULL end_date means active
        Index('idx_speaker_tenure', 'start_date', 'end_date'),
        Index('idx_speaker_speech_count', 'total_speeches'),
        Index('idx_speaker_updated', 'updated_at'),
        # Composite indexes for common queries
        Index('idx_speaker_institution_role', 'institution_code', 'role'),
        Index('idx_speaker_institution_active', 'institution_code', 'end_date'),
        UniqueConstraint('name', 'institution_code', 'start_date', name='uq_speaker_institution_tenure'),
        CheckConstraint('length(name) >= 2', name='chk_speaker_name_length'),
        CheckConstraint('start_date <= end_date OR end_date IS NULL', name='chk_speaker_valid_tenure'),
        {'comment': 'Central bank speakers and officials'}
    )


class SpeechModel(Base):
    """
    SQLAlchemy model for CentralBankSpeech entity.
    
    The core table storing all speech data including metadata, content,
    and analysis results. Optimized for both transactional and analytical queries.
    """
    __tablename__ = "speeches"

    # Primary key
    id = Column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Basic speech metadata (from SpeechMetadata)
    title = Column(String(500), nullable=True, comment="Title of the speech")
    url = Column(String(1000), nullable=True, comment="Source URL of the speech")
    speech_date = Column(Date, nullable=True, comment="Date when speech was delivered")
    location = Column(String(255), nullable=True, comment="Location where speech was delivered")
    language = Column(String(5), default='en', nullable=False, comment="Language of the speech")
    speech_type = Column(String(50), nullable=True, comment="Type of speech (formal, testimony, etc.)")
    
    # Foreign keys
    speaker_id = Column(
        PostgresUUID(as_uuid=True), 
        ForeignKey('speakers.id', ondelete='SET NULL'), 
        nullable=True,
        comment="Speaker who delivered the speech"
    )
    institution_code = Column(
        String(10), 
        ForeignKey('institutions.code', ondelete='CASCADE'), 
        nullable=True,
        comment="Institution associated with the speech"
    )
    
    # Speech content (from SpeechContent)
    raw_text = Column(Text, nullable=True, comment="Original extracted text")
    cleaned_text = Column(Text, nullable=True, comment="Processed and cleaned text")
    word_count = Column(Integer, nullable=True, comment="Number of words in cleaned text")
    
    # Content extraction metadata
    extraction_method = Column(String(50), nullable=True, comment="Method used to extract content")
    extraction_confidence = Column(Float, nullable=True, comment="Confidence in extraction quality")
    extraction_timestamp = Column(DateTime, nullable=True, comment="When content was extracted")
    
    # Content hashing for deduplication
    content_hash_sha256 = Column(String(64), nullable=True, comment="SHA256 hash of cleaned content")
    content_hash_md5 = Column(String(32), nullable=True, comment="MD5 hash of cleaned content")
    
    # Processing status and workflow
    status = Column(String(20), default='discovered', nullable=False, comment="Processing status")
    processing_history = Column(JSONB, nullable=True, comment="History of processing steps")
    error_messages = Column(JSONB, nullable=True, comment="Any error messages during processing")
    
    # Sentiment analysis results (from SentimentAnalysis)
    hawkish_dovish_score = Column(Float, nullable=True, comment="Hawkish-dovish sentiment score (-1 to 1)")
    policy_stance = Column(String(20), nullable=True, comment="Classified policy stance")
    uncertainty_score = Column(Float, nullable=True, comment="Uncertainty score (0 to 1)")
    confidence_score = Column(Float, nullable=True, comment="Analysis confidence score (0 to 1)")
    
    # Analysis metadata
    analysis_timestamp = Column(DateTime, nullable=True, comment="When sentiment analysis was performed")
    analyzer_version = Column(String(50), nullable=True, comment="Version of analyzer used")
    raw_scores = Column(JSONB, nullable=True, comment="Raw analysis scores and intermediate results")
    topic_classifications = Column(ARRAY(String(100)), nullable=True, comment="Identified topics")
    
    # Validation results (from ValidationResult)
    validation_status = Column(String(20), nullable=True, comment="Validation status")
    validation_confidence = Column(Float, nullable=True, comment="Validation confidence")
    validation_issues = Column(JSONB, nullable=True, comment="Validation issues found")
    validation_timestamp = Column(DateTime, nullable=True, comment="When validation was performed")
    
    # Full-text search
    search_vector = Column(TSVECTOR, nullable=True, comment="Full-text search vector")
    
    # Tags and categorization
    tags = Column(ARRAY(String(100)), nullable=True, comment="User-defined tags")
    categories = Column(ARRAY(String(50)), nullable=True, comment="System-assigned categories")
    
    # Audit and tracking
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    last_analyzed_at = Column(DateTime, nullable=True, comment="Last time speech was analyzed")
    
    # Relationships
    speaker = relationship("SpeakerModel", back_populates="speeches")
    institution = relationship("InstitutionModel", back_populates="speeches")
    sentiment_analyses = relationship(
        "SentimentAnalysisModel", 
        back_populates="speech",
        cascade="all, delete-orphan"
    )

    # Table constraints and indexes
    __table_args__ = (
        # Basic indexes for common queries
        Index('idx_speech_date', 'speech_date'),
        Index('idx_speech_speaker', 'speaker_id'),
        Index('idx_speech_institution', 'institution_code'),
        Index('idx_speech_status', 'status'),
        Index('idx_speech_url', 'url'),
        Index('idx_speech_language', 'language'),
        
        # Content and analysis indexes
        Index('idx_speech_content_hash', 'content_hash_sha256'),
        Index('idx_speech_word_count', 'word_count'),
        Index('idx_speech_policy_stance', 'policy_stance'),
        Index('idx_speech_sentiment_score', 'hawkish_dovish_score'),
        Index('idx_speech_confidence', 'confidence_score'),
        Index('idx_speech_uncertainty', 'uncertainty_score'),
        
        # Full-text search index
        Index('idx_speech_search', 'search_vector', postgresql_using='gin'),
        
        # Array indexes for tags and topics
        Index('idx_speech_tags', 'tags', postgresql_using='gin'),
        Index('idx_speech_topics', 'topic_classifications', postgresql_using='gin'),
        
        # Temporal indexes
        Index('idx_speech_updated', 'updated_at'),
        Index('idx_speech_analyzed', 'last_analyzed_at'),
        Index('idx_speech_extraction_time', 'extraction_timestamp'),
        
        # Composite indexes for complex queries
        Index('idx_speech_institution_date', 'institution_code', 'speech_date'),
        Index('idx_speech_speaker_date', 'speaker_id', 'speech_date'),
        Index('idx_speech_status_institution', 'status', 'institution_code'),
        Index('idx_speech_stance_date', 'policy_stance', 'speech_date'),
        Index('idx_speech_analysis_complete', 'hawkish_dovish_score', 'confidence_score'),
        
        # Unique constraints
        UniqueConstraint('url', name='uq_speech_url'),
        UniqueConstraint('content_hash_sha256', name='uq_speech_content_hash'),
        
        # Check constraints
        CheckConstraint('hawkish_dovish_score >= -1.0 AND hawkish_dovish_score <= 1.0', name='chk_speech_sentiment_range'),
        CheckConstraint('uncertainty_score >= 0.0 AND uncertainty_score <= 1.0', name='chk_speech_uncertainty_range'),
        CheckConstraint('confidence_score >= 0.0 AND confidence_score <= 1.0', name='chk_speech_confidence_range'),
        CheckConstraint('word_count >= 0', name='chk_speech_word_count_positive'),
        CheckConstraint('extraction_confidence >= 0.0 AND extraction_confidence <= 1.0', name='chk_speech_extraction_confidence'),
        
        {'comment': 'Central bank speeches with content and analysis results'}
    )


class SentimentAnalysisModel(Base):
    """
    SQLAlchemy model for storing multiple sentiment analysis results per speech.
    
    Supports versioning of analysis results and comparison of different
    analysis approaches or models.
    """
    __tablename__ = "sentiment_analyses"

    # Primary key
    id = Column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Foreign key to speech
    speech_id = Column(
        PostgresUUID(as_uuid=True), 
        ForeignKey('speeches.id', ondelete='CASCADE'), 
        nullable=False,
        comment="Speech that was analyzed"
    )
    
    # Core sentiment metrics
    hawkish_dovish_score = Column(Float, nullable=False, comment="Hawkish-dovish sentiment score (-1 to 1)")
    policy_stance = Column(String(20), nullable=False, comment="Classified policy stance")
    uncertainty_score = Column(Float, nullable=False, comment="Uncertainty score (0 to 1)")
    confidence_score = Column(Float, nullable=False, comment="Analysis confidence score (0 to 1)")
    
    # Analysis metadata
    analyzer_version = Column(String(50), nullable=False, comment="Version of analyzer used")
    analysis_timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    processing_time_ms = Column(Integer, nullable=True, comment="Processing time in milliseconds")
    
    # Detailed results
    raw_scores = Column(JSONB, nullable=True, comment="Raw analysis scores and probabilities")
    topic_classifications = Column(ARRAY(String(100)), nullable=True, comment="Identified topics")
    key_phrases = Column(JSONB, nullable=True, comment="Key phrases that influenced the analysis")
    attention_weights = Column(JSONB, nullable=True, comment="Attention weights for interpretability")
    
    # Model-specific metadata
    model_name = Column(String(100), nullable=True, comment="Name of the analysis model used")
    model_parameters = Column(JSONB, nullable=True, comment="Model parameters and configuration")
    
    # Quality metrics
    validation_score = Column(Float, nullable=True, comment="Validation score if available")
    human_review_score = Column(Float, nullable=True, comment="Human review score if available")
    
    # Audit fields
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    speech = relationship("SpeechModel", back_populates="sentiment_analyses")

    # Table constraints and indexes
    __table_args__ = (
        Index('idx_analysis_speech', 'speech_id'),
        Index('idx_analysis_version', 'analyzer_version'),
        Index('idx_analysis_timestamp', 'analysis_timestamp'),
        Index('idx_analysis_stance', 'policy_stance'),
        Index('idx_analysis_score', 'hawkish_dovish_score'),
        Index('idx_analysis_confidence', 'confidence_score'),
        Index('idx_analysis_uncertainty', 'uncertainty_score'),
        Index('idx_analysis_model', 'model_name'),
        
        # Composite indexes for latest analysis queries
        Index('idx_analysis_speech_timestamp', 'speech_id', 'analysis_timestamp'),
        Index('idx_analysis_version_timestamp', 'analyzer_version', 'analysis_timestamp'),
        
        # Unique constraint for versioning
        UniqueConstraint('speech_id', 'analyzer_version', name='uq_analysis_speech_version'),
        
        # Check constraints
        CheckConstraint('hawkish_dovish_score >= -1.0 AND hawkish_dovish_score <= 1.0', name='chk_analysis_sentiment_range'),
        CheckConstraint('uncertainty_score >= 0.0 AND uncertainty_score <= 1.0', name='chk_analysis_uncertainty_range'),
        CheckConstraint('confidence_score >= 0.0 AND confidence_score <= 1.0', name='chk_analysis_confidence_range'),
        CheckConstraint('processing_time_ms >= 0', name='chk_analysis_processing_time_positive'),
        
        {'comment': 'Sentiment analysis results with versioning support'}
    )


class ProcessingLogModel(Base):
    """
    SQLAlchemy model for tracking processing activities and performance metrics.
    
    Provides audit trail and performance monitoring for the platform.
    """
    __tablename__ = "processing_logs"

    # Primary key
    id = Column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Log metadata
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    operation_type = Column(String(50), nullable=False, comment="Type of operation (discovery, extraction, analysis)")
    component = Column(String(100), nullable=False, comment="Component that performed the operation")
    
    # Context information
    institution_code = Column(String(10), nullable=True, comment="Institution context")
    speech_id = Column(PostgresUUID(as_uuid=True), nullable=True, comment="Speech context if applicable")
    plugin_name = Column(String(100), nullable=True, comment="Plugin name if applicable")
    
    # Performance metrics
    duration_ms = Column(Integer, nullable=True, comment="Operation duration in milliseconds")
    memory_usage_mb = Column(Float, nullable=True, comment="Peak memory usage in MB")
    cpu_usage_percent = Column(Float, nullable=True, comment="CPU usage percentage")
    
    # Results and status
    status = Column(String(20), nullable=False, comment="Operation status (success, failure, warning)")
    items_processed = Column(Integer, nullable=True, comment="Number of items processed")
    errors_count = Column(Integer, default=0, nullable=False, comment="Number of errors encountered")
    warnings_count = Column(Integer, default=0, nullable=False, comment="Number of warnings generated")
    
    # Detailed information
    message = Column(Text, nullable=True, comment="Log message or description")
    error_details = Column(JSONB, nullable=True, comment="Error details and stack traces")
    metadata = Column(JSONB, nullable=True, comment="Additional operation metadata")
    
    # Table constraints and indexes
    __table_args__ = (
        Index('idx_log_timestamp', 'timestamp'),
        Index('idx_log_operation', 'operation_type'),
        Index('idx_log_component', 'component'),
        Index('idx_log_status', 'status'),
        Index('idx_log_institution', 'institution_code'),
        Index('idx_log_speech', 'speech_id'),
        Index('idx_log_plugin', 'plugin_name'),
        
        # Composite indexes for performance analysis
        Index('idx_log_operation_timestamp', 'operation_type', 'timestamp'),
        Index('idx_log_component_timestamp', 'component', 'timestamp'),
        Index('idx_log_status_timestamp', 'status', 'timestamp'),
        
        {'comment': 'Processing logs and performance metrics'}
    )


# Database functions and triggers (PostgreSQL-specific)

# Function to update the search vector
update_search_vector_sql = """
    CREATE OR REPLACE FUNCTION update_speech_search_vector()
    RETURNS TRIGGER AS $$
    BEGIN
        NEW.search_vector := to_tsvector('english', 
            COALESCE(NEW.title, '') || ' ' || 
            COALESCE(NEW.cleaned_text, '') || ' ' ||
            array_to_string(COALESCE(NEW.tags, ARRAY[]::text[]), ' ') || ' ' ||
            array_to_string(COALESCE(NEW.topic_classifications, ARRAY[]::text[]), ' ')
        );
        RETURN NEW;
    END;
    $$ LANGUAGE plpgsql;
"""

# Trigger to automatically update search vector
create_search_trigger_sql = """
    DROP TRIGGER IF EXISTS trigger_update_speech_search_vector ON speeches;
    CREATE TRIGGER trigger_update_speech_search_vector
        BEFORE INSERT OR UPDATE ON speeches
        FOR EACH ROW
        EXECUTE FUNCTION update_speech_search_vector();
"""

# Function to update speaker statistics
update_speaker_stats_sql = """
    CREATE OR REPLACE FUNCTION update_speaker_statistics()
    RETURNS TRIGGER AS $$
    BEGIN
        IF TG_OP = 'INSERT' OR TG_OP = 'UPDATE' THEN
            UPDATE speakers SET
                total_speeches = (
                    SELECT COUNT(*) FROM speeches 
                    WHERE speaker_id = NEW.speaker_id AND speaker_id IS NOT NULL
                ),
                first_speech_date = (
                    SELECT MIN(speech_date) FROM speeches 
                    WHERE speaker_id = NEW.speaker_id AND speech_date IS NOT NULL
                ),
                last_speech_date = (
                    SELECT MAX(speech_date) FROM speeches 
                    WHERE speaker_id = NEW.speaker_id AND speech_date IS NOT NULL
                )
            WHERE id = NEW.speaker_id;
        END IF;
        
        IF TG_OP = 'DELETE' AND OLD.speaker_id IS NOT NULL THEN
            UPDATE speakers SET
                total_speeches = (
                    SELECT COUNT(*) FROM speeches 
                    WHERE speaker_id = OLD.speaker_id
                ),
                first_speech_date = (
                    SELECT MIN(speech_date) FROM speeches 
                    WHERE speaker_id = OLD.speaker_id AND speech_date IS NOT NULL
                ),
                last_speech_date = (
                    SELECT MAX(speech_date) FROM speeches 
                    WHERE speaker_id = OLD.speaker_id AND speech_date IS NOT NULL
                )
            WHERE id = OLD.speaker_id;
        END IF;
        
        RETURN COALESCE(NEW, OLD);
    END;
    $$ LANGUAGE plpgsql;
"""

# Trigger to automatically update speaker statistics
create_speaker_stats_trigger_sql = """
    DROP TRIGGER IF EXISTS trigger_update_speaker_statistics ON speeches;
    CREATE TRIGGER trigger_update_speaker_statistics
        AFTER INSERT OR UPDATE OR DELETE ON speeches
        FOR EACH ROW
        EXECUTE FUNCTION update_speaker_statistics();
"""

# Views for common analytical queries

create_speech_summary_view_sql = """
    CREATE OR REPLACE VIEW speech_summary AS
    SELECT 
        s.id,
        s.title,
        s.speech_date,
        s.institution_code,
        i.name as institution_name,
        sp.name as speaker_name,
        sp.role as speaker_role,
        s.word_count,
        s.policy_stance,
        s.hawkish_dovish_score,
        s.uncertainty_score,
        s.confidence_score,
        s.status,
        s.created_at,
        s.updated_at
    FROM speeches s
    LEFT JOIN institutions i ON s.institution_code = i.code
    LEFT JOIN speakers sp ON s.speaker_id = sp.id;
"""

create_institution_stats_view_sql = """
    CREATE OR REPLACE VIEW institution_statistics AS
    SELECT 
        i.code,
        i.name,
        i.country,
        COUNT(s.id) as total_speeches,
        COUNT(DISTINCT s.speaker_id) as unique_speakers,
        MIN(s.speech_date) as first_speech_date,
        MAX(s.speech_date) as last_speech_date,
        AVG(s.hawkish_dovish_score) as avg_sentiment,
        AVG(s.uncertainty_score) as avg_uncertainty,
        COUNT(CASE WHEN s.status = 'analyzed' THEN 1 END) as analyzed_speeches
    FROM institutions i
    LEFT JOIN speeches s ON i.code = s.institution_code
    GROUP BY i.code, i.name, i.country;
"""


# SQL statements to be executed during database initialization
DDL_STATEMENTS = [
    update_search_vector_sql,
    create_search_trigger_sql,
    update_speaker_stats_sql,
    create_speaker_stats_trigger_sql,
    create_speech_summary_view_sql,
    create_institution_stats_view_sql
]