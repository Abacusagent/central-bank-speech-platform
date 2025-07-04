"""Initial database schema

Revision ID: 001_initial_schema
Revises: 
Create Date: 2025-01-15 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '001_initial_schema'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create initial database schema."""
    
    # Create institutions table
    op.create_table(
        'institutions',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('country', sa.String(), nullable=False),
        sa.Column('website', sa.String(), nullable=True),
        sa.Column('timezone', sa.String(), nullable=True),
        sa.Column('languages', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name')
    )
    
    # Create speakers table
    op.create_table(
        'speakers',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('institution_id', sa.String(), nullable=False),
        sa.Column('role', sa.String(), nullable=True),
        sa.Column('title', sa.String(), nullable=True),
        sa.Column('is_current', sa.Boolean(), nullable=False),
        sa.Column('start_date', sa.Date(), nullable=True),
        sa.Column('end_date', sa.Date(), nullable=True),
        sa.Column('biography', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['institution_id'], ['institutions.id'], ),
        sa.Index('idx_speakers_institution', 'institution_id'),
        sa.Index('idx_speakers_current', 'is_current'),
        sa.Index('idx_speakers_name', 'name')
    )
    
    # Create speeches table
    op.create_table(
        'speeches',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('title', sa.String(), nullable=False),
        sa.Column('speaker_id', sa.String(), nullable=True),
        sa.Column('speaker_name', sa.String(), nullable=False),
        sa.Column('institution_id', sa.String(), nullable=False),
        sa.Column('date', sa.Date(), nullable=False),
        sa.Column('url', sa.String(), nullable=False),
        sa.Column('content_hash', sa.String(), nullable=True),
        sa.Column('speech_type', sa.String(), nullable=True),
        sa.Column('language', sa.String(), nullable=False),
        sa.Column('raw_text', sa.Text(), nullable=True),
        sa.Column('cleaned_text', sa.Text(), nullable=True),
        sa.Column('word_count', sa.Integer(), nullable=True),
        sa.Column('extraction_method', sa.String(), nullable=True),
        sa.Column('extraction_confidence', sa.Float(), nullable=True),
        sa.Column('validation_status', sa.String(), nullable=True),
        sa.Column('validation_confidence', sa.Float(), nullable=True),
        sa.Column('validation_issues', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('extracted_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['institution_id'], ['institutions.id'], ),
        sa.ForeignKeyConstraint(['speaker_id'], ['speakers.id'], ),
        sa.UniqueConstraint('url'),
        sa.Index('idx_speeches_date', 'date'),
        sa.Index('idx_speeches_institution', 'institution_id'),
        sa.Index('idx_speeches_speaker', 'speaker_id'),
        sa.Index('idx_speeches_language', 'language'),
        sa.Index('idx_speeches_validation', 'validation_status'),
        sa.CheckConstraint('word_count >= 0', name='chk_speeches_word_count_positive'),
        sa.CheckConstraint('extraction_confidence >= 0.0 AND extraction_confidence <= 1.0', name='chk_speeches_extraction_confidence_range'),
        sa.CheckConstraint('validation_confidence >= 0.0 AND validation_confidence <= 1.0', name='chk_speeches_validation_confidence_range')
    )
    
    # Create sentiment_analysis table
    op.create_table(
        'sentiment_analysis',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('speech_id', sa.String(), nullable=False),
        sa.Column('analyzer_name', sa.String(), nullable=False),
        sa.Column('analyzer_version', sa.String(), nullable=False),
        sa.Column('analysis_timestamp', sa.DateTime(), nullable=False),
        sa.Column('hawkish_dovish_score', sa.Float(), nullable=True),
        sa.Column('uncertainty_score', sa.Float(), nullable=True),
        sa.Column('confidence_score', sa.Float(), nullable=False),
        sa.Column('topic_tags', sa.JSON(), nullable=True),
        sa.Column('key_phrases', sa.JSON(), nullable=True),
        sa.Column('processing_time_ms', sa.Integer(), nullable=True),
        sa.Column('model_name', sa.String(), nullable=True),
        sa.Column('raw_results', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['speech_id'], ['speeches.id'], ),
        sa.UniqueConstraint('speech_id', 'analyzer_version', name='uq_analysis_speech_version'),
        sa.Index('idx_analysis_speech', 'speech_id'),
        sa.Index('idx_analysis_timestamp', 'analysis_timestamp'),
        sa.Index('idx_analysis_sentiment', 'hawkish_dovish_score'),
        sa.Index('idx_analysis_confidence', 'confidence_score'),
        sa.Index('idx_analysis_uncertainty', 'uncertainty_score'),
        sa.CheckConstraint('hawkish_dovish_score >= -1.0 AND hawkish_dovish_score <= 1.0', name='chk_analysis_sentiment_range'),
        sa.CheckConstraint('uncertainty_score >= 0.0 AND uncertainty_score <= 1.0', name='chk_analysis_uncertainty_range'),
        sa.CheckConstraint('confidence_score >= 0.0 AND confidence_score <= 1.0', name='chk_analysis_confidence_range'),
        sa.CheckConstraint('processing_time_ms >= 0', name='chk_analysis_processing_time_positive')
    )
    
    # Create processing_logs table
    op.create_table(
        'processing_logs',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('operation_type', sa.String(), nullable=False),
        sa.Column('institution_id', sa.String(), nullable=True),
        sa.Column('speech_id', sa.String(), nullable=True),
        sa.Column('status', sa.String(), nullable=False),
        sa.Column('start_time', sa.DateTime(), nullable=False),
        sa.Column('end_time', sa.DateTime(), nullable=True),
        sa.Column('processing_time_ms', sa.Integer(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('error_details', sa.JSON(), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['institution_id'], ['institutions.id'], ),
        sa.ForeignKeyConstraint(['speech_id'], ['speeches.id'], ),
        sa.Index('idx_logs_operation', 'operation_type'),
        sa.Index('idx_logs_status', 'status'),
        sa.Index('idx_logs_institution', 'institution_id'),
        sa.Index('idx_logs_timestamp', 'start_time'),
        sa.CheckConstraint('processing_time_ms >= 0', name='chk_logs_processing_time_positive')
    )


def downgrade() -> None:
    """Drop all tables."""
    op.drop_table('processing_logs')
    op.drop_table('sentiment_analysis')
    op.drop_table('speeches')
    op.drop_table('speakers')
    op.drop_table('institutions')