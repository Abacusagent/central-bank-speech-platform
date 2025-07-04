"""
Central Bank Speech Analysis API

FastAPI-based REST API for the Central Bank Speech Analysis Platform.
Provides endpoints for speech collection, analysis, and querying.
"""

from fastapi import FastAPI, Depends, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional, Dict, Any
from datetime import date, datetime
from pydantic import BaseModel, Field

# Project imports
from config.settings import Settings
from domain.value_objects import DateRange
from application.orchestrators.speech_collection import SpeechCollectionOrchestrator
from application.services.analysis_service import SpeechAnalysisService
from infrastructure.persistence.uow import SqlAlchemyUnitOfWork
from infrastructure.persistence.repository_implementations import create_async_engine_from_url

# Initialize FastAPI app
app = FastAPI(
    title="Central Bank Speech Analysis API",
    description="Production-grade API for central bank speech collection and analysis",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
settings = Settings()

# Database setup
engine = create_async_engine_from_url(settings.get_database_url())

async def get_session() -> AsyncSession:
    """Dependency to get database session."""
    async with AsyncSession(engine) as session:
        yield session

async def get_uow(session: AsyncSession = Depends(get_session)) -> SqlAlchemyUnitOfWork:
    """Dependency to get Unit of Work."""
    return SqlAlchemyUnitOfWork(session)

# Request/Response models
class CollectionRequest(BaseModel):
    institution: str = Field(..., description="Institution code (FED, ECB, BOE, etc.)")
    start_date: date = Field(..., description="Start date for collection")
    end_date: date = Field(..., description="End date for collection")
    limit: Optional[int] = Field(None, description="Maximum number of speeches to collect")

class CollectionResponse(BaseModel):
    status: str
    institution: str
    speeches_found: int
    speeches_collected: int
    processing_time_seconds: float
    errors: List[str] = []

class AnalysisRequest(BaseModel):
    institution: Optional[str] = None
    speech_ids: Optional[List[str]] = None
    reanalyze: bool = False

class AnalysisResponse(BaseModel):
    status: str
    speeches_analyzed: int
    processing_time_seconds: float
    errors: List[str] = []

# API Endpoints
@app.post("/collect", response_model=CollectionResponse)
async def collect_speeches(
    request: CollectionRequest,
    background_tasks: BackgroundTasks,
    uow: SqlAlchemyUnitOfWork = Depends(get_uow)
):
    """
    Collect speeches from a central bank within a date range.
    
    This endpoint initiates speech collection from the specified institution.
    Processing happens in the background and results are stored in the database.
    """
    start_time = datetime.now()
    
    try:
        # Create orchestrator
        orchestrator = SpeechCollectionOrchestrator(uow)
        
        # Build date range
        date_range = DateRange(request.start_date, request.end_date)
        
        # Start collection (this could be made async in background)
        result = await orchestrator.collect_institution_speeches(
            institution_code=request.institution,
            date_range=date_range,
            limit=request.limit
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return CollectionResponse(
            status="success",
            institution=request.institution,
            speeches_found=result.get("discovered", 0),
            speeches_collected=result.get("collected", 0),
            processing_time_seconds=processing_time,
            errors=result.get("errors", [])
        )
        
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return CollectionResponse(
            status="error",
            institution=request.institution,
            speeches_found=0,
            speeches_collected=0,
            processing_time_seconds=processing_time,
            errors=[str(e)]
        )

@app.get("/institutions")
async def list_institutions():
    """List available central bank institutions."""
    return {
        "institutions": [
            {"code": "FED", "name": "Federal Reserve System", "country": "United States"},
            {"code": "ECB", "name": "European Central Bank", "country": "European Union"},
            {"code": "BOE", "name": "Bank of England", "country": "United Kingdom"},
            {"code": "BOJ", "name": "Bank of Japan", "country": "Japan"},
            {"code": "BIS", "name": "Bank for International Settlements", "country": "International"},
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0"
    }

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_speeches(
    request: AnalysisRequest,
    uow: SqlAlchemyUnitOfWork = Depends(get_uow)
):
    """
    Analyze collected speeches with NLP pipeline.
    
    Runs sentiment analysis, topic modeling, and other NLP processes
    on speeches from the specified institution.
    """
    start_time = datetime.now()
    
    try:
        # Create analysis service
        analysis_service = SpeechAnalysisService(uow)
        
        # Run analysis
        result = await analysis_service.analyze_institution_speeches(
            institution_code=request.institution,
            speech_ids=request.speech_ids,
            force_reanalyze=request.reanalyze
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return AnalysisResponse(
            status="success",
            speeches_analyzed=result.get("analyzed", 0),
            processing_time_seconds=processing_time,
            errors=result.get("errors", [])
        )
        
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return AnalysisResponse(
            status="error",
            speeches_analyzed=0,
            processing_time_seconds=processing_time,
            errors=[str(e)]
        )

@app.get("/speeches")
async def get_speeches(
    institution: Optional[str] = Query(None, description="Filter by institution"),
    limit: int = Query(50, description="Maximum number of speeches to return"),
    uow: SqlAlchemyUnitOfWork = Depends(get_uow)
):
    """
    Get speeches with optional filtering.
    
    Returns a list of speeches from the database with optional
    filtering by institution and other criteria.
    """
    try:
        # Get speeches from repository
        speeches = await uow.speeches.get_recent_speeches(
            institution_code=institution,
            limit=limit
        )
        
        # Convert to response format
        speech_data = []
        for speech in speeches:
            speech_data.append({
                "id": str(speech.id),
                "title": speech.title,
                "speaker": speech.speaker_name,
                "institution": speech.institution_code,
                "date": speech.date.isoformat(),
                "url": str(speech.url),
                "content_length": len(speech.content) if speech.content else 0
            })
        
        return {
            "speeches": speech_data,
            "count": len(speech_data)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)