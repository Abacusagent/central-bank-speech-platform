# tools/cli.py

"""
Central Bank Speech Analysis CLI Tool

Provides a user-friendly command line interface to collect speeches,
run NLP analysis, check system health, and view metrics.

Usage:
    python tools/cli.py collect --institution FED --start 2024-01-01 --end 2024-03-31
    python tools/cli.py analyze --institution FED
    python tools/cli.py health
    python tools/cli.py plugins

Author: Central Bank Speech Analysis Platform
Date: 2025
"""

import typer
import asyncio
from datetime import datetime
import logging

# Project imports (adapt as needed for your structure)
from config.settings import settings
from domain.value_objects import DateRange
from application.orchestrators.speech_collection import SpeechCollectionOrchestrator
from application.services.analysis_service import SpeechAnalysisService
from infrastructure.nlp.pipeline import NLPProcessingPipeline
from infrastructure.persistence.uow import SqlAlchemyUnitOfWork
from infrastructure.monitoring.metrics import start_metrics_server, set_system_health

# Example: SQLAlchemy async session factory (implement for your project)
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession

app = typer.Typer(add_completion=False)
logging.basicConfig(level=settings.LOG_LEVEL)

def get_async_session():
    engine = create_async_engine(settings.DATABASE_URL, echo=False, future=True)
    return AsyncSession(engine, expire_on_commit=False)

@app.command()
def collect(
    institution: str = typer.Option(..., help="Institution code (e.g., FED, ECB, BOJ)"),
    start: str = typer.Option(..., help="Start date YYYY-MM-DD"),
    end: str = typer.Option(..., help="End date YYYY-MM-DD"),
    limit: int = typer.Option(None, help="Optional limit of speeches per institution"),
    skip_existing: bool = typer.Option(True, help="Skip already collected speeches")
):
    """
    Discover and collect speeches for a given institution and date range.
    """
    async def run():
        async with get_async_session() as session:
            uow = SqlAlchemyUnitOfWork(session)
            orchestrator = SpeechCollectionOrchestrator(uow)
            # Dynamically register enabled plugins
            # (This assumes you have a plugin loader—adapt as needed)
            from plugins import load_plugins
            for plugin in load_plugins(settings.ENABLED_PLUGINS):
                orchestrator.register_plugin(plugin)
            dr = DateRange(
                start_date=datetime.strptime(start, "%Y-%m-%d").date(),
                end_date=datetime.strptime(end, "%Y-%m-%d").date()
            )
            result = await orchestrator.collect_by_institution(
                institution_code=institution,
                date_range=dr,
                limit=limit,
                skip_existing=skip_existing
            )
            typer.echo(result)
    asyncio.run(run())

@app.command()
def analyze(
    institution: str = typer.Option(..., help="Institution code (e.g., FED, ECB, BOJ)"),
    batch_size: int = typer.Option(10, help="Batch size for analysis"),
    persist: bool = typer.Option(True, help="Persist NLP analysis results to DB")
):
    """
    Run NLP analysis on all speeches for a given institution.
    """
    async def run():
        async with get_async_session() as session:
            uow = SqlAlchemyUnitOfWork(session)
            nlp_pipeline = NLPProcessingPipeline()
            analysis_service = SpeechAnalysisService(uow, nlp_pipeline)
            # Fetch all speeches (implement as needed)
            speeches = await uow.speeches.find_by_institution(institution)
            for i in range(0, len(speeches), batch_size):
                batch = speeches[i:i+batch_size]
                result = await analysis_service.analyze_speeches(batch, persist=persist)
                typer.echo(f"Batch {i//batch_size + 1}: {result['aggregate']}")
    asyncio.run(run())

@app.command()
def health():
    """
    Runs a basic system health check (DB connection, plugin availability).
    """
    try:
        # Basic check: DB connection
        async def check_db():
            async with get_async_session() as session:
                await session.execute("SELECT 1")
        asyncio.run(check_db())
        set_system_health(True)
        typer.echo("System health: OK")
    except Exception as e:
        set_system_health(False)
        typer.echo(f"System health: FAIL ({e})")

@app.command()
def plugins():
    """
    Lists all enabled plugins.
    """
    typer.echo("Enabled plugins:")
    for p in settings.ENABLED_PLUGINS:
        typer.echo(f" - {p}")

@app.command()
def start_metrics(
    port: int = typer.Option(9100, help="Port to expose /metrics (Prometheus scrape endpoint)")
):
    """
    Starts the Prometheus metrics server (exposes /metrics).
    """
    start_metrics_server(port)
    typer.echo(f"Metrics server started at http://localhost:{port}/metrics")

if __name__ == "__main__":
    app()
