# infrastructure/persistence/uow.py

"""
Unit of Work implementation for Central Bank Speech Analysis Platform.

Provides transactional consistency across all repository operations with comprehensive
error handling, logging, and proper async resource management. This is the cornerstone
of the persistence layer that ensures ACID properties and maintains data integrity.

Key Features:
- Async context manager with proper resource cleanup
- Lazy repository initialization for performance
- Comprehensive error handling with domain exceptions
- Transaction lifecycle logging and monitoring
- Support for nested transactions and savepoints
- Connection health monitoring and recovery

Author: Central Bank Speech Analysis Platform
Date: 2025
Version: 2.0.0
"""

import logging
import asyncio
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError, IntegrityError, OperationalError
from sqlalchemy import text

from infrastructure.persistence.repository_implementations import (
    SqlAlchemySpeechRepository,
    SqlAlchemySpeakerRepository,
    SqlAlchemyInstitutionRepository,
    SqlAlchemyAnalysisRepository
)
from domain.repositories import (
    UnitOfWork, RepositoryError, EntityNotFoundError, 
    DuplicateEntityError, ConcurrencyError
)

logger = logging.getLogger(__name__)


class SqlAlchemyUnitOfWork(UnitOfWork):
    """
    SQLAlchemy implementation of Unit of Work pattern.
    
    Coordinates transactional operations across multiple repositories ensuring
    ACID properties, proper error handling, and consistent state management.
    Implements lazy loading of repositories for optimal performance.
    """
    
    def __init__(self, session: AsyncSession):
        """
        Initialize Unit of Work with async session.
        
        Args:
            session: SQLAlchemy async session for database operations
        """
        self.session = session
        self._transaction_id = id(session)
        self._is_committed = False
        self._is_rolled_back = False
        self._repositories_initialized = False
        
        # Lazy-loaded repositories
        self._speeches: Optional[SqlAlchemySpeechRepository] = None
        self._speakers: Optional[SqlAlchemySpeakerRepository] = None
        self._institutions: Optional[SqlAlchemyInstitutionRepository] = None
        self._analyses: Optional[SqlAlchemyAnalysisRepository] = None
        
        # Transaction tracking
        self._savepoints: Dict[str, Any] = {}
        self._operation_count = 0
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.debug(f"UnitOfWork initialized with transaction ID: {self._transaction_id}")
    
    @property
    def speeches(self) -> SqlAlchemySpeechRepository:
        """
        Get speech repository with lazy initialization.
        
        Returns:
            Speech repository instance
        """
        if self._speeches is None:
            self._speeches = SqlAlchemySpeechRepository(self.session)
            self.logger.debug("Speech repository initialized")
        return self._speeches
    
    @property
    def speakers(self) -> SqlAlchemySpeakerRepository:
        """
        Get speaker repository with lazy initialization.
        
        Returns:
            Speaker repository instance
        """
        if self._speakers is None:
            self._speakers = SqlAlchemySpeakerRepository(self.session)
            self.logger.debug("Speaker repository initialized")
        return self._speakers
    
    @property
    def institutions(self) -> SqlAlchemyInstitutionRepository:
        """
        Get institution repository with lazy initialization.
        
        Returns:
            Institution repository instance
        """
        if self._institutions is None:
            self._institutions = SqlAlchemyInstitutionRepository(self.session)
            self.logger.debug("Institution repository initialized")
        return self._institutions
    
    @property
    def analyses(self) -> SqlAlchemyAnalysisRepository:
        """
        Get analysis repository with lazy initialization.
        
        Returns:
            Analysis repository instance
        """
        if self._analyses is None:
            self._analyses = SqlAlchemyAnalysisRepository(self.session)
            self.logger.debug("Analysis repository initialized")
        return self._analyses
    
    @property
    def transaction_id(self) -> int:
        """Get unique transaction identifier."""
        return self._transaction_id
    
    @property
    def is_active(self) -> bool:
        """Check if transaction is still active."""
        return not (self._is_committed or self._is_rolled_back)
    
    @property
    def operation_count(self) -> int:
        """Get number of operations performed in this transaction."""
        return self._operation_count
    
    async def __aenter__(self):
        """
        Enter async context manager.
        
        Returns:
            Self for use in with statement
        """
        try:
            # Verify session is active
            if not self.session.is_active:
                raise RepositoryError("Session is not active")
            
            # Test database connectivity
            await self._test_connection()
            
            self.logger.debug(f"Entering UnitOfWork context - Transaction ID: {self._transaction_id}")
            return self
            
        except Exception as e:
            self.logger.error(f"Error entering UnitOfWork context: {e}")
            await self._cleanup_on_error()
            raise RepositoryError(f"Failed to enter UnitOfWork context: {e}")
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Exit async context manager with proper cleanup.
        
        Args:
            exc_type: Exception type if any
            exc_val: Exception value if any 
            exc_tb: Exception traceback if any
        """
        try:
            if exc_type is not None:
                self.logger.warning(
                    f"Exception in UnitOfWork context: {exc_type.__name__}: {exc_val}"
                )
                await self.rollback()
            else:
                await self.commit()
                
        except Exception as cleanup_error:
            self.logger.error(f"Error during UnitOfWork cleanup: {cleanup_error}")
            try:
                await self.rollback()
            except Exception as rollback_error:
                self.logger.critical(f"Failed to rollback during cleanup: {rollback_error}")
                
        finally:
            await self._cleanup_resources()
    
    async def commit(self) -> None:
        """
        Commit all staged operations to the database.
        
        Raises:
            RepositoryError: If commit fails
            ConcurrencyError: If concurrent modification detected
        """
        if self._is_committed:
            self.logger.warning("Attempted to commit already committed transaction")
            return
            
        if self._is_rolled_back:
            raise RepositoryError("Cannot commit rolled back transaction")
        
        try:
            self.logger.debug(f"Committing transaction {self._transaction_id} with {self._operation_count} operations")
            
            # Flush to detect any constraint violations before commit
            await self.session.flush()
            
            # Commit the transaction
            await self.session.commit()
            
            self._is_committed = True
            self.logger.info(f"Transaction {self._transaction_id} committed successfully")
            
        except IntegrityError as e:
            self.logger.error(f"Integrity constraint violation during commit: {e}")
            await self._handle_commit_error()
            raise DuplicateEntityError(f"Data integrity violation: {e}")
            
        except OperationalError as e:
            self.logger.error(f"Database operational error during commit: {e}")
            await self._handle_commit_error()
            raise ConcurrencyError(f"Database operational error: {e}")
            
        except SQLAlchemyError as e:
            self.logger.error(f"SQLAlchemy error during commit: {e}")
            await self._handle_commit_error()
            raise RepositoryError(f"Database commit failed: {e}")
            
        except Exception as e:
            self.logger.error(f"Unexpected error during commit: {e}")
            await self._handle_commit_error()
            raise RepositoryError(f"Unexpected commit error: {e}")
    
    async def rollback(self) -> None:
        """
        Rollback all staged operations in the current transaction.
        
        Raises:
            RepositoryError: If rollback fails
        """
        if self._is_rolled_back:
            self.logger.warning("Attempted to rollback already rolled back transaction")
            return
            
        if self._is_committed:
            self.logger.warning("Cannot rollback committed transaction")
            return
        
        try:
            self.logger.debug(f"Rolling back transaction {self._transaction_id}")
            
            # Rollback the transaction
            await self.session.rollback()
            
            self._is_rolled_back = True
            self.logger.info(f"Transaction {self._transaction_id} rolled back successfully")
            
        except SQLAlchemyError as e:
            self.logger.error(f"SQLAlchemy error during rollback: {e}")
            raise RepositoryError(f"Database rollback failed: {e}")
            
        except Exception as e:
            self.logger.error(f"Unexpected error during rollback: {e}")
            raise RepositoryError(f"Unexpected rollback error: {e}")
    
    async def flush(self) -> None:
        """
        Flush pending changes to the database without committing.
        
        This sends all pending SQL statements to the database but keeps
        the transaction open. Useful for generating IDs or detecting
        constraint violations early.
        
        Raises:
            RepositoryError: If flush fails
        """
        if not self.is_active:
            raise RepositoryError("Cannot flush inactive transaction")
        
        try:
            self.logger.debug(f"Flushing transaction {self._transaction_id}")
            await self.session.flush()
            self._operation_count += 1
            self.logger.debug("Flush completed successfully")
            
        except IntegrityError as e:
            self.logger.error(f"Integrity constraint violation during flush: {e}")
            raise DuplicateEntityError(f"Data integrity violation: {e}")
            
        except SQLAlchemyError as e:
            self.logger.error(f"SQLAlchemy error during flush: {e}")
            raise RepositoryError(f"Database flush failed: {e}")
            
        except Exception as e:
            self.logger.error(f"Unexpected error during flush: {e}")
            raise RepositoryError(f"Unexpected flush error: {e}")
    
    async def refresh(self, entity) -> None:
        """
        Refresh an entity's state from the database.
        
        Args:
            entity: Entity to refresh
            
        Raises:
            RepositoryError: If refresh fails
            EntityNotFoundError: If entity no longer exists
        """
        if not self.is_active:
            raise RepositoryError("Cannot refresh entity in inactive transaction")
        
        try:
            self.logger.debug(f"Refreshing entity: {type(entity).__name__}")
            await self.session.refresh(entity)
            self.logger.debug("Entity refreshed successfully")
            
        except SQLAlchemyError as e:
            self.logger.error(f"SQLAlchemy error during refresh: {e}")
            if "could not refresh" in str(e).lower():
                raise EntityNotFoundError(f"Entity no longer exists: {e}")
            raise RepositoryError(f"Entity refresh failed: {e}")
            
        except Exception as e:
            self.logger.error(f"Unexpected error during refresh: {e}")
            raise RepositoryError(f"Unexpected refresh error: {e}")
    
    async def create_savepoint(self, name: str) -> None:
        """
        Create a savepoint for nested transaction support.
        
        Args:
            name: Savepoint name
            
        Raises:
            RepositoryError: If savepoint creation fails
        """
        if not self.is_active:
            raise RepositoryError("Cannot create savepoint in inactive transaction")
            
        if name in self._savepoints:
            raise RepositoryError(f"Savepoint '{name}' already exists")
        
        try:
            self.logger.debug(f"Creating savepoint: {name}")
            savepoint = await self.session.begin_nested()
            self._savepoints[name] = savepoint
            self.logger.debug(f"Savepoint '{name}' created successfully")
            
        except SQLAlchemyError as e:
            self.logger.error(f"SQLAlchemy error creating savepoint: {e}")
            raise RepositoryError(f"Savepoint creation failed: {e}")
            
        except Exception as e:
            self.logger.error(f"Unexpected error creating savepoint: {e}")
            raise RepositoryError(f"Unexpected savepoint creation error: {e}")
    
    async def rollback_to_savepoint(self, name: str) -> None:
        """
        Rollback to a specific savepoint.
        
        Args:
            name: Savepoint name
            
        Raises:
            RepositoryError: If rollback to savepoint fails
        """
        if not self.is_active:
            raise RepositoryError("Cannot rollback to savepoint in inactive transaction")
            
        if name not in self._savepoints:
            raise RepositoryError(f"Savepoint '{name}' does not exist")
        
        try:
            self.logger.debug(f"Rolling back to savepoint: {name}")
            savepoint = self._savepoints[name]
            await savepoint.rollback()
            del self._savepoints[name]
            self.logger.debug(f"Rolled back to savepoint '{name}' successfully")
            
        except SQLAlchemyError as e:
            self.logger.error(f"SQLAlchemy error rolling back to savepoint: {e}")
            raise RepositoryError(f"Savepoint rollback failed: {e}")
            
        except Exception as e:
            self.logger.error(f"Unexpected error rolling back to savepoint: {e}")
            raise RepositoryError(f"Unexpected savepoint rollback error: {e}")
    
    async def release_savepoint(self, name: str) -> None:
        """
        Release a savepoint (commit nested transaction).
        
        Args:
            name: Savepoint name
            
        Raises:
            RepositoryError: If savepoint release fails
        """
        if not self.is_active:
            raise RepositoryError("Cannot release savepoint in inactive transaction")
            
        if name not in self._savepoints:
            raise RepositoryError(f"Savepoint '{name}' does not exist")
        
        try:
            self.logger.debug(f"Releasing savepoint: {name}")
            savepoint = self._savepoints[name]
            await savepoint.commit()
            del self._savepoints[name]
            self.logger.debug(f"Savepoint '{name}' released successfully")
            
        except SQLAlchemyError as e:
            self.logger.error(f"SQLAlchemy error releasing savepoint: {e}")
            raise RepositoryError(f"Savepoint release failed: {e}")
            
        except Exception as e:
            self.logger.error(f"Unexpected error releasing savepoint: {e}")
            raise RepositoryError(f"Unexpected savepoint release error: {e}")
    
    async def execute_raw_sql(self, sql: str, parameters: Optional[Dict[str, Any]] = None) -> Any:
        """
        Execute raw SQL within the transaction context.
        
        Args:
            sql: SQL statement to execute
            parameters: Optional parameters for the query
            
        Returns:
            Query result
            
        Raises:
            RepositoryError: If SQL execution fails
        """
        if not self.is_active:
            raise RepositoryError("Cannot execute SQL in inactive transaction")
        
        try:
            self.logger.debug(f"Executing raw SQL: {sql[:100]}...")
            result = await self.session.execute(text(sql), parameters or {})
            self._operation_count += 1
            self.logger.debug("Raw SQL executed successfully")
            return result
            
        except SQLAlchemyError as e:
            self.logger.error(f"SQLAlchemy error executing raw SQL: {e}")
            raise RepositoryError(f"Raw SQL execution failed: {e}")
            
        except Exception as e:
            self.logger.error(f"Unexpected error executing raw SQL: {e}")
            raise RepositoryError(f"Unexpected raw SQL execution error: {e}")
    
    async def get_transaction_info(self) -> Dict[str, Any]:
        """
        Get comprehensive transaction information.
        
        Returns:
            Dictionary with transaction details
        """
        return {
            'transaction_id': self._transaction_id,
            'is_active': self.is_active,
            'is_committed': self._is_committed,
            'is_rolled_back': self._is_rolled_back,
            'operation_count': self._operation_count,
            'savepoints': list(self._savepoints.keys()),
            'repositories_initialized': {
                'speeches': self._speeches is not None,
                'speakers': self._speakers is not None,
                'institutions': self._institutions is not None,
                'analyses': self._analyses is not None
            }
        }
    
    # Private helper methods
    
    async def _test_connection(self) -> None:
        """Test database connection health."""
        try:
            await self.session.execute(text("SELECT 1"))
            self.logger.debug("Database connection test passed")
        except Exception as e:
            self.logger.error(f"Database connection test failed: {e}")
            raise RepositoryError(f"Database connection failed: {e}")
    
    async def _handle_commit_error(self) -> None:
        """Handle errors during commit by attempting rollback."""
        try:
            await self.rollback()
        except Exception as rollback_error:
            self.logger.critical(f"Failed to rollback after commit error: {rollback_error}")
    
    async def _cleanup_on_error(self) -> None:
        """Clean up resources when entering context fails."""
        try:
            if hasattr(self, 'session') and self.session.is_active:
                await self.session.close()
        except Exception as e:
            self.logger.error(f"Error during cleanup on error: {e}")
    
    async def _cleanup_resources(self) -> None:
        """Clean up all resources and close session."""
        try:
            # Release any remaining savepoints
            for name in list(self._savepoints.keys()):
                try:
                    await self.release_savepoint(name)
                except Exception as e:
                    self.logger.warning(f"Error releasing savepoint {name}: {e}")
            
            # Close session
            if hasattr(self, 'session') and self.session.is_active:
                await self.session.close()
                self.logger.debug("Session closed successfully")
                
        except Exception as e:
            self.logger.error(f"Error during resource cleanup: {e}")


@asynccontextmanager
async def unit_of_work_context(session: AsyncSession):
    """
    Async context manager for Unit of Work pattern.
    
    Args:
        session: SQLAlchemy async session
        
    Yields:
        UnitOfWork instance
        
    Example:
        async with unit_of_work_context(session) as uow:
            speech = await uow.speeches.get_by_id(speech_id)
            speech.status = SpeechStatus.ANALYZED
            await uow.speeches.save(speech)
            # Transaction commits automatically
    """
    async with SqlAlchemyUnitOfWork(session) as uow:
        yield uow


# Health check and monitoring functions

async def check_unit_of_work_health(session: AsyncSession) -> Dict[str, Any]:
    """
    Check Unit of Work health and functionality.
    
    Args:
        session: SQLAlchemy async session
        
    Returns:
        Health check results
    """
    try:
        async with SqlAlchemyUnitOfWork(session) as uow:
            # Test basic functionality
            info = await uow.get_transaction_info()
            
            # Test connection
            await uow.execute_raw_sql("SELECT 1")
            
            return {
                'status': 'healthy',
                'transaction_info': info,
                'connectivity': True,
                'error': None
            }
            
    except Exception as e:
        return {
            'status': 'unhealthy',
            'transaction_info': None,
            'connectivity': False,
            'error': str(e)
        }


# Export public interface
__all__ = [
    'SqlAlchemyUnitOfWork',
    'unit_of_work_context',
    'check_unit_of_work_health'
]