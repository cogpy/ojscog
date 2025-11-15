"""
unified_state_manager.py - Unified State Management System

Manages state across OJS MySQL database and agent memory stores,
implementing ontogenetic looms for learning and adaptation.
"""

from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, DateTime, JSON, Float, Boolean, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import QueuePool
import redis
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from contextlib import contextmanager
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnifiedStateManager:
    """
    Manage state across OJS database and agent memory stores.
    
    Implements three-tier state management:
    1. Hot cache (Redis) - Fast access, TTL-based
    2. Warm storage (MySQL) - Persistent, queryable
    3. Cold storage (SQLite) - Agent-specific long-term memory
    
    This architecture supports the ontogenetic looms concept by
    maintaining state across multiple timescales and contexts.
    """
    
    def __init__(self, 
                 ojs_db_url: str,
                 redis_url: str,
                 cache_ttl: int = 300,
                 pool_size: int = 10):
        """
        Initialize unified state manager.
        
        Args:
            ojs_db_url: SQLAlchemy database URL for OJS MySQL
            redis_url: Redis connection URL
            cache_ttl: Default cache TTL in seconds
            pool_size: Database connection pool size
        """
        # Database connection
        self.ojs_engine = create_engine(
            ojs_db_url,
            poolclass=QueuePool,
            pool_size=pool_size,
            max_overflow=20,
            pool_pre_ping=True
        )
        self.SessionLocal = sessionmaker(bind=self.ojs_engine)
        self.metadata = MetaData()
        
        # Redis cache
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        self.cache_ttl = cache_ttl
        
        # Initialize schema
        self._initialize_schema()
        
        logger.info("Unified state manager initialized")
        
    def _initialize_schema(self):
        """Initialize database schema if not exists"""
        try:
            with self.ojs_engine.connect() as conn:
                # Check if tables exist
                result = conn.execute(text(
                    "SHOW TABLES LIKE 'skz_agent_states'"
                ))
                if not result.fetchone():
                    logger.info("Creating SKZ agent tables...")
                    self._create_tables(conn)
        except Exception as e:
            logger.error(f"Error initializing schema: {e}")
            
    def _create_tables(self, conn):
        """Create SKZ agent tables"""
        # Agent states table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS skz_agent_states (
                agent_id VARCHAR(50) PRIMARY KEY,
                state_data JSON,
                last_updated DATETIME,
                submission_id INT,
                INDEX idx_submission_id (submission_id),
                INDEX idx_last_updated (last_updated)
            )
        """))
        
        # Agent communications table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS skz_agent_communications (
                id INT AUTO_INCREMENT PRIMARY KEY,
                agent_from VARCHAR(50),
                agent_to VARCHAR(50),
                agent_name VARCHAR(50),
                action VARCHAR(50),
                message_type VARCHAR(50),
                payload JSON,
                timestamp DATETIME,
                success BOOLEAN DEFAULT TRUE,
                response_time FLOAT DEFAULT 0,
                request_size INT DEFAULT 0,
                response_size INT DEFAULT 0,
                INDEX idx_agent_name (agent_name),
                INDEX idx_timestamp (timestamp),
                INDEX idx_success (success)
            )
        """))
        
        # Workflow automation table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS skz_workflow_automation (
                id INT AUTO_INCREMENT PRIMARY KEY,
                submission_id INT NOT NULL,
                workflow_type VARCHAR(50) NOT NULL,
                agent_name VARCHAR(50) NOT NULL,
                automation_data JSON,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                status VARCHAR(20) DEFAULT 'pending',
                INDEX idx_submission (submission_id),
                INDEX idx_workflow (workflow_type),
                INDEX idx_agent (agent_name),
                INDEX idx_status (status),
                INDEX idx_created (created_at)
            )
        """))
        
        # Agent learning table (ontogenetic loom)
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS skz_agent_learning (
                id INT AUTO_INCREMENT PRIMARY KEY,
                agent_id VARCHAR(50) NOT NULL,
                learning_type VARCHAR(50) NOT NULL,
                context_data JSON,
                action_taken VARCHAR(100),
                outcome VARCHAR(50),
                reward FLOAT,
                confidence FLOAT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_agent (agent_id),
                INDEX idx_learning_type (learning_type),
                INDEX idx_timestamp (timestamp)
            )
        """))
        
        conn.commit()
        logger.info("SKZ agent tables created")
        
    @contextmanager
    def get_session(self) -> Session:
        """Context manager for database sessions"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            session.close()
            
    # =========================================================================
    # Agent State Management
    # =========================================================================
    
    def sync_agent_state(self, agent_id: str, state_data: dict, submission_id: Optional[int] = None):
        """
        Sync agent state to both OJS database and Redis cache.
        
        Implements write-through caching strategy.
        """
        try:
            # Update OJS database
            with self.get_session() as session:
                session.execute(
                    text("""
                        INSERT INTO skz_agent_states (agent_id, state_data, last_updated, submission_id)
                        VALUES (:agent_id, :state_data, NOW(), :submission_id)
                        ON DUPLICATE KEY UPDATE 
                            state_data = :state_data,
                            last_updated = NOW(),
                            submission_id = :submission_id
                    """),
                    {
                        "agent_id": agent_id,
                        "state_data": json.dumps(state_data),
                        "submission_id": submission_id
                    }
                )
            
            # Update Redis cache
            cache_key = f"agent_state:{agent_id}"
            self.redis_client.setex(
                cache_key,
                self.cache_ttl,
                json.dumps(state_data)
            )
            
            logger.debug(f"Synced state for agent: {agent_id}")
            
        except Exception as e:
            logger.error(f"Error syncing agent state: {e}")
            raise
            
    def get_agent_state(self, agent_id: str) -> Optional[dict]:
        """
        Retrieve agent state with cache-first strategy.
        
        Implements read-through caching.
        """
        cache_key = f"agent_state:{agent_id}"
        
        try:
            # Try cache first (hot storage)
            cached = self.redis_client.get(cache_key)
            if cached:
                logger.debug(f"Cache hit for agent: {agent_id}")
                return json.loads(cached)
            
            # Fallback to database (warm storage)
            with self.get_session() as session:
                result = session.execute(
                    text("SELECT state_data FROM skz_agent_states WHERE agent_id = :agent_id"),
                    {"agent_id": agent_id}
                ).fetchone()
                
                if result:
                    state_data = json.loads(result[0])
                    
                    # Populate cache
                    self.redis_client.setex(
                        cache_key,
                        self.cache_ttl,
                        json.dumps(state_data)
                    )
                    
                    logger.debug(f"Database hit for agent: {agent_id}")
                    return state_data
                    
            logger.debug(f"No state found for agent: {agent_id}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting agent state: {e}")
            return None
            
    def delete_agent_state(self, agent_id: str):
        """Delete agent state from both cache and database"""
        try:
            # Delete from cache
            self.redis_client.delete(f"agent_state:{agent_id}")
            
            # Delete from database
            with self.get_session() as session:
                session.execute(
                    text("DELETE FROM skz_agent_states WHERE agent_id = :agent_id"),
                    {"agent_id": agent_id}
                )
                
            logger.info(f"Deleted state for agent: {agent_id}")
            
        except Exception as e:
            logger.error(f"Error deleting agent state: {e}")
            
    # =========================================================================
    # Communication Logging
    # =========================================================================
    
    def log_communication(self, 
                         agent_from: str,
                         agent_to: str,
                         message_type: str,
                         payload: dict,
                         success: bool = True,
                         response_time: float = 0.0):
        """Log agent communication for audit and analysis"""
        try:
            with self.get_session() as session:
                session.execute(
                    text("""
                        INSERT INTO skz_agent_communications 
                        (agent_from, agent_to, message_type, payload, timestamp, success, response_time)
                        VALUES (:agent_from, :agent_to, :message_type, :payload, NOW(), :success, :response_time)
                    """),
                    {
                        "agent_from": agent_from,
                        "agent_to": agent_to,
                        "message_type": message_type,
                        "payload": json.dumps(payload),
                        "success": success,
                        "response_time": response_time
                    }
                )
                
        except Exception as e:
            logger.error(f"Error logging communication: {e}")
            
    def get_communication_history(self, 
                                  agent_id: Optional[str] = None,
                                  start_time: Optional[datetime] = None,
                                  end_time: Optional[datetime] = None,
                                  limit: int = 100) -> List[dict]:
        """Retrieve communication history"""
        try:
            query = "SELECT * FROM skz_agent_communications WHERE 1=1"
            params = {}
            
            if agent_id:
                query += " AND (agent_from = :agent_id OR agent_to = :agent_id)"
                params["agent_id"] = agent_id
                
            if start_time:
                query += " AND timestamp >= :start_time"
                params["start_time"] = start_time
                
            if end_time:
                query += " AND timestamp <= :end_time"
                params["end_time"] = end_time
                
            query += " ORDER BY timestamp DESC LIMIT :limit"
            params["limit"] = limit
            
            with self.get_session() as session:
                result = session.execute(text(query), params)
                return [dict(row._mapping) for row in result]
                
        except Exception as e:
            logger.error(f"Error getting communication history: {e}")
            return []
            
    # =========================================================================
    # Workflow Automation
    # =========================================================================
    
    def create_workflow_automation(self,
                                   submission_id: int,
                                   workflow_type: str,
                                   agent_name: str,
                                   automation_data: dict) -> int:
        """Create workflow automation record"""
        try:
            with self.get_session() as session:
                result = session.execute(
                    text("""
                        INSERT INTO skz_workflow_automation 
                        (submission_id, workflow_type, agent_name, automation_data, status)
                        VALUES (:submission_id, :workflow_type, :agent_name, :automation_data, 'pending')
                    """),
                    {
                        "submission_id": submission_id,
                        "workflow_type": workflow_type,
                        "agent_name": agent_name,
                        "automation_data": json.dumps(automation_data)
                    }
                )
                return result.lastrowid
                
        except Exception as e:
            logger.error(f"Error creating workflow automation: {e}")
            return 0
            
    def update_workflow_status(self, automation_id: int, status: str, automation_data: Optional[dict] = None):
        """Update workflow automation status"""
        try:
            with self.get_session() as session:
                if automation_data:
                    session.execute(
                        text("""
                            UPDATE skz_workflow_automation 
                            SET status = :status, automation_data = :automation_data, updated_at = NOW()
                            WHERE id = :id
                        """),
                        {
                            "id": automation_id,
                            "status": status,
                            "automation_data": json.dumps(automation_data)
                        }
                    )
                else:
                    session.execute(
                        text("""
                            UPDATE skz_workflow_automation 
                            SET status = :status, updated_at = NOW()
                            WHERE id = :id
                        """),
                        {"id": automation_id, "status": status}
                    )
                    
        except Exception as e:
            logger.error(f"Error updating workflow status: {e}")
            
    # =========================================================================
    # Learning and Adaptation (Ontogenetic Looms)
    # =========================================================================
    
    def record_learning_event(self,
                             agent_id: str,
                             learning_type: str,
                             context_data: dict,
                             action_taken: str,
                             outcome: str,
                             reward: float,
                             confidence: float):
        """
        Record learning event for agent.
        
        Implements ontogenetic loom by capturing learning across time.
        """
        try:
            with self.get_session() as session:
                session.execute(
                    text("""
                        INSERT INTO skz_agent_learning 
                        (agent_id, learning_type, context_data, action_taken, outcome, reward, confidence)
                        VALUES (:agent_id, :learning_type, :context_data, :action_taken, :outcome, :reward, :confidence)
                    """),
                    {
                        "agent_id": agent_id,
                        "learning_type": learning_type,
                        "context_data": json.dumps(context_data),
                        "action_taken": action_taken,
                        "outcome": outcome,
                        "reward": reward,
                        "confidence": confidence
                    }
                )
                
            logger.info(f"Recorded learning event for {agent_id}: {learning_type}")
            
        except Exception as e:
            logger.error(f"Error recording learning event: {e}")
            
    def get_learning_history(self,
                            agent_id: str,
                            learning_type: Optional[str] = None,
                            days: int = 30) -> List[dict]:
        """Retrieve learning history for agent"""
        try:
            query = """
                SELECT * FROM skz_agent_learning 
                WHERE agent_id = :agent_id 
                AND timestamp >= DATE_SUB(NOW(), INTERVAL :days DAY)
            """
            params = {"agent_id": agent_id, "days": days}
            
            if learning_type:
                query += " AND learning_type = :learning_type"
                params["learning_type"] = learning_type
                
            query += " ORDER BY timestamp DESC"
            
            with self.get_session() as session:
                result = session.execute(text(query), params)
                return [dict(row._mapping) for row in result]
                
        except Exception as e:
            logger.error(f"Error getting learning history: {e}")
            return []
            
    def get_agent_performance_metrics(self, agent_id: str, days: int = 30) -> dict:
        """Calculate agent performance metrics from learning history"""
        try:
            learning_history = self.get_learning_history(agent_id, days=days)
            
            if not learning_history:
                return {
                    "total_events": 0,
                    "average_reward": 0.0,
                    "average_confidence": 0.0,
                    "success_rate": 0.0
                }
                
            total_events = len(learning_history)
            total_reward = sum(event['reward'] for event in learning_history)
            total_confidence = sum(event['confidence'] for event in learning_history)
            successful_events = sum(1 for event in learning_history if event['outcome'] == 'success')
            
            return {
                "total_events": total_events,
                "average_reward": total_reward / total_events,
                "average_confidence": total_confidence / total_events,
                "success_rate": successful_events / total_events
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}
            
    # =========================================================================
    # Cache Management
    # =========================================================================
    
    def invalidate_cache(self, pattern: str):
        """Invalidate cache entries matching pattern"""
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                self.redis_client.delete(*keys)
                logger.info(f"Invalidated {len(keys)} cache entries")
        except Exception as e:
            logger.error(f"Error invalidating cache: {e}")
            
    def get_cache_stats(self) -> dict:
        """Get cache statistics"""
        try:
            info = self.redis_client.info()
            return {
                "used_memory": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "total_commands_processed": info.get("total_commands_processed"),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0)
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {}
            
    def health_check(self) -> dict:
        """Check health of state management system"""
        health = {
            "database": False,
            "cache": False,
            "overall": False
        }
        
        try:
            # Check database
            with self.get_session() as session:
                session.execute(text("SELECT 1"))
            health["database"] = True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            
        try:
            # Check cache
            self.redis_client.ping()
            health["cache"] = True
        except Exception as e:
            logger.error(f"Cache health check failed: {e}")
            
        health["overall"] = health["database"] and health["cache"]
        return health


# Example usage
if __name__ == "__main__":
    # Initialize state manager
    state_manager = UnifiedStateManager(
        ojs_db_url=os.getenv("OJS_DB_URL", "mysql+pymysql://ojs:ojs@localhost:3306/ojs"),
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0")
    )
    
    # Sync agent state
    state_manager.sync_agent_state(
        "research-discovery",
        {"status": "active", "current_task": "analyzing_manuscript_123"},
        submission_id=123
    )
    
    # Retrieve agent state
    state = state_manager.get_agent_state("research-discovery")
    print(f"Agent state: {state}")
    
    # Record learning event
    state_manager.record_learning_event(
        agent_id="editorial-orchestration",
        learning_type="decision_making",
        context_data={"manuscript_quality": 0.85, "reviewer_consensus": "accept"},
        action_taken="accept_manuscript",
        outcome="success",
        reward=1.0,
        confidence=0.9
    )
    
    # Get performance metrics
    metrics = state_manager.get_agent_performance_metrics("editorial-orchestration")
    print(f"Performance metrics: {metrics}")
    
    # Health check
    health = state_manager.health_check()
    print(f"Health: {health}")
