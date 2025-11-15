"""
OJS-SKZ Workflow Integration Layer
Version 1.0 - November 2025

This module provides the integration layer between OJS core workflows
and the SKZ autonomous agents framework. It handles workflow events,
agent coordination, and bidirectional communication.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
import mysql.connector
from mysql.connector import pooling
import redis
from dataclasses import dataclass, asdict
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WorkflowStage(Enum):
    """OJS workflow stages"""
    SUBMISSION = "submission"
    REVIEW = "review"
    COPYEDITING = "copyediting"
    PRODUCTION = "production"
    PUBLISHED = "published"


class AgentType(Enum):
    """Autonomous agent types"""
    RESEARCH_DISCOVERY = "research_discovery"
    SUBMISSION_ASSISTANT = "submission_assistant"
    EDITORIAL_ORCHESTRATION = "editorial_orchestration"
    REVIEW_COORDINATION = "review_coordination"
    CONTENT_QUALITY = "content_quality"
    PUBLISHING_PRODUCTION = "publishing_production"
    ANALYTICS_MONITORING = "analytics_monitoring"


class TaskStatus(Enum):
    """Agent task status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class WorkflowEvent:
    """Workflow event data structure"""
    event_type: str
    submission_id: int
    stage: WorkflowStage
    metadata: Dict[str, Any]
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class AgentTask:
    """Agent task data structure"""
    agent_id: str
    agent_type: AgentType
    task_type: str
    submission_id: int
    priority: int
    task_data: Dict[str, Any]
    status: TaskStatus = TaskStatus.PENDING
    task_id: Optional[int] = None
    result_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class DatabaseManager:
    """Manages database connections and operations"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize database manager with connection pool"""
        self.config = config
        self.pool = pooling.MySQLConnectionPool(
            pool_name="ojs_skz_pool",
            pool_size=10,
            **config
        )
        logger.info("Database connection pool initialized")
    
    def get_connection(self):
        """Get a connection from the pool"""
        return self.pool.get_connection()
    
    def execute_query(self, query: str, params: tuple = None, fetch: bool = True):
        """Execute a database query"""
        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute(query, params or ())
            
            if fetch:
                result = cursor.fetchall()
                return result
            else:
                conn.commit()
                return cursor.lastrowid
        except Exception as e:
            logger.error(f"Database query error: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
    
    def call_procedure(self, proc_name: str, params: tuple):
        """Call a stored procedure"""
        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.callproc(proc_name, params)
            conn.commit()
            
            # Get OUT parameters if any
            results = []
            for result in cursor.stored_results():
                results.extend(result.fetchall())
            return results
        except Exception as e:
            logger.error(f"Stored procedure error: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()


class CacheManager:
    """Manages Redis cache for performance optimization"""
    
    def __init__(self, redis_config: Dict[str, Any]):
        """Initialize Redis connection"""
        self.redis_client = redis.Redis(**redis_config, decode_responses=True)
        logger.info("Redis cache manager initialized")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            value = self.redis_client.get(key)
            return json.loads(value) if value else None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: int = 3600):
        """Set value in cache with TTL"""
        try:
            self.redis_client.setex(key, ttl, json.dumps(value))
        except Exception as e:
            logger.error(f"Cache set error: {e}")
    
    def delete(self, key: str):
        """Delete key from cache"""
        try:
            self.redis_client.delete(key)
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
    
    def publish(self, channel: str, message: Dict[str, Any]):
        """Publish message to Redis pub/sub channel"""
        try:
            self.redis_client.publish(channel, json.dumps(message))
        except Exception as e:
            logger.error(f"Cache publish error: {e}")


class WorkflowIntegration:
    """Main workflow integration coordinator"""
    
    def __init__(self, db_config: Dict[str, Any], redis_config: Dict[str, Any]):
        """Initialize workflow integration"""
        self.db = DatabaseManager(db_config)
        self.cache = CacheManager(redis_config)
        self.event_handlers: Dict[str, List[Callable]] = {}
        self.agent_registry: Dict[str, Dict[str, Any]] = {}
        logger.info("Workflow integration initialized")
    
    def register_agent(self, agent_id: str, agent_type: AgentType, 
                      handler: Callable, config: Dict[str, Any] = None):
        """Register an agent with the workflow system"""
        self.agent_registry[agent_id] = {
            'agent_type': agent_type,
            'handler': handler,
            'config': config or {},
            'registered_at': datetime.now()
        }
        logger.info(f"Agent registered: {agent_id} ({agent_type.value})")
    
    def register_event_handler(self, event_type: str, handler: Callable):
        """Register a handler for workflow events"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
        logger.info(f"Event handler registered for: {event_type}")
    
    async def process_workflow_event(self, event: WorkflowEvent):
        """Process a workflow event and trigger appropriate agents"""
        logger.info(f"Processing workflow event: {event.event_type} for submission {event.submission_id}")
        
        try:
            # Call registered event handlers
            if event.event_type in self.event_handlers:
                for handler in self.event_handlers[event.event_type]:
                    try:
                        await handler(event)
                    except Exception as e:
                        logger.error(f"Event handler error: {e}")
            
            # Determine which agents to activate based on event
            agents_to_activate = self._get_agents_for_event(event)
            
            # Queue tasks for each agent
            tasks = []
            for agent_config in agents_to_activate:
                task = self._create_agent_task(event, agent_config)
                task_id = await self.queue_agent_task(task)
                tasks.append(task_id)
            
            logger.info(f"Queued {len(tasks)} agent tasks for event {event.event_type}")
            
            # Publish event to Redis for real-time updates
            self.cache.publish('workflow_events', {
                'event_type': event.event_type,
                'submission_id': event.submission_id,
                'stage': event.stage.value,
                'tasks_queued': len(tasks),
                'timestamp': event.timestamp.isoformat()
            })
            
            return tasks
            
        except Exception as e:
            logger.error(f"Error processing workflow event: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def _get_agents_for_event(self, event: WorkflowEvent) -> List[Dict[str, Any]]:
        """Determine which agents should be activated for an event"""
        agents = []
        
        if event.event_type == 'submission_created':
            agents.extend([
                {'agent_id': 'research_discovery_001', 'agent_type': AgentType.RESEARCH_DISCOVERY,
                 'task_type': 'analyze_novelty', 'priority': 8},
                {'agent_id': 'submission_assistant_001', 'agent_type': AgentType.SUBMISSION_ASSISTANT,
                 'task_type': 'quality_assessment', 'priority': 8}
            ])
        
        elif event.event_type == 'review_round_created':
            agents.append(
                {'agent_id': 'review_coordination_001', 'agent_type': AgentType.REVIEW_COORDINATION,
                 'task_type': 'match_reviewers', 'priority': 9}
            )
        
        elif event.event_type == 'review_completed':
            agents.extend([
                {'agent_id': 'content_quality_001', 'agent_type': AgentType.CONTENT_QUALITY,
                 'task_type': 'analyze_review', 'priority': 7},
                {'agent_id': 'editorial_orchestration_001', 'agent_type': AgentType.EDITORIAL_ORCHESTRATION,
                 'task_type': 'synthesize_reviews', 'priority': 7}
            ])
        
        elif event.event_type == 'copyediting_started':
            agents.append(
                {'agent_id': 'content_quality_001', 'agent_type': AgentType.CONTENT_QUALITY,
                 'task_type': 'copyediting_support', 'priority': 6}
            )
        
        elif event.event_type == 'production_started':
            agents.append(
                {'agent_id': 'publishing_production_001', 'agent_type': AgentType.PUBLISHING_PRODUCTION,
                 'task_type': 'prepare_publication', 'priority': 8}
            )
        
        elif event.event_type == 'submission_published':
            agents.append(
                {'agent_id': 'analytics_monitoring_001', 'agent_type': AgentType.ANALYTICS_MONITORING,
                 'task_type': 'track_publication', 'priority': 5}
            )
        
        return agents
    
    def _create_agent_task(self, event: WorkflowEvent, agent_config: Dict[str, Any]) -> AgentTask:
        """Create an agent task from workflow event"""
        task_data = {
            'submission_id': event.submission_id,
            'event_type': event.event_type,
            'stage': event.stage.value,
            **event.metadata
        }
        
        return AgentTask(
            agent_id=agent_config['agent_id'],
            agent_type=agent_config['agent_type'],
            task_type=agent_config['task_type'],
            submission_id=event.submission_id,
            priority=agent_config['priority'],
            task_data=task_data
        )
    
    async def queue_agent_task(self, task: AgentTask) -> int:
        """Queue an agent task in the database"""
        try:
            query = """
                INSERT INTO agent_task_queue 
                (agent_id, agent_type, task_type, submission_id, priority, task_data, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            params = (
                task.agent_id,
                task.agent_type.value,
                task.task_type,
                task.submission_id,
                task.priority,
                json.dumps(task.task_data),
                task.status.value
            )
            
            task_id = self.db.execute_query(query, params, fetch=False)
            task.task_id = task_id
            
            # Cache task for quick access
            self.cache.set(f"agent_task:{task_id}", asdict(task), ttl=3600)
            
            # Publish task queued event
            self.cache.publish('agent_tasks', {
                'action': 'queued',
                'task_id': task_id,
                'agent_id': task.agent_id,
                'task_type': task.task_type,
                'submission_id': task.submission_id
            })
            
            logger.info(f"Task queued: {task_id} for agent {task.agent_id}")
            return task_id
            
        except Exception as e:
            logger.error(f"Error queuing agent task: {e}")
            raise
    
    async def get_pending_tasks(self, agent_id: Optional[str] = None, 
                               limit: int = 10) -> List[AgentTask]:
        """Get pending tasks from the queue"""
        try:
            if agent_id:
                query = """
                    SELECT * FROM agent_task_queue
                    WHERE status = 'pending' AND agent_id = %s
                    ORDER BY priority DESC, created_at ASC
                    LIMIT %s
                """
                params = (agent_id, limit)
            else:
                query = """
                    SELECT * FROM agent_task_queue
                    WHERE status = 'pending'
                    ORDER BY priority DESC, created_at ASC
                    LIMIT %s
                """
                params = (limit,)
            
            results = self.db.execute_query(query, params)
            
            tasks = []
            for row in results:
                task = AgentTask(
                    agent_id=row['agent_id'],
                    agent_type=AgentType(row['agent_type']),
                    task_type=row['task_type'],
                    submission_id=row['submission_id'],
                    priority=row['priority'],
                    task_data=json.loads(row['task_data']) if row['task_data'] else {},
                    status=TaskStatus(row['status']),
                    task_id=row['task_id'],
                    retry_count=row['retry_count'],
                    max_retries=row['max_retries'],
                    created_at=row['created_at']
                )
                tasks.append(task)
            
            return tasks
            
        except Exception as e:
            logger.error(f"Error getting pending tasks: {e}")
            return []
    
    async def update_task_status(self, task_id: int, status: TaskStatus,
                                result_data: Optional[Dict[str, Any]] = None,
                                error_message: Optional[str] = None):
        """Update task status in database"""
        try:
            query = """
                UPDATE agent_task_queue
                SET status = %s,
                    result_data = %s,
                    error_message = %s,
                    completed_at = CASE WHEN %s IN ('completed', 'failed', 'cancelled') 
                                   THEN NOW() ELSE completed_at END
                WHERE task_id = %s
            """
            params = (
                status.value,
                json.dumps(result_data) if result_data else None,
                error_message,
                status.value,
                task_id
            )
            
            self.db.execute_query(query, params, fetch=False)
            
            # Update cache
            self.cache.delete(f"agent_task:{task_id}")
            
            # Publish status update
            self.cache.publish('agent_tasks', {
                'action': 'status_updated',
                'task_id': task_id,
                'status': status.value,
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info(f"Task {task_id} status updated to {status.value}")
            
        except Exception as e:
            logger.error(f"Error updating task status: {e}")
            raise
    
    async def log_agent_decision(self, agent_id: str, agent_type: AgentType,
                                submission_id: int, decision_type: str,
                                decision_data: Dict[str, Any], 
                                confidence_score: float,
                                reasoning: str):
        """Log an agent decision to the database"""
        try:
            query = """
                INSERT INTO agent_decisions
                (agent_id, agent_type, submission_id, decision_type, 
                 decision_data, confidence_score, reasoning)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            params = (
                agent_id,
                agent_type.value,
                submission_id,
                decision_type,
                json.dumps(decision_data),
                confidence_score,
                reasoning
            )
            
            decision_id = self.db.execute_query(query, params, fetch=False)
            
            # Publish decision event
            self.cache.publish('agent_decisions', {
                'decision_id': decision_id,
                'agent_id': agent_id,
                'submission_id': submission_id,
                'decision_type': decision_type,
                'confidence_score': confidence_score,
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info(f"Agent decision logged: {decision_id}")
            return decision_id
            
        except Exception as e:
            logger.error(f"Error logging agent decision: {e}")
            raise
    
    async def update_agent_heartbeat(self, agent_id: str, status: str, 
                                    current_task: Optional[str] = None):
        """Update agent heartbeat to indicate it's alive"""
        try:
            self.db.call_procedure('sp_update_agent_heartbeat', 
                                  (agent_id, status, current_task))
            
            # Cache agent status
            self.cache.set(f"agent_status:{agent_id}", {
                'status': status,
                'current_task': current_task,
                'last_heartbeat': datetime.now().isoformat()
            }, ttl=300)
            
        except Exception as e:
            logger.error(f"Error updating agent heartbeat: {e}")
    
    async def get_submission_analysis(self, submission_id: int) -> List[Dict[str, Any]]:
        """Get all analysis results for a submission"""
        try:
            query = """
                SELECT sa.*, ast.agent_name
                FROM submission_analysis sa
                LEFT JOIN agent_state ast ON sa.agent_id = ast.agent_id
                WHERE sa.submission_id = %s
                ORDER BY sa.created_at DESC
            """
            results = self.db.execute_query(query, (submission_id,))
            
            # Parse JSON fields
            for result in results:
                if result.get('analysis_results'):
                    result['analysis_results'] = json.loads(result['analysis_results'])
                if result.get('flags'):
                    result['flags'] = json.loads(result['flags'])
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting submission analysis: {e}")
            return []
    
    async def get_workflow_metrics(self, days: int = 30) -> Dict[str, Any]:
        """Get workflow performance metrics"""
        try:
            # Get overall metrics
            query = """
                SELECT 
                    COUNT(DISTINCT submission_id) as total_submissions,
                    AVG(TIMESTAMPDIFF(SECOND, created_at, completed_at)) as avg_processing_time,
                    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) / COUNT(*) as success_rate,
                    SUM(CASE WHEN transition_type = 'automatic' THEN 1 ELSE 0 END) / COUNT(*) as automation_rate
                FROM workflow_transitions
                WHERE created_at >= DATE_SUB(NOW(), INTERVAL %s DAY)
            """
            overall = self.db.execute_query(query, (days,))[0]
            
            # Get stage-specific metrics
            query = """
                SELECT 
                    workflow_stage,
                    AVG(avg_processing_time) as avg_time,
                    AVG(automation_rate) as automation,
                    AVG(success_rate) as success
                FROM workflow_analytics
                WHERE date_period >= DATE_SUB(CURDATE(), INTERVAL %s DAY)
                GROUP BY workflow_stage
            """
            by_stage = self.db.execute_query(query, (days,))
            
            # Get agent performance
            query = """
                SELECT 
                    agent_type,
                    COUNT(*) as total_decisions,
                    AVG(confidence_score) as avg_confidence,
                    SUM(CASE WHEN human_override = TRUE THEN 1 ELSE 0 END) / COUNT(*) as override_rate
                FROM agent_decisions
                WHERE created_at >= DATE_SUB(NOW(), INTERVAL %s DAY)
                GROUP BY agent_type
            """
            agent_performance = self.db.execute_query(query, (days,))
            
            return {
                'overall': overall,
                'by_stage': by_stage,
                'agent_performance': agent_performance,
                'period_days': days,
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting workflow metrics: {e}")
            return {}


# Example usage and integration patterns
async def example_usage():
    """Example of how to use the workflow integration"""
    
    # Configuration
    db_config = {
        'host': 'localhost',
        'port': 3306,
        'database': 'ojs_db',
        'user': 'ojs_user',
        'password': 'ojs_password'
    }
    
    redis_config = {
        'host': 'localhost',
        'port': 6379,
        'db': 0
    }
    
    # Initialize integration
    integration = WorkflowIntegration(db_config, redis_config)
    
    # Register event handlers
    async def handle_submission_created(event: WorkflowEvent):
        logger.info(f"Custom handler: New submission {event.submission_id}")
    
    integration.register_event_handler('submission_created', handle_submission_created)
    
    # Process a workflow event
    event = WorkflowEvent(
        event_type='submission_created',
        submission_id=12345,
        stage=WorkflowStage.SUBMISSION,
        metadata={
            'title': 'Novel Cosmetic Ingredient Study',
            'author': 'Dr. Smith',
            'keywords': ['cosmetic', 'ingredient', 'safety']
        }
    )
    
    tasks = await integration.process_workflow_event(event)
    logger.info(f"Processed event, created {len(tasks)} tasks")
    
    # Get pending tasks for an agent
    pending = await integration.get_pending_tasks('research_discovery_001', limit=5)
    logger.info(f"Found {len(pending)} pending tasks")
    
    # Update task status
    if pending:
        task = pending[0]
        await integration.update_task_status(
            task.task_id,
            TaskStatus.COMPLETED,
            result_data={'novelty_score': 0.85, 'similar_papers': 3}
        )
    
    # Log agent decision
    await integration.log_agent_decision(
        agent_id='research_discovery_001',
        agent_type=AgentType.RESEARCH_DISCOVERY,
        submission_id=12345,
        decision_type='novelty_assessment',
        decision_data={'score': 0.85, 'recommendation': 'proceed'},
        confidence_score=0.92,
        reasoning='High novelty score with limited similar research'
    )
    
    # Get metrics
    metrics = await integration.get_workflow_metrics(days=30)
    logger.info(f"Workflow metrics: {json.dumps(metrics, indent=2)}")


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())
