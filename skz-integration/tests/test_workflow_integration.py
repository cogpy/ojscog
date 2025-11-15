"""
Test Suite for Workflow Integration
Version 1.0 - November 2025

Comprehensive tests for OJS-SKZ workflow integration layer
"""

import unittest
import asyncio
import json
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from workflow_integration import (
    WorkflowIntegration,
    WorkflowEvent,
    WorkflowStage,
    AgentTask,
    AgentType,
    TaskStatus,
    DatabaseManager,
    CacheManager
)


class TestWorkflowEvent(unittest.TestCase):
    """Test WorkflowEvent data structure"""
    
    def test_workflow_event_creation(self):
        """Test creating a workflow event"""
        event = WorkflowEvent(
            event_type='submission_created',
            submission_id=12345,
            stage=WorkflowStage.SUBMISSION,
            metadata={'title': 'Test Paper'}
        )
        
        self.assertEqual(event.event_type, 'submission_created')
        self.assertEqual(event.submission_id, 12345)
        self.assertEqual(event.stage, WorkflowStage.SUBMISSION)
        self.assertIsNotNone(event.timestamp)
    
    def test_workflow_event_with_timestamp(self):
        """Test creating event with custom timestamp"""
        custom_time = datetime(2025, 11, 15, 12, 0, 0)
        event = WorkflowEvent(
            event_type='review_completed',
            submission_id=67890,
            stage=WorkflowStage.REVIEW,
            metadata={},
            timestamp=custom_time
        )
        
        self.assertEqual(event.timestamp, custom_time)


class TestAgentTask(unittest.TestCase):
    """Test AgentTask data structure"""
    
    def test_agent_task_creation(self):
        """Test creating an agent task"""
        task = AgentTask(
            agent_id='research_discovery_001',
            agent_type=AgentType.RESEARCH_DISCOVERY,
            task_type='analyze_novelty',
            submission_id=12345,
            priority=8,
            task_data={'title': 'Test Paper'}
        )
        
        self.assertEqual(task.agent_id, 'research_discovery_001')
        self.assertEqual(task.agent_type, AgentType.RESEARCH_DISCOVERY)
        self.assertEqual(task.status, TaskStatus.PENDING)
        self.assertEqual(task.retry_count, 0)
        self.assertIsNotNone(task.created_at)


class TestDatabaseManager(unittest.TestCase):
    """Test DatabaseManager functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'host': 'localhost',
            'port': 3306,
            'database': 'test_db',
            'user': 'test_user',
            'password': 'test_pass'
        }
    
    @patch('workflow_integration.pooling.MySQLConnectionPool')
    def test_database_manager_initialization(self, mock_pool):
        """Test database manager initialization"""
        db = DatabaseManager(self.config)
        
        mock_pool.assert_called_once()
        self.assertIsNotNone(db.pool)
    
    @patch('workflow_integration.pooling.MySQLConnectionPool')
    def test_get_connection(self, mock_pool):
        """Test getting connection from pool"""
        mock_pool_instance = MagicMock()
        mock_pool.return_value = mock_pool_instance
        
        db = DatabaseManager(self.config)
        connection = db.get_connection()
        
        mock_pool_instance.get_connection.assert_called_once()


class TestCacheManager(unittest.TestCase):
    """Test CacheManager functionality"""
    
    @patch('workflow_integration.redis.Redis')
    def test_cache_manager_initialization(self, mock_redis):
        """Test cache manager initialization"""
        redis_config = {'host': 'localhost', 'port': 6379, 'db': 0}
        cache = CacheManager(redis_config)
        
        mock_redis.assert_called_once()
    
    @patch('workflow_integration.redis.Redis')
    def test_cache_set_get(self, mock_redis):
        """Test cache set and get operations"""
        mock_redis_instance = MagicMock()
        mock_redis.return_value = mock_redis_instance
        mock_redis_instance.get.return_value = json.dumps({'key': 'value'})
        
        redis_config = {'host': 'localhost', 'port': 6379, 'db': 0}
        cache = CacheManager(redis_config)
        
        # Test set
        cache.set('test_key', {'key': 'value'})
        mock_redis_instance.setex.assert_called_once()
        
        # Test get
        result = cache.get('test_key')
        self.assertEqual(result, {'key': 'value'})
    
    @patch('workflow_integration.redis.Redis')
    def test_cache_publish(self, mock_redis):
        """Test cache publish operation"""
        mock_redis_instance = MagicMock()
        mock_redis.return_value = mock_redis_instance
        
        redis_config = {'host': 'localhost', 'port': 6379, 'db': 0}
        cache = CacheManager(redis_config)
        
        message = {'event': 'test', 'data': 'value'}
        cache.publish('test_channel', message)
        
        mock_redis_instance.publish.assert_called_once()


class TestWorkflowIntegration(unittest.TestCase):
    """Test WorkflowIntegration main coordinator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.db_config = {
            'host': 'localhost',
            'port': 3306,
            'database': 'test_db',
            'user': 'test_user',
            'password': 'test_pass'
        }
        self.redis_config = {
            'host': 'localhost',
            'port': 6379,
            'db': 0
        }
    
    @patch('workflow_integration.DatabaseManager')
    @patch('workflow_integration.CacheManager')
    def test_workflow_integration_initialization(self, mock_cache, mock_db):
        """Test workflow integration initialization"""
        integration = WorkflowIntegration(self.db_config, self.redis_config)
        
        self.assertIsNotNone(integration.db)
        self.assertIsNotNone(integration.cache)
        self.assertEqual(len(integration.event_handlers), 0)
        self.assertEqual(len(integration.agent_registry), 0)
    
    @patch('workflow_integration.DatabaseManager')
    @patch('workflow_integration.CacheManager')
    def test_register_agent(self, mock_cache, mock_db):
        """Test agent registration"""
        integration = WorkflowIntegration(self.db_config, self.redis_config)
        
        handler = Mock()
        integration.register_agent(
            'test_agent_001',
            AgentType.RESEARCH_DISCOVERY,
            handler,
            {'config_key': 'config_value'}
        )
        
        self.assertIn('test_agent_001', integration.agent_registry)
        self.assertEqual(
            integration.agent_registry['test_agent_001']['agent_type'],
            AgentType.RESEARCH_DISCOVERY
        )
    
    @patch('workflow_integration.DatabaseManager')
    @patch('workflow_integration.CacheManager')
    def test_register_event_handler(self, mock_cache, mock_db):
        """Test event handler registration"""
        integration = WorkflowIntegration(self.db_config, self.redis_config)
        
        handler = Mock()
        integration.register_event_handler('submission_created', handler)
        
        self.assertIn('submission_created', integration.event_handlers)
        self.assertEqual(len(integration.event_handlers['submission_created']), 1)
    
    @patch('workflow_integration.DatabaseManager')
    @patch('workflow_integration.CacheManager')
    def test_get_agents_for_event_submission(self, mock_cache, mock_db):
        """Test getting agents for submission event"""
        integration = WorkflowIntegration(self.db_config, self.redis_config)
        
        event = WorkflowEvent(
            event_type='submission_created',
            submission_id=12345,
            stage=WorkflowStage.SUBMISSION,
            metadata={}
        )
        
        agents = integration._get_agents_for_event(event)
        
        self.assertEqual(len(agents), 2)  # Research Discovery + Submission Assistant
        agent_types = [a['agent_type'] for a in agents]
        self.assertIn(AgentType.RESEARCH_DISCOVERY, agent_types)
        self.assertIn(AgentType.SUBMISSION_ASSISTANT, agent_types)
    
    @patch('workflow_integration.DatabaseManager')
    @patch('workflow_integration.CacheManager')
    def test_get_agents_for_event_review(self, mock_cache, mock_db):
        """Test getting agents for review event"""
        integration = WorkflowIntegration(self.db_config, self.redis_config)
        
        event = WorkflowEvent(
            event_type='review_round_created',
            submission_id=12345,
            stage=WorkflowStage.REVIEW,
            metadata={}
        )
        
        agents = integration._get_agents_for_event(event)
        
        self.assertEqual(len(agents), 1)  # Review Coordination
        self.assertEqual(agents[0]['agent_type'], AgentType.REVIEW_COORDINATION)
    
    @patch('workflow_integration.DatabaseManager')
    @patch('workflow_integration.CacheManager')
    def test_create_agent_task(self, mock_cache, mock_db):
        """Test creating agent task from event"""
        integration = WorkflowIntegration(self.db_config, self.redis_config)
        
        event = WorkflowEvent(
            event_type='submission_created',
            submission_id=12345,
            stage=WorkflowStage.SUBMISSION,
            metadata={'title': 'Test Paper'}
        )
        
        agent_config = {
            'agent_id': 'test_agent_001',
            'agent_type': AgentType.RESEARCH_DISCOVERY,
            'task_type': 'analyze_novelty',
            'priority': 8
        }
        
        task = integration._create_agent_task(event, agent_config)
        
        self.assertEqual(task.agent_id, 'test_agent_001')
        self.assertEqual(task.agent_type, AgentType.RESEARCH_DISCOVERY)
        self.assertEqual(task.task_type, 'analyze_novelty')
        self.assertEqual(task.submission_id, 12345)
        self.assertEqual(task.priority, 8)
        self.assertIn('title', task.task_data)


class TestWorkflowIntegrationAsync(unittest.IsolatedAsyncioTestCase):
    """Test async methods of WorkflowIntegration"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.db_config = {
            'host': 'localhost',
            'port': 3306,
            'database': 'test_db',
            'user': 'test_user',
            'password': 'test_pass'
        }
        self.redis_config = {
            'host': 'localhost',
            'port': 6379,
            'db': 0
        }
    
    @patch('workflow_integration.DatabaseManager')
    @patch('workflow_integration.CacheManager')
    async def test_queue_agent_task(self, mock_cache, mock_db):
        """Test queuing an agent task"""
        mock_db_instance = MagicMock()
        mock_db_instance.execute_query.return_value = 12345
        mock_db.return_value = mock_db_instance
        
        mock_cache_instance = MagicMock()
        mock_cache.return_value = mock_cache_instance
        
        integration = WorkflowIntegration(self.db_config, self.redis_config)
        
        task = AgentTask(
            agent_id='test_agent_001',
            agent_type=AgentType.RESEARCH_DISCOVERY,
            task_type='analyze_novelty',
            submission_id=12345,
            priority=8,
            task_data={'title': 'Test Paper'}
        )
        
        task_id = await integration.queue_agent_task(task)
        
        self.assertEqual(task_id, 12345)
        mock_db_instance.execute_query.assert_called_once()
    
    @patch('workflow_integration.DatabaseManager')
    @patch('workflow_integration.CacheManager')
    async def test_update_task_status(self, mock_cache, mock_db):
        """Test updating task status"""
        mock_db_instance = MagicMock()
        mock_db.return_value = mock_db_instance
        
        mock_cache_instance = MagicMock()
        mock_cache.return_value = mock_cache_instance
        
        integration = WorkflowIntegration(self.db_config, self.redis_config)
        
        await integration.update_task_status(
            task_id=12345,
            status=TaskStatus.COMPLETED,
            result_data={'novelty_score': 0.85}
        )
        
        mock_db_instance.execute_query.assert_called_once()
    
    @patch('workflow_integration.DatabaseManager')
    @patch('workflow_integration.CacheManager')
    async def test_log_agent_decision(self, mock_cache, mock_db):
        """Test logging agent decision"""
        mock_db_instance = MagicMock()
        mock_db_instance.execute_query.return_value = 67890
        mock_db.return_value = mock_db_instance
        
        mock_cache_instance = MagicMock()
        mock_cache.return_value = mock_cache_instance
        
        integration = WorkflowIntegration(self.db_config, self.redis_config)
        
        decision_id = await integration.log_agent_decision(
            agent_id='test_agent_001',
            agent_type=AgentType.RESEARCH_DISCOVERY,
            submission_id=12345,
            decision_type='novelty_assessment',
            decision_data={'score': 0.85},
            confidence_score=0.92,
            reasoning='High novelty detected'
        )
        
        self.assertEqual(decision_id, 67890)
        mock_db_instance.execute_query.assert_called_once()
    
    @patch('workflow_integration.DatabaseManager')
    @patch('workflow_integration.CacheManager')
    async def test_process_workflow_event(self, mock_cache, mock_db):
        """Test processing workflow event"""
        mock_db_instance = MagicMock()
        mock_db_instance.execute_query.return_value = 12345
        mock_db.return_value = mock_db_instance
        
        mock_cache_instance = MagicMock()
        mock_cache.return_value = mock_cache_instance
        
        integration = WorkflowIntegration(self.db_config, self.redis_config)
        
        # Register event handler
        handler = Mock()
        handler.return_value = asyncio.coroutine(lambda: None)()
        integration.register_event_handler('submission_created', handler)
        
        event = WorkflowEvent(
            event_type='submission_created',
            submission_id=12345,
            stage=WorkflowStage.SUBMISSION,
            metadata={'title': 'Test Paper'}
        )
        
        tasks = await integration.process_workflow_event(event)
        
        self.assertIsInstance(tasks, list)
        self.assertGreater(len(tasks), 0)


class TestIntegrationScenarios(unittest.IsolatedAsyncioTestCase):
    """Test complete integration scenarios"""
    
    @patch('workflow_integration.DatabaseManager')
    @patch('workflow_integration.CacheManager')
    async def test_submission_workflow(self, mock_cache, mock_db):
        """Test complete submission workflow"""
        mock_db_instance = MagicMock()
        mock_db_instance.execute_query.return_value = 12345
        mock_db.return_value = mock_db_instance
        
        mock_cache_instance = MagicMock()
        mock_cache.return_value = mock_cache_instance
        
        integration = WorkflowIntegration(
            {'host': 'localhost', 'database': 'test'},
            {'host': 'localhost'}
        )
        
        # Simulate submission created event
        event = WorkflowEvent(
            event_type='submission_created',
            submission_id=12345,
            stage=WorkflowStage.SUBMISSION,
            metadata={
                'title': 'Novel Peptide Formulation',
                'abstract': 'This study presents...',
                'keywords': ['peptides', 'anti-aging']
            }
        )
        
        # Process event
        tasks = await integration.process_workflow_event(event)
        
        # Verify tasks were created
        self.assertEqual(len(tasks), 2)  # Research Discovery + Submission Assistant
    
    @patch('workflow_integration.DatabaseManager')
    @patch('workflow_integration.CacheManager')
    async def test_review_workflow(self, mock_cache, mock_db):
        """Test review workflow"""
        mock_db_instance = MagicMock()
        mock_db_instance.execute_query.return_value = 12345
        mock_db.return_value = mock_db_instance
        
        mock_cache_instance = MagicMock()
        mock_cache.return_value = mock_cache_instance
        
        integration = WorkflowIntegration(
            {'host': 'localhost', 'database': 'test'},
            {'host': 'localhost'}
        )
        
        # Simulate review round created event
        event = WorkflowEvent(
            event_type='review_round_created',
            submission_id=12345,
            stage=WorkflowStage.REVIEW,
            metadata={'review_round_id': 1}
        )
        
        # Process event
        tasks = await integration.process_workflow_event(event)
        
        # Verify Review Coordination Agent was activated
        self.assertEqual(len(tasks), 1)


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestWorkflowEvent))
    suite.addTests(loader.loadTestsFromTestCase(TestAgentTask))
    suite.addTests(loader.loadTestsFromTestCase(TestDatabaseManager))
    suite.addTests(loader.loadTestsFromTestCase(TestCacheManager))
    suite.addTests(loader.loadTestsFromTestCase(TestWorkflowIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestWorkflowIntegrationAsync))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegrationScenarios))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
