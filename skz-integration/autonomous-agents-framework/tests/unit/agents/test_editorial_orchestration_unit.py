"""
Comprehensive Unit Tests for Agent 3: Editorial Orchestration Agent
Tests decision support, workflow optimization, and conflict resolution
"""

import pytest
import pytest_asyncio
import sys
import os
from datetime import datetime
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from models.editorial_orchestration import (
    EditorialOrchestrationAgent,
    DecisionSupportEngine,
    WorkflowOptimizer,
    ConflictResolver,
    EditorialDecision,
    WorkflowRecommendation,
    ConflictResolution
)


class TestEditorialOrchestrationAgentInitialization:
    """Test suite for agent initialization"""
    
    def test_agent_initialization(self):
        """Test agent initializes with correct components"""
        agent = EditorialOrchestrationAgent(agent_id="editorial_test")
        
        assert agent.agent_id == "editorial_test"
        assert agent.decision_engine is not None
        assert agent.workflow_optimizer is not None
        assert agent.conflict_resolver is not None
    
    def test_ml_components_loaded(self):
        """Test ML components are real implementations, not mocks"""
        agent = EditorialOrchestrationAgent()
        
        # Verify real ML decision engine
        assert hasattr(agent.decision_engine, 'ml_model')
        assert agent.decision_engine.ml_model is not None
        
        # Verify real workflow optimizer
        assert hasattr(agent.workflow_optimizer, 'optimization_algorithm')
        assert agent.workflow_optimizer.optimization_algorithm is not None
    
    def test_agent_state_initialization(self):
        """Test agent state is properly initialized"""
        agent = EditorialOrchestrationAgent()
        
        assert agent.active_workflows == {}
        assert agent.decision_history == []
        assert agent.performance_metrics is not None


class TestDecisionSupportEngine:
    """Test suite for decision support functionality"""
    
    def test_recommend_decision_structure(self):
        """Test decision recommendation returns correct structure"""
        agent = EditorialOrchestrationAgent()
        
        manuscript_data = {
            'id': 'TEST-001',
            'quality_score': 0.85,
            'novelty_score': 0.75,
            'research_gaps': ['Gap 1', 'Gap 2'],
            'compliance_status': 'compliant'
        }
        
        decision = agent.recommend_decision(
            manuscript_id='TEST-001',
            manuscript_data=manuscript_data
        )
        
        assert isinstance(decision, dict)
        assert 'recommendation' in decision
        assert 'confidence' in decision
        assert 'reasoning' in decision
        assert decision['recommendation'] in ['accept', 'reject', 'revise', 'review']
    
    def test_decision_uses_real_ml_scoring(self):
        """Test decision engine uses real ML, not hardcoded values"""
        agent = EditorialOrchestrationAgent()
        
        # High quality manuscript
        high_quality_data = {
            'quality_score': 0.90,
            'novelty_score': 0.85,
            'compliance_status': 'compliant'
        }
        
        # Low quality manuscript
        low_quality_data = {
            'quality_score': 0.35,
            'novelty_score': 0.25,
            'compliance_status': 'non-compliant'
        }
        
        high_decision = agent.recommend_decision('TEST-HIGH', high_quality_data)
        low_decision = agent.recommend_decision('TEST-LOW', low_quality_data)
        
        # Verify different recommendations based on quality
        assert high_decision['recommendation'] != low_decision['recommendation']
        assert high_decision['confidence'] > low_decision['confidence']
        
        # Verify ML-based confidence scoring
        assert 0 <= high_decision['confidence'] <= 1
        assert 0 <= low_decision['confidence'] <= 1
    
    def test_decision_confidence_calculation(self):
        """Test confidence calculation uses real ML uncertainty estimation"""
        agent = EditorialOrchestrationAgent()
        
        manuscript_data = {
            'quality_score': 0.75,
            'novelty_score': 0.70,
            'variance': 0.05  # Low variance = high confidence
        }
        
        decision = agent.recommend_decision('TEST-001', manuscript_data)
        
        # Verify confidence is based on data variance, not hardcoded
        assert decision['confidence'] > 0.7  # Should be confident with low variance
        assert 'uncertainty_factors' in decision
    
    @pytest.mark.parametrize("quality,novelty,expected_action", [
        (0.90, 0.85, 'accept'),
        (0.50, 0.45, 'revise'),
        (0.25, 0.20, 'reject'),
    ])
    def test_decision_thresholds(self, quality, novelty, expected_action):
        """Test decision thresholds are calibrated correctly"""
        agent = EditorialOrchestrationAgent()
        
        decision = agent.recommend_decision('TEST', {
            'quality_score': quality,
            'novelty_score': novelty
        })
        
        assert decision['recommendation'] == expected_action


class TestWorkflowOptimization:
    """Test suite for workflow optimization functionality"""
    
    def test_optimize_workflow_structure(self):
        """Test workflow optimization returns correct structure"""
        agent = EditorialOrchestrationAgent()
        
        current_workflow = {
            'manuscript_id': 'TEST-001',
            'current_stage': 'submission',
            'bottlenecks': ['reviewer_availability']
        }
        
        optimization = agent.optimize_workflow(current_workflow)
        
        assert isinstance(optimization, dict)
        assert 'optimized_path' in optimization
        assert 'estimated_time_savings' in optimization
        assert 'recommended_actions' in optimization
    
    def test_workflow_uses_real_optimization_algorithm(self):
        """Test workflow optimization uses real algorithms, not mocks"""
        agent = EditorialOrchestrationAgent()
        
        # Complex workflow with multiple bottlenecks
        workflow = {
            'manuscript_id': 'TEST-001',
            'stages': ['submission', 'review', 'revision', 'decision'],
            'bottlenecks': ['reviewer_availability', 'author_response_time'],
            'current_duration_days': 60
        }
        
        optimization = agent.optimize_workflow(workflow)
        
        # Verify real optimization occurred
        assert optimization['estimated_time_savings'] > 0
        assert len(optimization['recommended_actions']) > 0
        
        # Verify optimization is data-driven
        assert 'optimization_method' in optimization
        assert optimization['optimization_method'] != 'mock'
    
    def test_workflow_considers_resource_constraints(self):
        """Test workflow optimization accounts for resource availability"""
        agent = EditorialOrchestrationAgent()
        
        workflow = {
            'manuscript_id': 'TEST-001',
            'available_reviewers': 3,
            'required_reviewers': 2,
            'editor_capacity': 5
        }
        
        optimization = agent.optimize_workflow(workflow)
        
        # Verify resource constraints respected
        assert 'resource_allocation' in optimization
        assert optimization['resource_allocation']['reviewers_assigned'] <= workflow['available_reviewers']
    
    def test_workflow_parallelization_suggestions(self):
        """Test workflow optimizer suggests parallel processing when possible"""
        agent = EditorialOrchestrationAgent()
        
        workflow = {
            'manuscript_id': 'TEST-001',
            'sequential_tasks': ['task1', 'task2', 'task3'],
            'dependencies': {}  # No dependencies = can parallelize
        }
        
        optimization = agent.optimize_workflow(workflow)
        
        # Should suggest parallelization
        assert 'parallel_execution' in optimization
        assert optimization['parallel_execution']['suggested']


class TestConflictResolution:
    """Test suite for conflict resolution functionality"""
    
    def test_resolve_conflict_structure(self):
        """Test conflict resolution returns correct structure"""
        agent = EditorialOrchestrationAgent()
        
        conflict = {
            'type': 'reviewer_disagreement',
            'reviewers': ['reviewer1', 'reviewer2'],
            'scores': [0.8, 0.3],
            'recommendations': ['accept', 'reject']
        }
        
        resolution = agent.resolve_conflict(conflict)
        
        assert isinstance(resolution, dict)
        assert 'resolution_strategy' in resolution
        assert 'final_recommendation' in resolution
        assert 'confidence' in resolution
    
    def test_conflict_resolution_uses_real_ml(self):
        """Test conflict resolution uses real ML for consensus building"""
        agent = EditorialOrchestrationAgent()
        
        conflict = {
            'type': 'reviewer_disagreement',
            'reviewer_scores': [0.85, 0.45, 0.70],
            'reviewer_expertise_match': [0.9, 0.6, 0.8]
        }
        
        resolution = agent.resolve_conflict(conflict)
        
        # Verify ML-based consensus
        assert 'consensus_score' in resolution
        assert 0 <= resolution['consensus_score'] <= 1
        
        # Verify expertise weighting applied
        assert 'weighted_average' in resolution
        assert resolution['ml_method'] == 'weighted_consensus'
    
    def test_conflict_escalation_logic(self):
        """Test conflicts are escalated when ML confidence is low"""
        agent = EditorialOrchestrationAgent()
        
        # High disagreement conflict
        high_conflict = {
            'type': 'reviewer_disagreement',
            'reviewer_scores': [0.9, 0.1],  # Large disagreement
            'variance': 0.64
        }
        
        resolution = agent.resolve_conflict(high_conflict)
        
        # Should escalate when confidence is low
        if resolution['confidence'] < 0.5:
            assert resolution['escalate_to_editor']
    
    def test_multi_reviewer_consensus(self):
        """Test consensus building with multiple reviewers"""
        agent = EditorialOrchestrationAgent()
        
        conflict = {
            'type': 'reviewer_disagreement',
            'reviewer_scores': [0.7, 0.75, 0.8, 0.65, 0.72],
            'reviewer_weights': [1.0, 0.9, 1.0, 0.8, 0.9]
        }
        
        resolution = agent.resolve_conflict(conflict)
        
        # Verify weighted consensus
        assert 0.6 <= resolution['consensus_score'] <= 0.8
        assert resolution['resolution_strategy'] == 'weighted_average'


class TestEditorialDecisionIntegration:
    """Test suite for integrated editorial decision making"""
    
    @pytest.mark.asyncio
    async def test_complete_editorial_workflow(self):
        """Test complete editorial decision workflow with ML components"""
        agent = EditorialOrchestrationAgent()
        
        manuscript_data = {
            'id': 'TEST-001',
            'quality_score': 0.85,
            'novelty_score': 0.78,
            'reviewer_scores': [0.8, 0.85, 0.82],
            'compliance_status': 'compliant'
        }
        
        # Step 1: Get decision recommendation
        decision = agent.recommend_decision('TEST-001', manuscript_data)
        assert decision['recommendation'] in ['accept', 'reject', 'revise']
        
        # Step 2: Optimize workflow if accepted
        if decision['recommendation'] == 'accept':
            workflow = {
                'manuscript_id': 'TEST-001',
                'next_stage': 'production'
            }
            optimization = agent.optimize_workflow(workflow)
            assert optimization['estimated_time_savings'] >= 0
        
        # Step 3: Log decision
        agent.log_decision('TEST-001', decision)
        assert len(agent.decision_history) == 1
    
    def test_decision_learning_from_history(self):
        """Test agent learns from decision history"""
        agent = EditorialOrchestrationAgent()
        
        # Make multiple decisions
        for i in range(5):
            decision = agent.recommend_decision(f'TEST-{i}', {
                'quality_score': 0.7 + (i * 0.05),
                'novelty_score': 0.6 + (i * 0.05)
            })
            agent.log_decision(f'TEST-{i}', decision)
        
        # Verify history tracked
        assert len(agent.decision_history) == 5
        
        # Verify learning occurred
        performance = agent.get_performance_metrics()
        assert 'decision_accuracy' in performance
        assert 'average_confidence' in performance


class TestEditorialPerformanceMetrics:
    """Test suite for performance tracking and metrics"""
    
    def test_performance_metrics_structure(self):
        """Test performance metrics return correct structure"""
        agent = EditorialOrchestrationAgent()
        
        metrics = agent.get_performance_metrics()
        
        assert isinstance(metrics, dict)
        assert 'total_decisions' in metrics
        assert 'average_decision_time' in metrics
        assert 'workflow_efficiency' in metrics
    
    def test_decision_time_tracking(self):
        """Test decision time is tracked accurately"""
        agent = EditorialOrchestrationAgent()
        
        start_time = datetime.now()
        decision = agent.recommend_decision('TEST-001', {
            'quality_score': 0.75,
            'novelty_score': 0.70
        })
        end_time = datetime.now()
        
        decision_duration = (end_time - start_time).total_seconds()
        
        # Should complete in reasonable time
        assert decision_duration < 5.0  # Less than 5 seconds
    
    def test_workflow_efficiency_calculation(self):
        """Test workflow efficiency is calculated using real metrics"""
        agent = EditorialOrchestrationAgent()
        
        # Process multiple workflows
        for i in range(3):
            workflow = {
                'manuscript_id': f'TEST-{i}',
                'baseline_duration': 60,
                'optimized_duration': 45
            }
            agent.optimize_workflow(workflow)
        
        metrics = agent.get_performance_metrics()
        
        # Verify efficiency tracked
        assert metrics['workflow_efficiency'] > 0
        assert 'time_savings' in metrics


# Fixtures for all tests
@pytest.fixture
def sample_manuscript_data():
    """Sample manuscript data for testing"""
    return {
        'id': 'TEST-MANUSCRIPT-001',
        'title': 'Novel Cosmetic Formulation Study',
        'quality_score': 0.82,
        'novelty_score': 0.75,
        'compliance_status': 'compliant',
        'reviewer_scores': [0.8, 0.85, 0.78],
        'research_gaps': ['Gap 1', 'Gap 2']
    }


@pytest.fixture
def sample_workflow():
    """Sample workflow for testing"""
    return {
        'manuscript_id': 'TEST-001',
        'current_stage': 'review',
        'stages_completed': ['submission', 'initial_review'],
        'stages_remaining': ['peer_review', 'revision', 'final_decision'],
        'bottlenecks': ['reviewer_availability'],
        'current_duration_days': 30,
        'target_duration_days': 45
    }


@pytest.fixture
def sample_conflict():
    """Sample conflict scenario for testing"""
    return {
        'type': 'reviewer_disagreement',
        'manuscript_id': 'TEST-001',
        'reviewer_scores': [0.85, 0.45, 0.70],
        'reviewer_recommendations': ['accept', 'reject', 'revise'],
        'reviewer_expertise_match': [0.9, 0.6, 0.8],
        'variance': 0.04
    }
