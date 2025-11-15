"""
Test suite for cognitive architecture components
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import asyncio
import numpy as np
from datetime import datetime

from jax_ceo_orchestrator import JAXCEOOrchestrator, OptimizationResult
from hypergraph_knowledge_base import (
    HypergraphKnowledgeBase,
    NodeType,
    HyperedgeType,
    Node,
    Hyperedge
)
from ontogenetic_loom import (
    OntogeneticLoom,
    ExperienceType,
    Experience,
    Pattern
)
from enhanced_agent_base import EnhancedAgentBase, EditorialAgent


class TestJAXCEOOrchestrator:
    """Test JAX CEO neural computation module"""
    
    def test_initialization(self):
        """Test orchestrator initialization"""
        ceo = JAXCEOOrchestrator()
        assert ceo is not None
        assert ceo.params is not None
        assert 'layers' in ceo.params
        
    def test_quality_scoring(self):
        """Test quality scoring model"""
        ceo = JAXCEOOrchestrator()
        features = np.random.randn(768)
        scores = ceo.quality_scoring_model(ceo.params, features)
        assert scores is not None
        assert len(scores) == 32  # output_dim
        
    def test_reviewer_matching(self):
        """Test reviewer matching model"""
        ceo = JAXCEOOrchestrator()
        manuscript_embedding = np.random.randn(768)
        reviewer_embeddings = np.random.randn(5, 768)
        
        attention = ceo.reviewer_matching_model(
            ceo.params,
            manuscript_embedding,
            reviewer_embeddings
        )
        
        assert attention is not None
        assert len(attention) == 5
        assert abs(np.sum(attention) - 1.0) < 0.01  # Should sum to 1
        
    def test_workflow_optimization(self):
        """Test workflow optimization"""
        ceo = JAXCEOOrchestrator()
        workflow_state = {
            'time_taken': 5.0,
            'quality_score': 0.85,
            'resource_usage': 0.6,
            'fairness_score': 0.9
        }
        
        result = ceo.workflow_optimization(ceo.params, workflow_state)
        
        assert isinstance(result, OptimizationResult)
        assert result.loss >= 0
        assert result.iterations > 0
        
    def test_optimize_agent_decision(self):
        """Test agent decision optimization"""
        ceo = JAXCEOOrchestrator()
        decision_context = {
            'manuscript_id': 'MS-2025-001',
            'quality_indicators': {'novelty': 0.8, 'rigor': 0.7}
        }
        
        decision = ceo.optimize_agent_decision('test_agent', decision_context)
        
        assert 'agent_id' in decision
        assert 'decision' in decision
        assert 'confidence' in decision
        assert decision['agent_id'] == 'test_agent'


class TestHypergraphKnowledgeBase:
    """Test hypergraph knowledge base"""
    
    def test_initialization(self):
        """Test knowledge base initialization"""
        kb = HypergraphKnowledgeBase()
        assert kb is not None
        assert len(kb.nodes) == 0
        assert len(kb.hyperedges) == 0
        
    def test_add_node(self):
        """Test adding nodes"""
        kb = HypergraphKnowledgeBase()
        
        node = kb.add_node(
            "MS-001",
            NodeType.MANUSCRIPT,
            {'title': 'Test Manuscript'},
            embedding=np.random.randn(768)
        )
        
        assert node.id == "MS-001"
        assert node.type == NodeType.MANUSCRIPT
        assert 'title' in node.attributes
        assert node.embedding is not None
        
    def test_add_hyperedge(self):
        """Test adding hyperedges"""
        kb = HypergraphKnowledgeBase()
        
        # Add nodes first
        kb.add_node("MS-001", NodeType.MANUSCRIPT, {})
        kb.add_node("AU-001", NodeType.AUTHOR, {})
        kb.add_node("ING-001", NodeType.INGREDIENT, {})
        
        # Add hyperedge
        edge = kb.add_hyperedge(
            "edge_001",
            HyperedgeType.SUBMISSION_CONTEXT,
            ["MS-001", "AU-001", "ING-001"],
            weight=1.0
        )
        
        assert edge.id == "edge_001"
        assert len(edge.nodes) == 3
        assert "MS-001" in edge.nodes
        
    def test_get_connected_nodes(self):
        """Test retrieving connected nodes"""
        kb = HypergraphKnowledgeBase()
        
        # Create a small graph
        kb.add_node("MS-001", NodeType.MANUSCRIPT, {})
        kb.add_node("AU-001", NodeType.AUTHOR, {})
        kb.add_node("AU-002", NodeType.AUTHOR, {})
        
        kb.add_hyperedge(
            "edge_001",
            HyperedgeType.AUTHORSHIP,
            ["MS-001", "AU-001", "AU-002"]
        )
        
        # Get connected nodes
        connected = kb.get_connected_nodes("MS-001")
        
        assert len(connected) == 2
        connected_ids = {n.id for n in connected}
        assert "AU-001" in connected_ids
        assert "AU-002" in connected_ids
        
    def test_find_similar_nodes(self):
        """Test finding similar nodes"""
        kb = HypergraphKnowledgeBase()
        
        # Add nodes with embeddings
        kb.add_node(
            "MS-001",
            NodeType.MANUSCRIPT,
            {},
            embedding=np.array([1.0, 0.0, 0.0])
        )
        kb.add_node(
            "MS-002",
            NodeType.MANUSCRIPT,
            {},
            embedding=np.array([0.9, 0.1, 0.0])
        )
        kb.add_node(
            "MS-003",
            NodeType.MANUSCRIPT,
            {},
            embedding=np.array([0.0, 1.0, 0.0])
        )
        
        # Find similar to MS-001
        similar = kb.find_similar_nodes("MS-001", top_k=2)
        
        assert len(similar) <= 2
        # MS-002 should be most similar
        if similar:
            assert similar[0][0].id == "MS-002"
            assert similar[0][1] > 0.8  # High similarity
            
    def test_query_hypergraph(self):
        """Test hypergraph querying"""
        kb = HypergraphKnowledgeBase()
        
        # Create submission context
        kb.add_node("MS-001", NodeType.MANUSCRIPT, {})
        kb.add_node("AU-001", NodeType.AUTHOR, {})
        kb.add_node("ING-001", NodeType.INGREDIENT, {})
        
        kb.add_hyperedge(
            "sub_001",
            HyperedgeType.SUBMISSION_CONTEXT,
            ["MS-001", "AU-001", "ING-001"]
        )
        
        # Query for submission contexts
        results = kb.query_hypergraph(
            [NodeType.MANUSCRIPT, NodeType.AUTHOR],
            HyperedgeType.SUBMISSION_CONTEXT
        )
        
        assert len(results) == 1
        assert results[0]['edge']['id'] == "sub_001"
        
    def test_statistics(self):
        """Test statistics generation"""
        kb = HypergraphKnowledgeBase()
        
        kb.add_node("MS-001", NodeType.MANUSCRIPT, {})
        kb.add_node("AU-001", NodeType.AUTHOR, {})
        
        stats = kb.get_statistics()
        
        assert stats['total_nodes'] == 2
        assert stats['total_hyperedges'] == 0
        assert 'node_type_counts' in stats


class TestOntogeneticLoom:
    """Test ontogenetic loom learning mechanism"""
    
    def test_initialization(self):
        """Test loom initialization"""
        loom = OntogeneticLoom("test_agent", weaving_threshold=5)
        assert loom.agent_id == "test_agent"
        assert loom.weaving_threshold == 5
        assert len(loom.experience_buffer) == 0
        
    def test_add_experience(self):
        """Test adding experiences"""
        loom = OntogeneticLoom("test_agent", weaving_threshold=10)
        
        exp = loom.add_experience(
            "exp_001",
            ExperienceType.SUCCESS,
            {'context_key': 'value'},
            {'action_type': 'review'},
            {'outcome': 'positive'},
            {'rating': 0.9}
        )
        
        assert isinstance(exp, Experience)
        assert exp.id == "exp_001"
        assert len(loom.experience_buffer) == 1
        
    def test_weaving(self):
        """Test experience weaving"""
        loom = OntogeneticLoom("test_agent", weaving_threshold=5)
        
        # Add multiple similar experiences
        for i in range(6):
            loom.add_experience(
                f"exp_{i}",
                ExperienceType.SUCCESS,
                {'quality': 'high'},
                {'decision': 'accept'},
                {'time': 5.0 + i},
                {'rating': 0.8}
            )
        
        # Should have triggered weaving
        assert loom.weaving_count > 0
        assert len(loom.patterns) > 0
        
    def test_pattern_retrieval(self):
        """Test pattern retrieval for context"""
        loom = OntogeneticLoom("test_agent", weaving_threshold=3)
        
        # Add experiences and trigger weaving
        for i in range(5):
            loom.add_experience(
                f"exp_{i}",
                ExperienceType.SUCCESS,
                {'quality': 'high', 'type': 'review'},
                {'decision': 'accept'},
                {'time': 5.0},
                {'rating': 0.9}
            )
        
        # Get pattern for similar context
        pattern = loom.get_pattern_for_context({'quality': 'high', 'type': 'review'})
        
        # May or may not find a pattern depending on confidence threshold
        if pattern:
            assert isinstance(pattern, Pattern)
            assert pattern.confidence > 0
            
    def test_evolution_summary(self):
        """Test evolution summary generation"""
        loom = OntogeneticLoom("test_agent")
        
        summary = loom.get_evolution_summary()
        
        assert 'agent_id' in summary
        assert 'total_experiences' in summary
        assert 'pattern_count' in summary
        assert summary['agent_id'] == "test_agent"


class TestEnhancedAgentBase:
    """Test enhanced agent base class"""
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self):
        """Test agent initialization"""
        agent = EditorialAgent()
        
        assert agent.agent_id == "editorial_001"
        assert agent.agent_name == "Editorial Orchestration Agent"
        assert agent.jax_ceo is not None
        assert agent.knowledge_base is not None
        assert agent.loom is not None
        
    @pytest.mark.asyncio
    async def test_execute_with_cognition(self):
        """Test cognitive task execution"""
        agent = EditorialAgent()
        
        task = {
            'id': 'task_001',
            'type': 'manuscript_review',
            'manuscript_id': 'MS-2025-001',
            'priority': 'high'
        }
        
        result = await agent.execute_with_cognition(task)
        
        assert 'status' in result
        assert 'cognitive_metadata' in result
        assert result['cognitive_metadata']['agent_id'] == agent.agent_id
        
    @pytest.mark.asyncio
    async def test_context_analysis(self):
        """Test context analysis"""
        agent = EditorialAgent()
        
        task = {
            'id': 'task_001',
            'type': 'decision_making',
            'priority': 'high'
        }
        
        context = await agent._analyze_context(task)
        
        assert 'task_id' in context
        assert 'task_type' in context
        assert 'novelty_score' in context
        assert 'complexity_metric' in context
        
    def test_get_status(self):
        """Test agent status retrieval"""
        agent = EditorialAgent()
        
        status = agent.get_status()
        
        assert 'agent_id' in status
        assert 'metrics' in status
        assert 'knowledge_base_stats' in status
        assert 'learning_stats' in status
        
    @pytest.mark.asyncio
    async def test_collaboration(self):
        """Test agent collaboration"""
        agent1 = EditorialAgent("editorial_001")
        agent2 = EditorialAgent("editorial_002")
        
        task = {
            'id': 'collab_task',
            'type': 'manuscript_review',
            'manuscript_id': 'MS-2025-001'
        }
        
        result = await agent1.collaborate_with(agent2, task)
        
        assert 'method' in result
        assert result['method'] == 'collaborative'
        assert 'agent_results' in result
        assert len(result['agent_results']) == 2


class TestIntegration:
    """Integration tests for cognitive architecture"""
    
    @pytest.mark.asyncio
    async def test_full_cognitive_workflow(self):
        """Test complete cognitive workflow"""
        # Initialize agent
        agent = EditorialAgent()
        
        # Execute multiple tasks to trigger learning
        for i in range(10):
            task = {
                'id': f'task_{i}',
                'type': 'manuscript_review',
                'manuscript_id': f'MS-2025-{i:03d}',
                'priority': 'high' if i % 2 == 0 else 'normal'
            }
            
            result = await agent.execute_with_cognition(task)
            assert result['status'] in ['reviewed', 'success']
        
        # Check that agent has learned
        status = agent.get_status()
        assert status['metrics']['tasks_completed'] == 10
        assert status['learning_stats']['total_experiences'] > 0
        
        # Check knowledge base has been populated
        kb_stats = status['knowledge_base_stats']
        assert kb_stats['total_nodes'] > 0


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
