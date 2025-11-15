"""
Enhanced Agent Base Class
Integrates cognitive architecture components into agent framework

This module provides an enhanced base class for agents that integrates:
- JAX CEO neural computation
- Hypergraph knowledge representation
- Ontogenetic loom learning
- Deep Tree Echo / Marduk principles
"""

from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
from datetime import datetime
import asyncio
import logging

from jax_ceo_orchestrator import JAXCEOOrchestrator, OptimizationResult
from hypergraph_knowledge_base import (
    HypergraphKnowledgeBase, 
    NodeType, 
    HyperedgeType
)
from ontogenetic_loom import (
    OntogeneticLoom, 
    ExperienceType, 
    Experience, 
    Pattern
)


logger = logging.getLogger(__name__)


class CognitiveMode:
    """Cognitive processing modes"""
    DEEP_TREE_ECHO = "deep_tree_echo"  # Novelty detection, intuition
    MARDUK = "marduk"  # Metric analysis, logic
    INTEGRATED = "integrated"  # Balanced hemispheric processing


class EnhancedAgentBase(ABC):
    """
    Enhanced base class for autonomous agents
    
    Integrates cognitive architecture components:
    - JAX CEO for neural optimization
    - Hypergraph KB for knowledge representation
    - Ontogenetic loom for learning
    - Hemispheric processing (Deep Tree Echo / Marduk)
    """
    
    def __init__(
        self,
        agent_id: str,
        agent_name: str,
        agent_type: str,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize enhanced agent
        
        Args:
            agent_id: Unique agent identifier
            agent_name: Human-readable agent name
            agent_type: Type of agent (e.g., 'editorial', 'review')
            config: Optional configuration dictionary
        """
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.agent_type = agent_type
        self.config = config or {}
        
        # Initialize cognitive components
        self.jax_ceo = JAXCEOOrchestrator(
            config=self.config.get('jax_ceo', {})
        )
        self.knowledge_base = HypergraphKnowledgeBase()
        self.loom = OntogeneticLoom(
            agent_id=agent_id,
            weaving_threshold=self.config.get('weaving_threshold', 10)
        )
        
        # Cognitive state
        self.cognitive_mode = CognitiveMode.INTEGRATED
        self.current_task = None
        self.task_history = []
        
        # Performance metrics
        self.metrics = {
            'tasks_completed': 0,
            'success_rate': 0.0,
            'average_processing_time': 0.0,
            'learning_rate': 0.0
        }
        
        logger.info(f"Initialized enhanced agent: {agent_name} ({agent_id})")
    
    @abstractmethod
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a task (to be implemented by subclasses)
        
        Args:
            task: Task specification
            
        Returns:
            Task result
        """
        pass
    
    async def execute_with_cognition(
        self, 
        task: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute task with full cognitive processing
        
        This method wraps task execution with:
        1. Context analysis (Deep Tree Echo / Marduk)
        2. Neural optimization (JAX CEO)
        3. Knowledge integration (Hypergraph)
        4. Experience recording (Ontogenetic Loom)
        
        Args:
            task: Task to execute
            
        Returns:
            Task result with cognitive metadata
        """
        start_time = datetime.now()
        self.current_task = task
        
        try:
            # Phase 1: Analyze context
            context_analysis = await self._analyze_context(task)
            
            # Phase 2: Retrieve relevant knowledge
            relevant_knowledge = self._retrieve_knowledge(context_analysis)
            
            # Phase 3: Check for learned patterns
            pattern = self.loom.get_pattern_for_context(context_analysis)
            
            # Phase 4: Execute task (with or without pattern)
            if pattern and pattern.confidence > 0.8:
                # Use learned pattern
                result = await self._execute_with_pattern(task, pattern)
            else:
                # Execute normally
                result = await self.process_task(task)
            
            # Phase 5: Optimize decision using JAX CEO
            optimized_result = self._optimize_decision(result, context_analysis)
            
            # Phase 6: Record experience
            self._record_experience(
                task, 
                context_analysis, 
                optimized_result,
                success=True
            )
            
            # Phase 7: Update knowledge base
            self._update_knowledge_base(task, optimized_result)
            
            # Update metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_metrics(success=True, processing_time=processing_time)
            
            return {
                **optimized_result,
                'cognitive_metadata': {
                    'agent_id': self.agent_id,
                    'cognitive_mode': self.cognitive_mode,
                    'pattern_used': pattern.id if pattern else None,
                    'processing_time': processing_time,
                    'knowledge_nodes_accessed': len(relevant_knowledge),
                    'optimization_applied': True
                }
            }
            
        except Exception as e:
            logger.error(f"Error in cognitive execution: {e}")
            
            # Record failure experience
            self._record_experience(
                task,
                {'error': str(e)},
                {'status': 'failed'},
                success=False
            )
            
            # Update metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_metrics(success=False, processing_time=processing_time)
            
            raise
        
        finally:
            self.current_task = None
    
    async def _analyze_context(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze task context using hemispheric processing
        
        Deep Tree Echo: Novelty detection, pattern recognition
        Marduk: Metric calculation, categorical analysis
        """
        context = {
            'task_id': task.get('id', 'unknown'),
            'task_type': task.get('type', 'unknown'),
            'timestamp': datetime.now().isoformat()
        }
        
        # Deep Tree Echo: Detect novelty
        if self.cognitive_mode in [CognitiveMode.DEEP_TREE_ECHO, CognitiveMode.INTEGRATED]:
            context['novelty_score'] = self._detect_novelty(task)
            context['pattern_signature'] = self._extract_pattern_signature(task)
        
        # Marduk: Calculate metrics
        if self.cognitive_mode in [CognitiveMode.MARDUK, CognitiveMode.INTEGRATED]:
            context['complexity_metric'] = self._calculate_complexity(task)
            context['priority_score'] = self._calculate_priority(task)
            context['resource_estimate'] = self._estimate_resources(task)
        
        return context
    
    def _detect_novelty(self, task: Dict[str, Any]) -> float:
        """
        Detect novelty in task (Deep Tree Echo)
        
        Returns:
            Novelty score (0.0 to 1.0)
        """
        # Compare with task history
        if not self.task_history:
            return 1.0  # First task is completely novel
        
        # Simple novelty: check if task type seen before
        task_type = task.get('type', 'unknown')
        seen_types = {t.get('type') for t in self.task_history}
        
        if task_type not in seen_types:
            return 0.9  # High novelty
        
        # Check for novel attributes
        task_attrs = set(task.keys())
        historical_attrs = set()
        for hist_task in self.task_history:
            if hist_task.get('type') == task_type:
                historical_attrs.update(hist_task.keys())
        
        novel_attrs = task_attrs - historical_attrs
        novelty = len(novel_attrs) / len(task_attrs) if task_attrs else 0.0
        
        return novelty
    
    def _extract_pattern_signature(self, task: Dict[str, Any]) -> str:
        """Extract pattern signature from task"""
        # Simple signature: task type + key attributes
        task_type = task.get('type', 'unknown')
        key_attrs = sorted([k for k in task.keys() if k != 'id'])
        return f"{task_type}::{':'.join(key_attrs)}"
    
    def _calculate_complexity(self, task: Dict[str, Any]) -> float:
        """
        Calculate task complexity (Marduk metric)
        
        Returns:
            Complexity score (0.0 to 1.0)
        """
        # Factors: number of attributes, nested structures, data size
        attr_count = len(task)
        
        # Count nested structures
        nested_count = sum(
            1 for v in task.values() 
            if isinstance(v, (dict, list))
        )
        
        # Normalize to 0-1 range
        complexity = min((attr_count + nested_count * 2) / 20.0, 1.0)
        
        return complexity
    
    def _calculate_priority(self, task: Dict[str, Any]) -> float:
        """Calculate task priority"""
        # Check explicit priority
        if 'priority' in task:
            priority_map = {'low': 0.3, 'normal': 0.5, 'high': 0.8, 'urgent': 1.0}
            return priority_map.get(task['priority'], 0.5)
        
        # Default priority
        return 0.5
    
    def _estimate_resources(self, task: Dict[str, Any]) -> Dict[str, float]:
        """Estimate resource requirements"""
        complexity = self._calculate_complexity(task)
        
        return {
            'cpu': complexity * 0.5,
            'memory': complexity * 0.3,
            'time': complexity * 10.0  # seconds
        }
    
    def _retrieve_knowledge(self, context: Dict[str, Any]) -> List[Any]:
        """Retrieve relevant knowledge from hypergraph"""
        # Placeholder: In practice, would do semantic search
        relevant_nodes = []
        
        # Get nodes related to task type
        task_type = context.get('task_type', 'unknown')
        
        # Simple retrieval based on recent additions
        # In practice, would use embeddings and similarity search
        
        return relevant_nodes
    
    async def _execute_with_pattern(
        self, 
        task: Dict[str, Any], 
        pattern: Pattern
    ) -> Dict[str, Any]:
        """Execute task using a learned pattern"""
        logger.info(f"Executing with pattern: {pattern.id}")
        
        # Apply pattern actions
        result = {
            'status': 'success',
            'method': 'pattern_based',
            'pattern_id': pattern.id,
            'confidence': pattern.confidence,
            'actions_taken': pattern.actions,
            'expected_outcome': pattern.expected_outcome
        }
        
        return result
    
    def _optimize_decision(
        self, 
        result: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize decision using JAX CEO"""
        # Use JAX CEO to optimize the decision
        optimized = self.jax_ceo.optimize_agent_decision(
            self.agent_id,
            {**context, 'initial_result': result}
        )
        
        # Merge optimization with result
        return {
            **result,
            'optimization': optimized,
            'confidence': optimized.get('confidence', result.get('confidence', 0.5))
        }
    
    def _record_experience(
        self,
        task: Dict[str, Any],
        context: Dict[str, Any],
        result: Dict[str, Any],
        success: bool
    ):
        """Record experience in ontogenetic loom"""
        experience_type = ExperienceType.SUCCESS if success else ExperienceType.FAILURE
        
        self.loom.add_experience(
            experience_id=f"exp_{self.agent_id}_{datetime.now().timestamp()}",
            experience_type=experience_type,
            context=context,
            action={
                'task_type': task.get('type', 'unknown'),
                'method': result.get('method', 'standard')
            },
            outcome={
                'status': result.get('status', 'unknown'),
                'confidence': result.get('confidence', 0.0)
            },
            feedback={
                'success': success,
                'rating': 1.0 if success else 0.0
            }
        )
    
    def _update_knowledge_base(self, task: Dict[str, Any], result: Dict[str, Any]):
        """Update hypergraph knowledge base"""
        # Add task as a node (if not exists)
        task_id = task.get('id', f"task_{datetime.now().timestamp()}")
        
        # Determine node type based on task
        node_type = NodeType.DECISION  # Default
        
        self.knowledge_base.add_node(
            task_id,
            node_type,
            {
                'task_type': task.get('type'),
                'result_status': result.get('status'),
                'confidence': result.get('confidence', 0.0),
                'timestamp': datetime.now().isoformat()
            }
        )
    
    def _update_metrics(self, success: bool, processing_time: float):
        """Update agent performance metrics"""
        self.metrics['tasks_completed'] += 1
        
        # Update success rate
        total_tasks = self.metrics['tasks_completed']
        current_successes = self.metrics['success_rate'] * (total_tasks - 1)
        new_successes = current_successes + (1 if success else 0)
        self.metrics['success_rate'] = new_successes / total_tasks
        
        # Update average processing time
        current_avg = self.metrics['average_processing_time']
        self.metrics['average_processing_time'] = (
            (current_avg * (total_tasks - 1) + processing_time) / total_tasks
        )
        
        # Update learning rate (from loom)
        evolution_summary = self.loom.get_evolution_summary()
        self.metrics['learning_rate'] = evolution_summary.get(
            'average_performance_improvement', 
            0.0
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            'agent_id': self.agent_id,
            'agent_name': self.agent_name,
            'agent_type': self.agent_type,
            'cognitive_mode': self.cognitive_mode,
            'current_task': self.current_task.get('id') if self.current_task else None,
            'metrics': self.metrics,
            'knowledge_base_stats': self.knowledge_base.get_statistics(),
            'learning_stats': self.loom.get_evolution_summary(),
            'timestamp': datetime.now().isoformat()
        }
    
    def set_cognitive_mode(self, mode: str):
        """Set cognitive processing mode"""
        if mode in [CognitiveMode.DEEP_TREE_ECHO, CognitiveMode.MARDUK, CognitiveMode.INTEGRATED]:
            self.cognitive_mode = mode
            logger.info(f"Cognitive mode set to: {mode}")
        else:
            logger.warning(f"Invalid cognitive mode: {mode}")
    
    async def collaborate_with(
        self, 
        other_agent: 'EnhancedAgentBase', 
        task: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Collaborate with another agent on a task
        
        Args:
            other_agent: Another enhanced agent
            task: Collaborative task
            
        Returns:
            Collaborative result
        """
        logger.info(f"Collaboration: {self.agent_name} <-> {other_agent.agent_name}")
        
        # Execute tasks in parallel
        results = await asyncio.gather(
            self.execute_with_cognition(task),
            other_agent.execute_with_cognition(task)
        )
        
        # Merge results
        merged_result = self._merge_results(results[0], results[1])
        
        # Record collaboration experience
        self._record_experience(
            task,
            {'collaboration_with': other_agent.agent_id},
            merged_result,
            success=True
        )
        
        return merged_result
    
    def _merge_results(
        self, 
        result1: Dict[str, Any], 
        result2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge results from collaborative execution"""
        # Simple merging: average confidence, combine insights
        return {
            'status': 'success',
            'method': 'collaborative',
            'confidence': (
                result1.get('confidence', 0.5) + 
                result2.get('confidence', 0.5)
            ) / 2.0,
            'agent_results': [result1, result2]
        }
    
    def save_state(self, filepath: str):
        """Save agent state to file"""
        import json
        
        state = {
            'agent_id': self.agent_id,
            'agent_name': self.agent_name,
            'agent_type': self.agent_type,
            'cognitive_mode': self.cognitive_mode,
            'metrics': self.metrics,
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        # Save cognitive components
        self.loom.export_to_json(f"{filepath}.loom.json")
        self.knowledge_base.export_to_json(f"{filepath}.kb.json")
        self.jax_ceo.save_model(f"{filepath}.jax.json")


# Example concrete agent implementation
class EditorialAgent(EnhancedAgentBase):
    """Editorial orchestration agent with cognitive capabilities"""
    
    def __init__(self, agent_id: str = "editorial_001"):
        super().__init__(
            agent_id=agent_id,
            agent_name="Editorial Orchestration Agent",
            agent_type="editorial"
        )
    
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process editorial task"""
        task_type = task.get('type', 'unknown')
        
        if task_type == 'manuscript_review':
            return await self._review_manuscript(task)
        elif task_type == 'decision_making':
            return await self._make_decision(task)
        else:
            return {'status': 'unknown_task_type', 'task_type': task_type}
    
    async def _review_manuscript(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Review manuscript"""
        manuscript_id = task.get('manuscript_id', 'unknown')
        
        # Simulate review process
        await asyncio.sleep(0.1)
        
        return {
            'status': 'reviewed',
            'manuscript_id': manuscript_id,
            'decision': 'accept_with_revisions',
            'confidence': 0.85
        }
    
    async def _make_decision(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Make editorial decision"""
        # Simulate decision process
        await asyncio.sleep(0.1)
        
        return {
            'status': 'decided',
            'decision': 'approved',
            'confidence': 0.90
        }


# Example usage
if __name__ == "__main__":
    async def main():
        # Create enhanced agent
        agent = EditorialAgent()
        
        # Execute task with cognition
        task = {
            'id': 'task_001',
            'type': 'manuscript_review',
            'manuscript_id': 'MS-2025-001',
            'priority': 'high'
        }
        
        result = await agent.execute_with_cognition(task)
        print(f"Result: {result}")
        
        # Get agent status
        status = agent.get_status()
        print(f"Agent status: {status}")
    
    asyncio.run(main())
