"""
Ontogenetic Loom Module
Agent learning and evolution mechanism

This module implements the "ontogenetic loom" - a learning mechanism that
weaves agent experiences into improved capabilities over time. Each agent
has its own loom that continuously integrates feedback and adapts behavior.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import numpy as np
from collections import deque


class ExperienceType(Enum):
    """Types of experiences an agent can have"""
    SUCCESS = "success"
    FAILURE = "failure"
    FEEDBACK = "feedback"
    OBSERVATION = "observation"
    COLLABORATION = "collaboration"


@dataclass
class Experience:
    """A single experience in an agent's history"""
    id: str
    agent_id: str
    experience_type: ExperienceType
    context: Dict[str, Any]
    action: Dict[str, Any]
    outcome: Dict[str, Any]
    feedback: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert experience to dictionary"""
        return {
            'id': self.id,
            'agent_id': self.agent_id,
            'experience_type': self.experience_type.value,
            'context': self.context,
            'action': self.action,
            'outcome': self.outcome,
            'feedback': self.feedback,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class Pattern:
    """A learned pattern extracted from experiences"""
    id: str
    pattern_type: str
    conditions: Dict[str, Any]
    actions: Dict[str, Any]
    expected_outcome: Dict[str, Any]
    confidence: float
    usage_count: int = 0
    success_rate: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class CapabilityDelta:
    """Change in agent capabilities"""
    timestamp: datetime
    patterns_learned: int
    patterns_refined: int
    patterns_deprecated: int
    performance_improvement: float
    confidence_improvement: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class OntogeneticLoom:
    """
    Agent learning and evolution mechanism
    
    The ontogenetic loom weaves experiences into improved capabilities,
    implementing continuous learning and adaptation for autonomous agents.
    """
    
    def __init__(
        self, 
        agent_id: str,
        weaving_threshold: int = 10,
        pattern_confidence_threshold: float = 0.7,
        max_experience_buffer: int = 1000
    ):
        """
        Initialize the ontogenetic loom
        
        Args:
            agent_id: Identifier of the agent
            weaving_threshold: Number of experiences before weaving
            pattern_confidence_threshold: Minimum confidence for pattern usage
            max_experience_buffer: Maximum experiences to keep in buffer
        """
        self.agent_id = agent_id
        self.weaving_threshold = weaving_threshold
        self.pattern_confidence_threshold = pattern_confidence_threshold
        self.max_experience_buffer = max_experience_buffer
        
        self.experience_buffer: deque = deque(maxlen=max_experience_buffer)
        self.patterns: Dict[str, Pattern] = {}
        self.capability_model: Optional[Dict[str, Any]] = None
        self.evolution_history: List[CapabilityDelta] = []
        
        self.total_experiences = 0
        self.weaving_count = 0
        self.last_weaving = None
        
    def add_experience(
        self,
        experience_id: str,
        experience_type: ExperienceType,
        context: Dict[str, Any],
        action: Dict[str, Any],
        outcome: Dict[str, Any],
        feedback: Optional[Dict[str, Any]] = None
    ) -> Experience:
        """
        Add a new experience to the loom
        
        Args:
            experience_id: Unique identifier for the experience
            experience_type: Type of experience
            context: Context in which the action was taken
            action: Action that was performed
            outcome: Result of the action
            feedback: Optional feedback on the action
            
        Returns:
            The created Experience object
        """
        experience = Experience(
            id=experience_id,
            agent_id=self.agent_id,
            experience_type=experience_type,
            context=context,
            action=action,
            outcome=outcome,
            feedback=feedback
        )
        
        self.experience_buffer.append(experience)
        self.total_experiences += 1
        
        # Check if it's time to weave
        if len(self.experience_buffer) >= self.weaving_threshold:
            self.weave()
        
        return experience
    
    def weave(self) -> CapabilityDelta:
        """
        Weave accumulated experiences into improved capabilities
        
        This is the core learning mechanism that:
        1. Extracts patterns from experiences
        2. Updates existing patterns
        3. Deprecates ineffective patterns
        4. Updates the capability model
        
        Returns:
            CapabilityDelta describing the changes
        """
        print(f"[{self.agent_id}] Weaving {len(self.experience_buffer)} experiences...")
        
        # Extract patterns from experiences
        new_patterns = self._extract_patterns()
        
        # Update existing patterns
        refined_patterns = self._refine_patterns()
        
        # Deprecate ineffective patterns
        deprecated_patterns = self._deprecate_patterns()
        
        # Update capability model
        performance_improvement = self._update_capability_model()
        
        # Calculate confidence improvement
        confidence_improvement = self._calculate_confidence_improvement()
        
        # Create capability delta
        delta = CapabilityDelta(
            timestamp=datetime.now(),
            patterns_learned=len(new_patterns),
            patterns_refined=len(refined_patterns),
            patterns_deprecated=len(deprecated_patterns),
            performance_improvement=performance_improvement,
            confidence_improvement=confidence_improvement,
            metadata={
                'experiences_woven': len(self.experience_buffer),
                'total_patterns': len(self.patterns),
                'weaving_number': self.weaving_count + 1
            }
        )
        
        self.evolution_history.append(delta)
        self.weaving_count += 1
        self.last_weaving = datetime.now()
        
        # Clear experience buffer (they've been integrated)
        self.experience_buffer.clear()
        
        print(f"[{self.agent_id}] Weaving complete: {delta.patterns_learned} new, "
              f"{delta.patterns_refined} refined, {delta.patterns_deprecated} deprecated")
        
        return delta
    
    def _extract_patterns(self) -> List[Pattern]:
        """
        Extract new patterns from experiences
        
        Returns:
            List of newly discovered patterns
        """
        new_patterns = []
        
        # Group experiences by similarity
        experience_groups = self._group_similar_experiences()
        
        for group_id, experiences in experience_groups.items():
            if len(experiences) < 3:  # Need at least 3 similar experiences
                continue
            
            # Extract common pattern
            pattern = self._create_pattern_from_group(group_id, experiences)
            
            if pattern and pattern.confidence >= self.pattern_confidence_threshold:
                self.patterns[pattern.id] = pattern
                new_patterns.append(pattern)
        
        return new_patterns
    
    def _group_similar_experiences(self) -> Dict[str, List[Experience]]:
        """Group experiences by similarity"""
        groups: Dict[str, List[Experience]] = {}
        
        for exp in self.experience_buffer:
            # Simple grouping by experience type and action type
            group_key = f"{exp.experience_type.value}_{exp.action.get('type', 'unknown')}"
            
            if group_key not in groups:
                groups[group_key] = []
            
            groups[group_key].append(exp)
        
        return groups
    
    def _create_pattern_from_group(
        self, 
        group_id: str, 
        experiences: List[Experience]
    ) -> Optional[Pattern]:
        """
        Create a pattern from a group of similar experiences
        
        Args:
            group_id: Identifier for the experience group
            experiences: List of similar experiences
            
        Returns:
            Extracted Pattern or None
        """
        if not experiences:
            return None
        
        # Extract common conditions
        conditions = self._extract_common_conditions(experiences)
        
        # Extract common actions
        actions = self._extract_common_actions(experiences)
        
        # Calculate expected outcome
        expected_outcome = self._calculate_expected_outcome(experiences)
        
        # Calculate confidence based on consistency
        confidence = self._calculate_pattern_confidence(experiences)
        
        # Calculate success rate
        success_count = sum(
            1 for exp in experiences 
            if exp.experience_type == ExperienceType.SUCCESS
        )
        success_rate = success_count / len(experiences)
        
        pattern_id = f"pattern_{self.agent_id}_{group_id}_{self.weaving_count}"
        
        return Pattern(
            id=pattern_id,
            pattern_type=group_id,
            conditions=conditions,
            actions=actions,
            expected_outcome=expected_outcome,
            confidence=confidence,
            success_rate=success_rate,
            usage_count=len(experiences)
        )
    
    def _extract_common_conditions(self, experiences: List[Experience]) -> Dict[str, Any]:
        """Extract common conditions from experiences"""
        # Simple implementation: find most common context values
        common_conditions = {}
        
        if not experiences:
            return common_conditions
        
        # Get all context keys
        all_keys = set()
        for exp in experiences:
            all_keys.update(exp.context.keys())
        
        # For each key, find most common value
        for key in all_keys:
            values = [exp.context.get(key) for exp in experiences if key in exp.context]
            if values:
                # Use most common value
                common_conditions[key] = max(set(values), key=values.count)
        
        return common_conditions
    
    def _extract_common_actions(self, experiences: List[Experience]) -> Dict[str, Any]:
        """Extract common actions from experiences"""
        common_actions = {}
        
        if not experiences:
            return common_actions
        
        # Get all action keys
        all_keys = set()
        for exp in experiences:
            all_keys.update(exp.action.keys())
        
        # For each key, find most common value
        for key in all_keys:
            values = [exp.action.get(key) for exp in experiences if key in exp.action]
            if values:
                common_actions[key] = max(set(values), key=values.count)
        
        return common_actions
    
    def _calculate_expected_outcome(self, experiences: List[Experience]) -> Dict[str, Any]:
        """Calculate expected outcome from experiences"""
        # Average numeric outcomes, most common for categorical
        expected = {}
        
        if not experiences:
            return expected
        
        # Get all outcome keys
        all_keys = set()
        for exp in experiences:
            all_keys.update(exp.outcome.keys())
        
        for key in all_keys:
            values = [exp.outcome.get(key) for exp in experiences if key in exp.outcome]
            
            if not values:
                continue
            
            # Check if numeric
            if all(isinstance(v, (int, float)) for v in values):
                expected[key] = np.mean(values)
            else:
                # Use most common value
                expected[key] = max(set(values), key=values.count)
        
        return expected
    
    def _calculate_pattern_confidence(self, experiences: List[Experience]) -> float:
        """Calculate confidence in a pattern based on consistency"""
        if not experiences:
            return 0.0
        
        # Confidence based on:
        # 1. Number of experiences (more is better)
        # 2. Consistency of outcomes (less variance is better)
        # 3. Positive feedback ratio
        
        n = len(experiences)
        count_factor = min(n / 10.0, 1.0)  # Max at 10 experiences
        
        # Consistency: check if outcomes are similar
        success_count = sum(
            1 for exp in experiences 
            if exp.experience_type == ExperienceType.SUCCESS
        )
        consistency_factor = success_count / n
        
        # Feedback factor
        positive_feedback = sum(
            1 for exp in experiences 
            if exp.feedback and exp.feedback.get('rating', 0) > 0.5
        )
        feedback_factor = positive_feedback / n if n > 0 else 0.5
        
        confidence = (count_factor + consistency_factor + feedback_factor) / 3.0
        
        return confidence
    
    def _refine_patterns(self) -> List[str]:
        """Refine existing patterns based on new experiences"""
        refined = []
        
        for pattern_id, pattern in self.patterns.items():
            # Find experiences matching this pattern
            matching_experiences = self._find_matching_experiences(pattern)
            
            if matching_experiences:
                # Update pattern statistics
                old_confidence = pattern.confidence
                pattern.confidence = self._calculate_pattern_confidence(matching_experiences)
                pattern.usage_count += len(matching_experiences)
                pattern.updated_at = datetime.now()
                
                if abs(pattern.confidence - old_confidence) > 0.1:
                    refined.append(pattern_id)
        
        return refined
    
    def _find_matching_experiences(self, pattern: Pattern) -> List[Experience]:
        """Find experiences that match a pattern"""
        matching = []
        
        for exp in self.experience_buffer:
            # Check if context matches pattern conditions
            if self._context_matches_conditions(exp.context, pattern.conditions):
                matching.append(exp)
        
        return matching
    
    def _context_matches_conditions(
        self, 
        context: Dict[str, Any], 
        conditions: Dict[str, Any]
    ) -> bool:
        """Check if context matches pattern conditions"""
        # Simple exact match (can be made more sophisticated)
        for key, value in conditions.items():
            if key not in context or context[key] != value:
                return False
        return True
    
    def _deprecate_patterns(self) -> List[str]:
        """Deprecate patterns that are no longer effective"""
        deprecated = []
        
        for pattern_id, pattern in list(self.patterns.items()):
            # Deprecate if confidence too low or not used recently
            if pattern.confidence < self.pattern_confidence_threshold * 0.5:
                deprecated.append(pattern_id)
                del self.patterns[pattern_id]
        
        return deprecated
    
    def _update_capability_model(self) -> float:
        """
        Update the agent's capability model
        
        Returns:
            Performance improvement score
        """
        # Calculate average pattern confidence as capability metric
        if not self.patterns:
            avg_confidence = 0.0
        else:
            avg_confidence = np.mean([p.confidence for p in self.patterns.values()])
        
        old_capability = (
            self.capability_model.get('avg_confidence', 0.0) 
            if self.capability_model else 0.0
        )
        
        self.capability_model = {
            'avg_confidence': avg_confidence,
            'pattern_count': len(self.patterns),
            'total_experiences': self.total_experiences,
            'weaving_count': self.weaving_count,
            'updated_at': datetime.now().isoformat()
        }
        
        improvement = avg_confidence - old_capability
        return improvement
    
    def _calculate_confidence_improvement(self) -> float:
        """Calculate overall confidence improvement"""
        if len(self.evolution_history) < 2:
            return 0.0
        
        current_confidence = self.capability_model.get('avg_confidence', 0.0)
        previous_confidence = (
            self.evolution_history[-1].confidence_improvement 
            if self.evolution_history else 0.0
        )
        
        return current_confidence - previous_confidence
    
    def get_pattern_for_context(self, context: Dict[str, Any]) -> Optional[Pattern]:
        """
        Get the best matching pattern for a given context
        
        Args:
            context: Current context
            
        Returns:
            Best matching Pattern or None
        """
        best_pattern = None
        best_score = 0.0
        
        for pattern in self.patterns.values():
            # Calculate match score
            score = self._calculate_match_score(context, pattern)
            
            if score > best_score and pattern.confidence >= self.pattern_confidence_threshold:
                best_score = score
                best_pattern = pattern
        
        return best_pattern
    
    def _calculate_match_score(self, context: Dict[str, Any], pattern: Pattern) -> float:
        """Calculate how well a context matches a pattern"""
        if not pattern.conditions:
            return 0.0
        
        matching_keys = 0
        total_keys = len(pattern.conditions)
        
        for key, value in pattern.conditions.items():
            if key in context and context[key] == value:
                matching_keys += 1
        
        match_ratio = matching_keys / total_keys if total_keys > 0 else 0.0
        
        # Weight by pattern confidence
        score = match_ratio * pattern.confidence
        
        return score
    
    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get summary of agent evolution"""
        return {
            'agent_id': self.agent_id,
            'total_experiences': self.total_experiences,
            'weaving_count': self.weaving_count,
            'pattern_count': len(self.patterns),
            'capability_model': self.capability_model,
            'last_weaving': self.last_weaving.isoformat() if self.last_weaving else None,
            'evolution_history_length': len(self.evolution_history),
            'average_performance_improvement': (
                np.mean([d.performance_improvement for d in self.evolution_history])
                if self.evolution_history else 0.0
            )
        }
    
    def export_to_json(self, filepath: str):
        """Export loom state to JSON file"""
        data = {
            'agent_id': self.agent_id,
            'config': {
                'weaving_threshold': self.weaving_threshold,
                'pattern_confidence_threshold': self.pattern_confidence_threshold,
                'max_experience_buffer': self.max_experience_buffer
            },
            'patterns': {
                pid: {
                    'id': p.id,
                    'pattern_type': p.pattern_type,
                    'conditions': p.conditions,
                    'actions': p.actions,
                    'expected_outcome': p.expected_outcome,
                    'confidence': p.confidence,
                    'usage_count': p.usage_count,
                    'success_rate': p.success_rate,
                    'created_at': p.created_at.isoformat(),
                    'updated_at': p.updated_at.isoformat()
                }
                for pid, p in self.patterns.items()
            },
            'capability_model': self.capability_model,
            'evolution_history': [
                {
                    'timestamp': d.timestamp.isoformat(),
                    'patterns_learned': d.patterns_learned,
                    'patterns_refined': d.patterns_refined,
                    'patterns_deprecated': d.patterns_deprecated,
                    'performance_improvement': d.performance_improvement,
                    'confidence_improvement': d.confidence_improvement,
                    'metadata': d.metadata
                }
                for d in self.evolution_history
            ],
            'statistics': self.get_evolution_summary(),
            'exported_at': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


# Example usage
if __name__ == "__main__":
    # Initialize loom for an agent
    loom = OntogeneticLoom("editorial_agent", weaving_threshold=5)
    
    # Simulate experiences
    for i in range(12):
        loom.add_experience(
            experience_id=f"exp_{i}",
            experience_type=ExperienceType.SUCCESS if i % 3 != 0 else ExperienceType.FAILURE,
            context={'manuscript_quality': 'high' if i % 2 == 0 else 'medium'},
            action={'decision': 'accept' if i % 2 == 0 else 'revise'},
            outcome={'processing_time': 5.0 + i * 0.5, 'quality_maintained': True},
            feedback={'rating': 0.8 if i % 2 == 0 else 0.6}
        )
    
    # Get evolution summary
    summary = loom.get_evolution_summary()
    print(f"Evolution summary: {json.dumps(summary, indent=2)}")
    
    # Get pattern for new context
    pattern = loom.get_pattern_for_context({'manuscript_quality': 'high'})
    if pattern:
        print(f"Best pattern: {pattern.id} (confidence: {pattern.confidence:.2f})")
