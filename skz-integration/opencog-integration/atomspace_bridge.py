"""
atomspace_bridge.py - OpenCog AtomSpace Integration Bridge

Integrates OpenCog AGI framework for knowledge representation and reasoning
in the autonomous research journal system.

This module implements the integration layer between OJSCog and the OpenCog
AGI framework ecosystem (occ), preparing for future integration with hurdcog
(modified GNU Hurd OS) and cognumach (GNU Mach microkernel).
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

# Note: OpenCog imports will be available when opencog-python is installed
# For now, we provide a compatibility layer that can work without OpenCog
try:
    from opencog.atomspace import AtomSpace, TruthValue, types
    from opencog.type_constructors import *
    from opencog.utilities import initialize_opencog
    from opencog.bindlink import execute_atom
    OPENCOG_AVAILABLE = True
except ImportError:
    OPENCOG_AVAILABLE = False
    logging.warning("OpenCog not available, using compatibility mode")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ManuscriptKnowledge:
    """Structured representation of manuscript knowledge"""
    manuscript_id: int
    title: str
    abstract: str
    keywords: List[str]
    authors: List[str]
    submission_date: datetime
    quality_score: float
    novelty_score: float
    methodology_score: float


@dataclass
class ReviewerKnowledge:
    """Structured representation of reviewer knowledge"""
    reviewer_id: int
    name: str
    expertise_areas: List[str]
    publications_count: int
    avg_review_quality: float
    review_history: List[int]  # Manuscript IDs


class OpenCogKnowledgeBase:
    """
    Integrate OpenCog AtomSpace for knowledge representation and reasoning.
    
    The AtomSpace provides a hypergraph database for representing knowledge
    about manuscripts, reviewers, and editorial decisions. This enables
    sophisticated reasoning using OpenCog's PLN (Probabilistic Logic Networks)
    and pattern matching capabilities.
    """
    
    def __init__(self, use_opencog: bool = OPENCOG_AVAILABLE):
        """
        Initialize OpenCog knowledge base.
        
        Args:
            use_opencog: Whether to use actual OpenCog (requires installation)
        """
        self.use_opencog = use_opencog and OPENCOG_AVAILABLE
        
        if self.use_opencog:
            self.atomspace = AtomSpace()
            initialize_opencog(self.atomspace)
            logger.info("OpenCog AtomSpace initialized")
        else:
            # Compatibility mode: use simple dict-based storage
            self.atomspace = {}
            logger.info("Using compatibility mode (OpenCog not available)")
            
        self.manuscript_nodes = {}
        self.reviewer_nodes = {}
        
    # =========================================================================
    # Manuscript Knowledge Management
    # =========================================================================
    
    def add_manuscript_knowledge(self, manuscript: ManuscriptKnowledge):
        """
        Add manuscript to knowledge base.
        
        Creates a network of atoms representing the manuscript and its properties.
        """
        if self.use_opencog:
            self._add_manuscript_opencog(manuscript)
        else:
            self._add_manuscript_compat(manuscript)
            
    def _add_manuscript_opencog(self, manuscript: ManuscriptKnowledge):
        """Add manuscript using OpenCog AtomSpace"""
        # Create manuscript node
        manuscript_node = ConceptNode(f"Manuscript_{manuscript.manuscript_id}")
        self.manuscript_nodes[manuscript.manuscript_id] = manuscript_node
        
        # Add title
        title_node = ConceptNode(manuscript.title)
        EvaluationLink(
            PredicateNode("has_title"),
            ListLink(manuscript_node, title_node),
            tv=TruthValue(1.0, 1.0)
        )
        
        # Add quality scores
        quality_node = NumberNode(str(manuscript.quality_score))
        EvaluationLink(
            PredicateNode("quality_score"),
            ListLink(manuscript_node, quality_node),
            tv=TruthValue(1.0, 1.0)
        )
        
        novelty_node = NumberNode(str(manuscript.novelty_score))
        EvaluationLink(
            PredicateNode("novelty_score"),
            ListLink(manuscript_node, novelty_node),
            tv=TruthValue(1.0, 1.0)
        )
        
        methodology_node = NumberNode(str(manuscript.methodology_score))
        EvaluationLink(
            PredicateNode("methodology_score"),
            ListLink(manuscript_node, methodology_node),
            tv=TruthValue(1.0, 1.0)
        )
        
        # Add keywords
        for keyword in manuscript.keywords:
            keyword_node = ConceptNode(keyword)
            EvaluationLink(
                PredicateNode("has_keyword"),
                ListLink(manuscript_node, keyword_node),
                tv=TruthValue(1.0, 1.0)
            )
            
        # Add authors
        for author in manuscript.authors:
            author_node = ConceptNode(author)
            EvaluationLink(
                PredicateNode("has_author"),
                ListLink(manuscript_node, author_node),
                tv=TruthValue(1.0, 1.0)
            )
            
        logger.info(f"Added manuscript {manuscript.manuscript_id} to AtomSpace")
        
    def _add_manuscript_compat(self, manuscript: ManuscriptKnowledge):
        """Add manuscript using compatibility mode"""
        self.atomspace[f"manuscript_{manuscript.manuscript_id}"] = {
            "type": "manuscript",
            "id": manuscript.manuscript_id,
            "title": manuscript.title,
            "abstract": manuscript.abstract,
            "keywords": manuscript.keywords,
            "authors": manuscript.authors,
            "quality_score": manuscript.quality_score,
            "novelty_score": manuscript.novelty_score,
            "methodology_score": manuscript.methodology_score,
            "submission_date": manuscript.submission_date.isoformat()
        }
        
    # =========================================================================
    # Reviewer Knowledge Management
    # =========================================================================
    
    def add_reviewer_knowledge(self, reviewer: ReviewerKnowledge):
        """Add reviewer to knowledge base"""
        if self.use_opencog:
            self._add_reviewer_opencog(reviewer)
        else:
            self._add_reviewer_compat(reviewer)
            
    def _add_reviewer_opencog(self, reviewer: ReviewerKnowledge):
        """Add reviewer using OpenCog AtomSpace"""
        # Create reviewer node
        reviewer_node = ConceptNode(f"Reviewer_{reviewer.reviewer_id}")
        self.reviewer_nodes[reviewer.reviewer_id] = reviewer_node
        
        # Add expertise areas
        for expertise in reviewer.expertise_areas:
            expertise_node = ConceptNode(expertise)
            EvaluationLink(
                PredicateNode("has_expertise"),
                ListLink(reviewer_node, expertise_node),
                tv=TruthValue(1.0, 1.0)
            )
            
        # Add review quality
        quality_node = NumberNode(str(reviewer.avg_review_quality))
        EvaluationLink(
            PredicateNode("review_quality"),
            ListLink(reviewer_node, quality_node),
            tv=TruthValue(1.0, 1.0)
        )
        
        logger.info(f"Added reviewer {reviewer.reviewer_id} to AtomSpace")
        
    def _add_reviewer_compat(self, reviewer: ReviewerKnowledge):
        """Add reviewer using compatibility mode"""
        self.atomspace[f"reviewer_{reviewer.reviewer_id}"] = {
            "type": "reviewer",
            "id": reviewer.reviewer_id,
            "name": reviewer.name,
            "expertise_areas": reviewer.expertise_areas,
            "publications_count": reviewer.publications_count,
            "avg_review_quality": reviewer.avg_review_quality,
            "review_history": reviewer.review_history
        }
        
    # =========================================================================
    # Query and Reasoning
    # =========================================================================
    
    def query_similar_manuscripts(self, 
                                  query_keywords: List[str],
                                  min_similarity: float = 0.5) -> List[Tuple[int, float]]:
        """
        Query for similar manuscripts using pattern matching.
        
        Returns list of (manuscript_id, similarity_score) tuples.
        """
        if self.use_opencog:
            return self._query_similar_opencog(query_keywords, min_similarity)
        else:
            return self._query_similar_compat(query_keywords, min_similarity)
            
    def _query_similar_opencog(self, query_keywords: List[str], min_similarity: float) -> List[Tuple[int, float]]:
        """Query using OpenCog pattern matching"""
        # Build pattern matching query
        # This would use OpenCog's BindLink for sophisticated pattern matching
        
        results = []
        for manuscript_id, manuscript_node in self.manuscript_nodes.items():
            # Calculate similarity based on keyword overlap
            # In full implementation, would use PLN for probabilistic reasoning
            similarity = self._calculate_similarity(manuscript_id, query_keywords)
            if similarity >= min_similarity:
                results.append((manuscript_id, similarity))
                
        return sorted(results, key=lambda x: x[1], reverse=True)
        
    def _query_similar_compat(self, query_keywords: List[str], min_similarity: float) -> List[Tuple[int, float]]:
        """Query using compatibility mode"""
        results = []
        
        for key, value in self.atomspace.items():
            if value.get("type") == "manuscript":
                manuscript_keywords = value.get("keywords", [])
                overlap = len(set(query_keywords) & set(manuscript_keywords))
                similarity = overlap / max(len(query_keywords), len(manuscript_keywords))
                
                if similarity >= min_similarity:
                    results.append((value["id"], similarity))
                    
        return sorted(results, key=lambda x: x[1], reverse=True)
        
    def match_reviewers_to_manuscript(self, 
                                     manuscript_id: int,
                                     top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Match reviewers to manuscript based on expertise.
        
        Returns list of (reviewer_id, match_score) tuples.
        """
        if self.use_opencog:
            return self._match_reviewers_opencog(manuscript_id, top_k)
        else:
            return self._match_reviewers_compat(manuscript_id, top_k)
            
    def _match_reviewers_opencog(self, manuscript_id: int, top_k: int) -> List[Tuple[int, float]]:
        """Match reviewers using OpenCog reasoning"""
        # Get manuscript keywords
        manuscript_data = self.atomspace.get(f"manuscript_{manuscript_id}")
        if not manuscript_data:
            return []
            
        manuscript_keywords = manuscript_data.get("keywords", [])
        
        # Find reviewers with matching expertise
        matches = []
        for reviewer_id, reviewer_node in self.reviewer_nodes.items():
            match_score = self._calculate_reviewer_match(reviewer_id, manuscript_keywords)
            matches.append((reviewer_id, match_score))
            
        return sorted(matches, key=lambda x: x[1], reverse=True)[:top_k]
        
    def _match_reviewers_compat(self, manuscript_id: int, top_k: int) -> List[Tuple[int, float]]:
        """Match reviewers using compatibility mode"""
        manuscript_data = self.atomspace.get(f"manuscript_{manuscript_id}")
        if not manuscript_data:
            return []
            
        manuscript_keywords = manuscript_data.get("keywords", [])
        
        matches = []
        for key, value in self.atomspace.items():
            if value.get("type") == "reviewer":
                expertise = value.get("expertise_areas", [])
                overlap = len(set(manuscript_keywords) & set(expertise))
                match_score = overlap / max(len(manuscript_keywords), len(expertise))
                
                # Boost by review quality
                match_score *= (0.5 + 0.5 * value.get("avg_review_quality", 0.5))
                
                matches.append((value["id"], match_score))
                
        return sorted(matches, key=lambda x: x[1], reverse=True)[:top_k]
        
    def infer_editorial_decision(self, manuscript_id: int) -> Dict[str, Any]:
        """
        Use OpenCog reasoning to infer editorial decision.
        
        Combines multiple sources of evidence using PLN (Probabilistic Logic Networks).
        """
        manuscript_data = self.atomspace.get(f"manuscript_{manuscript_id}")
        if not manuscript_data:
            return {"action": "reject", "confidence": 0.0, "reasoning": ["Manuscript not found"]}
            
        # Extract scores
        quality_score = manuscript_data.get("quality_score", 0.0)
        novelty_score = manuscript_data.get("novelty_score", 0.0)
        methodology_score = manuscript_data.get("methodology_score", 0.0)
        
        # Simple decision logic (would use PLN in full implementation)
        avg_score = (quality_score + novelty_score + methodology_score) / 3
        
        reasoning = []
        
        if avg_score >= 0.8:
            action = "accept"
            confidence = avg_score
            reasoning.append(f"High average score: {avg_score:.2f}")
        elif avg_score >= 0.6:
            action = "revise"
            confidence = 0.7
            reasoning.append(f"Moderate score: {avg_score:.2f}, revisions recommended")
        else:
            action = "reject"
            confidence = 0.8
            reasoning.append(f"Low average score: {avg_score:.2f}")
            
        # Add specific reasoning
        if quality_score < 0.5:
            reasoning.append("Quality score below threshold")
        if novelty_score < 0.5:
            reasoning.append("Novelty score below threshold")
        if methodology_score < 0.5:
            reasoning.append("Methodology score below threshold")
            
        return {
            "action": action,
            "confidence": confidence,
            "reasoning": reasoning,
            "scores": {
                "quality": quality_score,
                "novelty": novelty_score,
                "methodology": methodology_score,
                "average": avg_score
            }
        }
        
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _calculate_similarity(self, manuscript_id: int, query_keywords: List[str]) -> float:
        """Calculate similarity between manuscript and query"""
        manuscript_data = self.atomspace.get(f"manuscript_{manuscript_id}")
        if not manuscript_data:
            return 0.0
            
        manuscript_keywords = manuscript_data.get("keywords", [])
        overlap = len(set(query_keywords) & set(manuscript_keywords))
        return overlap / max(len(query_keywords), len(manuscript_keywords))
        
    def _calculate_reviewer_match(self, reviewer_id: int, manuscript_keywords: List[str]) -> float:
        """Calculate match score between reviewer and manuscript"""
        reviewer_data = self.atomspace.get(f"reviewer_{reviewer_id}")
        if not reviewer_data:
            return 0.0
            
        expertise = reviewer_data.get("expertise_areas", [])
        overlap = len(set(manuscript_keywords) & set(expertise))
        match_score = overlap / max(len(manuscript_keywords), len(expertise))
        
        # Boost by review quality
        review_quality = reviewer_data.get("avg_review_quality", 0.5)
        return match_score * (0.5 + 0.5 * review_quality)
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        if self.use_opencog:
            return {
                "atomspace_size": len(self.atomspace.get_atoms_by_type(types.Atom)),
                "manuscripts": len(self.manuscript_nodes),
                "reviewers": len(self.reviewer_nodes),
                "opencog_enabled": True
            }
        else:
            manuscript_count = sum(1 for v in self.atomspace.values() if v.get("type") == "manuscript")
            reviewer_count = sum(1 for v in self.atomspace.values() if v.get("type") == "reviewer")
            return {
                "atomspace_size": len(self.atomspace),
                "manuscripts": manuscript_count,
                "reviewers": reviewer_count,
                "opencog_enabled": False
            }


# Example usage
if __name__ == "__main__":
    # Initialize knowledge base
    kb = OpenCogKnowledgeBase()
    
    # Add manuscript
    manuscript = ManuscriptKnowledge(
        manuscript_id=123,
        title="Novel Approach to Cosmetic Formulation",
        abstract="This paper presents a novel approach...",
        keywords=["cosmetics", "formulation", "skin care", "innovation"],
        authors=["Dr. Smith", "Dr. Jones"],
        submission_date=datetime.now(),
        quality_score=0.85,
        novelty_score=0.90,
        methodology_score=0.80
    )
    kb.add_manuscript_knowledge(manuscript)
    
    # Add reviewer
    reviewer = ReviewerKnowledge(
        reviewer_id=456,
        name="Dr. Expert",
        expertise_areas=["cosmetics", "skin care", "dermatology"],
        publications_count=50,
        avg_review_quality=0.92,
        review_history=[100, 101, 102]
    )
    kb.add_reviewer_knowledge(reviewer)
    
    # Query similar manuscripts
    similar = kb.query_similar_manuscripts(["cosmetics", "innovation"])
    print(f"Similar manuscripts: {similar}")
    
    # Match reviewers
    matches = kb.match_reviewers_to_manuscript(123, top_k=3)
    print(f"Matched reviewers: {matches}")
    
    # Infer decision
    decision = kb.infer_editorial_decision(123)
    print(f"Editorial decision: {decision}")
    
    # Statistics
    stats = kb.get_statistics()
    print(f"Knowledge base statistics: {stats}")
