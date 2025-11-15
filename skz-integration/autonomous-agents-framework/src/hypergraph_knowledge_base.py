"""
Hypergraph Knowledge Base Module
Multi-dimensional knowledge representation for autonomous research journal

This module implements a hypergraph-based knowledge representation system
that captures complex multi-way relationships between manuscripts, authors,
reviewers, ingredients, regulations, and decisions.
"""

from typing import Dict, List, Set, Tuple, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import numpy as np
from collections import defaultdict


class NodeType(Enum):
    """Types of nodes in the hypergraph"""
    MANUSCRIPT = "manuscript"
    AUTHOR = "author"
    REVIEWER = "reviewer"
    INGREDIENT = "ingredient"
    REGULATION = "regulation"
    DECISION = "decision"
    CONCEPT = "concept"
    INSTITUTION = "institution"


class HyperedgeType(Enum):
    """Types of hyperedges (multi-way relationships)"""
    SUBMISSION_CONTEXT = "submission_context"
    REVIEW_ASSIGNMENT = "review_assignment"
    QUALITY_ASSESSMENT = "quality_assessment"
    PUBLICATION_NETWORK = "publication_network"
    REGULATORY_COMPLIANCE = "regulatory_compliance"
    INGREDIENT_INTERACTION = "ingredient_interaction"
    AUTHORSHIP = "authorship"
    EXPERTISE_MATCH = "expertise_match"


@dataclass
class Node:
    """A node in the hypergraph"""
    id: str
    type: NodeType
    attributes: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary"""
        return {
            'id': self.id,
            'type': self.type.value,
            'attributes': self.attributes,
            'embedding': self.embedding.tolist() if self.embedding is not None else None,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }


@dataclass
class Hyperedge:
    """A hyperedge connecting multiple nodes"""
    id: str
    type: HyperedgeType
    nodes: Set[str]  # Set of node IDs
    weight: float = 1.0
    attributes: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert hyperedge to dictionary"""
        return {
            'id': self.id,
            'type': self.type.value,
            'nodes': list(self.nodes),
            'weight': self.weight,
            'attributes': self.attributes,
            'created_at': self.created_at.isoformat()
        }


class HypergraphKnowledgeBase:
    """
    Hypergraph-based knowledge representation system
    
    Supports:
    - Multi-way relationships between entities
    - Semantic embeddings for similarity search
    - Dynamic schema evolution
    - Complex query patterns
    """
    
    def __init__(self):
        """Initialize the hypergraph knowledge base"""
        self.nodes: Dict[str, Node] = {}
        self.hyperedges: Dict[str, Hyperedge] = {}
        self.node_to_edges: Dict[str, Set[str]] = defaultdict(set)
        self.type_to_nodes: Dict[NodeType, Set[str]] = defaultdict(set)
        self.schema_version = "1.0"
        self.evolution_history = []
        
    def add_node(
        self, 
        node_id: str, 
        node_type: NodeType, 
        attributes: Dict[str, Any],
        embedding: Optional[np.ndarray] = None
    ) -> Node:
        """
        Add a node to the hypergraph
        
        Args:
            node_id: Unique identifier for the node
            node_type: Type of the node
            attributes: Node attributes
            embedding: Optional semantic embedding vector
            
        Returns:
            The created Node object
        """
        if node_id in self.nodes:
            # Update existing node
            node = self.nodes[node_id]
            node.attributes.update(attributes)
            if embedding is not None:
                node.embedding = embedding
            node.updated_at = datetime.now()
        else:
            # Create new node
            node = Node(
                id=node_id,
                type=node_type,
                attributes=attributes,
                embedding=embedding
            )
            self.nodes[node_id] = node
            self.type_to_nodes[node_type].add(node_id)
        
        return node
    
    def add_hyperedge(
        self,
        edge_id: str,
        edge_type: HyperedgeType,
        node_ids: List[str],
        weight: float = 1.0,
        attributes: Optional[Dict[str, Any]] = None
    ) -> Hyperedge:
        """
        Add a hyperedge connecting multiple nodes
        
        Args:
            edge_id: Unique identifier for the hyperedge
            edge_type: Type of the hyperedge
            node_ids: List of node IDs to connect
            weight: Edge weight (default 1.0)
            attributes: Optional edge attributes
            
        Returns:
            The created Hyperedge object
        """
        # Validate that all nodes exist
        for node_id in node_ids:
            if node_id not in self.nodes:
                raise ValueError(f"Node {node_id} does not exist")
        
        # Create hyperedge
        hyperedge = Hyperedge(
            id=edge_id,
            type=edge_type,
            nodes=set(node_ids),
            weight=weight,
            attributes=attributes or {}
        )
        
        self.hyperedges[edge_id] = hyperedge
        
        # Update node-to-edge mapping
        for node_id in node_ids:
            self.node_to_edges[node_id].add(edge_id)
        
        return hyperedge
    
    def get_node(self, node_id: str) -> Optional[Node]:
        """Get a node by ID"""
        return self.nodes.get(node_id)
    
    def get_hyperedge(self, edge_id: str) -> Optional[Hyperedge]:
        """Get a hyperedge by ID"""
        return self.hyperedges.get(edge_id)
    
    def get_nodes_by_type(self, node_type: NodeType) -> List[Node]:
        """Get all nodes of a specific type"""
        node_ids = self.type_to_nodes.get(node_type, set())
        return [self.nodes[nid] for nid in node_ids]
    
    def get_connected_nodes(
        self, 
        node_id: str, 
        edge_type: Optional[HyperedgeType] = None
    ) -> List[Node]:
        """
        Get all nodes connected to a given node
        
        Args:
            node_id: The source node ID
            edge_type: Optional filter by edge type
            
        Returns:
            List of connected nodes
        """
        if node_id not in self.nodes:
            return []
        
        connected_node_ids = set()
        edge_ids = self.node_to_edges[node_id]
        
        for edge_id in edge_ids:
            edge = self.hyperedges[edge_id]
            
            # Filter by edge type if specified
            if edge_type and edge.type != edge_type:
                continue
            
            # Add all nodes in the hyperedge except the source
            connected_node_ids.update(edge.nodes - {node_id})
        
        return [self.nodes[nid] for nid in connected_node_ids]
    
    def find_similar_nodes(
        self, 
        node_id: str, 
        top_k: int = 10,
        node_type: Optional[NodeType] = None
    ) -> List[Tuple[Node, float]]:
        """
        Find similar nodes based on embedding similarity
        
        Args:
            node_id: The query node ID
            top_k: Number of similar nodes to return
            node_type: Optional filter by node type
            
        Returns:
            List of (node, similarity_score) tuples
        """
        query_node = self.nodes.get(node_id)
        if not query_node or query_node.embedding is None:
            return []
        
        similarities = []
        
        for nid, node in self.nodes.items():
            if nid == node_id:
                continue
            
            # Filter by type if specified
            if node_type and node.type != node_type:
                continue
            
            if node.embedding is not None:
                # Cosine similarity
                similarity = self._cosine_similarity(
                    query_node.embedding, 
                    node.embedding
                )
                similarities.append((node, similarity))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def query_hypergraph(
        self,
        node_types: List[NodeType],
        edge_type: HyperedgeType,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Query the hypergraph for specific patterns
        
        Args:
            node_types: Types of nodes to include in the pattern
            edge_type: Type of hyperedge connecting the nodes
            filters: Optional attribute filters
            
        Returns:
            List of matching subgraphs
        """
        results = []
        
        # Find all hyperedges of the specified type
        for edge_id, edge in self.hyperedges.items():
            if edge.type != edge_type:
                continue
            
            # Get nodes in this hyperedge
            edge_nodes = [self.nodes[nid] for nid in edge.nodes]
            
            # Check if node types match
            edge_node_types = {node.type for node in edge_nodes}
            if not set(node_types).issubset(edge_node_types):
                continue
            
            # Apply filters if specified
            if filters and not self._match_filters(edge, edge_nodes, filters):
                continue
            
            # Add to results
            results.append({
                'edge': edge.to_dict(),
                'nodes': [node.to_dict() for node in edge_nodes]
            })
        
        return results
    
    def _match_filters(
        self, 
        edge: Hyperedge, 
        nodes: List[Node], 
        filters: Dict[str, Any]
    ) -> bool:
        """Check if edge and nodes match the specified filters"""
        # Simple attribute matching (can be extended)
        for key, value in filters.items():
            if key in edge.attributes and edge.attributes[key] != value:
                return False
            
            # Check node attributes
            match_found = False
            for node in nodes:
                if key in node.attributes and node.attributes[key] == value:
                    match_found = True
                    break
            
            if not match_found:
                return False
        
        return True
    
    def add_manuscript_submission(
        self,
        manuscript_id: str,
        authors: List[str],
        ingredients: List[str],
        regulations: List[str],
        manuscript_data: Dict[str, Any]
    ) -> str:
        """
        Add a complete manuscript submission context to the hypergraph
        
        Args:
            manuscript_id: Manuscript identifier
            authors: List of author IDs
            ingredients: List of ingredient IDs
            regulations: List of regulation IDs
            manuscript_data: Manuscript attributes
            
        Returns:
            The hyperedge ID representing the submission context
        """
        # Add manuscript node
        self.add_node(
            manuscript_id,
            NodeType.MANUSCRIPT,
            manuscript_data
        )
        
        # Create submission context hyperedge
        edge_id = f"submission_{manuscript_id}"
        all_nodes = [manuscript_id] + authors + ingredients + regulations
        
        self.add_hyperedge(
            edge_id,
            HyperedgeType.SUBMISSION_CONTEXT,
            all_nodes,
            attributes={'submission_date': datetime.now().isoformat()}
        )
        
        return edge_id
    
    def add_review_assignment(
        self,
        manuscript_id: str,
        reviewer_id: str,
        decision_id: str,
        assignment_data: Dict[str, Any]
    ) -> str:
        """
        Add a review assignment to the hypergraph
        
        Args:
            manuscript_id: Manuscript identifier
            reviewer_id: Reviewer identifier
            decision_id: Decision identifier
            assignment_data: Assignment attributes
            
        Returns:
            The hyperedge ID representing the review assignment
        """
        edge_id = f"review_{manuscript_id}_{reviewer_id}"
        
        self.add_hyperedge(
            edge_id,
            HyperedgeType.REVIEW_ASSIGNMENT,
            [manuscript_id, reviewer_id, decision_id],
            attributes=assignment_data
        )
        
        return edge_id
    
    def evolve_schema(self, evolution_description: str):
        """
        Record schema evolution
        
        Args:
            evolution_description: Description of the schema change
        """
        self.evolution_history.append({
            'timestamp': datetime.now().isoformat(),
            'description': evolution_description,
            'node_count': len(self.nodes),
            'edge_count': len(self.hyperedges)
        })
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the hypergraph"""
        node_type_counts = {
            nt.value: len(self.type_to_nodes[nt])
            for nt in NodeType
        }
        
        edge_type_counts = defaultdict(int)
        for edge in self.hyperedges.values():
            edge_type_counts[edge.type.value] += 1
        
        return {
            'total_nodes': len(self.nodes),
            'total_hyperedges': len(self.hyperedges),
            'node_type_counts': node_type_counts,
            'edge_type_counts': dict(edge_type_counts),
            'schema_version': self.schema_version,
            'evolution_count': len(self.evolution_history)
        }
    
    def export_to_json(self, filepath: str):
        """Export hypergraph to JSON file"""
        data = {
            'schema_version': self.schema_version,
            'nodes': [node.to_dict() for node in self.nodes.values()],
            'hyperedges': [edge.to_dict() for edge in self.hyperedges.values()],
            'evolution_history': self.evolution_history,
            'statistics': self.get_statistics(),
            'exported_at': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def import_from_json(self, filepath: str):
        """Import hypergraph from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Clear existing data
        self.nodes.clear()
        self.hyperedges.clear()
        self.node_to_edges.clear()
        self.type_to_nodes.clear()
        
        # Import nodes
        for node_data in data['nodes']:
            node = Node(
                id=node_data['id'],
                type=NodeType(node_data['type']),
                attributes=node_data['attributes'],
                embedding=np.array(node_data['embedding']) if node_data['embedding'] else None,
                created_at=datetime.fromisoformat(node_data['created_at']),
                updated_at=datetime.fromisoformat(node_data['updated_at'])
            )
            self.nodes[node.id] = node
            self.type_to_nodes[node.type].add(node.id)
        
        # Import hyperedges
        for edge_data in data['hyperedges']:
            edge = Hyperedge(
                id=edge_data['id'],
                type=HyperedgeType(edge_data['type']),
                nodes=set(edge_data['nodes']),
                weight=edge_data['weight'],
                attributes=edge_data['attributes'],
                created_at=datetime.fromisoformat(edge_data['created_at'])
            )
            self.hyperedges[edge.id] = edge
            
            for node_id in edge.nodes:
                self.node_to_edges[node_id].add(edge.id)
        
        self.schema_version = data['schema_version']
        self.evolution_history = data['evolution_history']


# Example usage
if __name__ == "__main__":
    # Initialize knowledge base
    kb = HypergraphKnowledgeBase()
    
    # Add manuscript
    kb.add_node(
        "MS-2025-001",
        NodeType.MANUSCRIPT,
        {
            'title': 'Novel Peptide for Skin Rejuvenation',
            'abstract': 'This study investigates...',
            'keywords': ['peptide', 'anti-aging', 'collagen']
        },
        embedding=np.random.randn(768)
    )
    
    # Add authors
    kb.add_node("AU-001", NodeType.AUTHOR, {'name': 'Dr. Smith', 'h_index': 15})
    kb.add_node("AU-002", NodeType.AUTHOR, {'name': 'Dr. Jones', 'h_index': 12})
    
    # Add ingredients
    kb.add_node("ING-001", NodeType.INGREDIENT, {
        'inci_name': 'Palmitoyl Pentapeptide-4',
        'cas_number': '214047-00-4'
    })
    
    # Create submission context
    kb.add_manuscript_submission(
        "MS-2025-001",
        ["AU-001", "AU-002"],
        ["ING-001"],
        [],
        {'submission_date': '2025-01-15'}
    )
    
    # Print statistics
    stats = kb.get_statistics()
    print(f"Hypergraph statistics: {json.dumps(stats, indent=2)}")
    
    # Query connected nodes
    connected = kb.get_connected_nodes("MS-2025-001")
    print(f"Connected to MS-2025-001: {[n.id for n in connected]}")
