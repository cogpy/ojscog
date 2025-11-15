-- OJSCog Neon Database Schema
-- Purpose: Hypergraph Dynamics and Knowledge Graph
-- Version: 1.0
-- Date: 2025-11-15

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";  -- pgvector for embeddings
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- Trigram similarity

-- ============================================================================
-- HYPERGRAPH NODES TABLE
-- ============================================================================
-- Represents entities in the hypergraph (manuscripts, reviewers, concepts, etc.)

CREATE TABLE IF NOT EXISTS hypergraph_nodes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    node_id VARCHAR(100) UNIQUE NOT NULL,
    node_type VARCHAR(50) NOT NULL CHECK (node_type IN (
        'manuscript', 'reviewer', 'author', 'concept', 'keyword',
        'institution', 'journal', 'citation', 'agent', 'decision'
    )),
    entity_id VARCHAR(100) NOT NULL,
    entity_name TEXT NOT NULL,
    properties JSONB DEFAULT '{}',
    embedding VECTOR(768),  -- 768-dimensional embedding for semantic similarity
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_hypergraph_nodes_node_id ON hypergraph_nodes(node_id);
CREATE INDEX idx_hypergraph_nodes_type ON hypergraph_nodes(node_type);
CREATE INDEX idx_hypergraph_nodes_entity ON hypergraph_nodes(entity_id);
CREATE INDEX idx_hypergraph_nodes_name_trgm ON hypergraph_nodes USING gin (entity_name gin_trgm_ops);
CREATE INDEX idx_hypergraph_nodes_embedding ON hypergraph_nodes USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- ============================================================================
-- HYPERGRAPH EDGES TABLE
-- ============================================================================
-- Represents binary relationships between nodes

CREATE TABLE IF NOT EXISTS hypergraph_edges (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    edge_id VARCHAR(100) UNIQUE NOT NULL,
    edge_type VARCHAR(50) NOT NULL CHECK (edge_type IN (
        'authored_by', 'reviewed_by', 'cites', 'cited_by', 'similar_to',
        'assigned_to', 'decided_by', 'published_in', 'affiliated_with',
        'expertise_in', 'collaborated_with', 'supersedes', 'related_to'
    )),
    source_node_id VARCHAR(100) NOT NULL,
    target_node_id VARCHAR(100) NOT NULL,
    weight DECIMAL(5,4) DEFAULT 1.0 CHECK (weight BETWEEN 0 AND 1),
    properties JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    FOREIGN KEY (source_node_id) REFERENCES hypergraph_nodes(node_id) ON DELETE CASCADE,
    FOREIGN KEY (target_node_id) REFERENCES hypergraph_nodes(node_id) ON DELETE CASCADE
);

CREATE INDEX idx_hypergraph_edges_edge_id ON hypergraph_edges(edge_id);
CREATE INDEX idx_hypergraph_edges_type ON hypergraph_edges(edge_type);
CREATE INDEX idx_hypergraph_edges_source ON hypergraph_edges(source_node_id);
CREATE INDEX idx_hypergraph_edges_target ON hypergraph_edges(target_node_id);
CREATE INDEX idx_hypergraph_edges_weight ON hypergraph_edges(weight DESC);

-- ============================================================================
-- HYPERGRAPH HYPEREDGES TABLE
-- ============================================================================
-- Represents multi-way relationships (connecting 3+ nodes)

CREATE TABLE IF NOT EXISTS hypergraph_hyperedges (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    hyperedge_id VARCHAR(100) UNIQUE NOT NULL,
    hyperedge_type VARCHAR(50) NOT NULL CHECK (hyperedge_type IN (
        'co_authorship', 'review_panel', 'editorial_board', 'research_collaboration',
        'citation_cluster', 'topic_cluster', 'decision_committee', 'conflict_group'
    )),
    node_ids JSONB NOT NULL,  -- Array of node_ids participating in this hyperedge
    properties JSONB DEFAULT '{}',
    weight DECIMAL(5,4) DEFAULT 1.0 CHECK (weight BETWEEN 0 AND 1),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_hypergraph_hyperedges_hyperedge_id ON hypergraph_hyperedges(hyperedge_id);
CREATE INDEX idx_hypergraph_hyperedges_type ON hypergraph_hyperedges(hyperedge_type);
CREATE INDEX idx_hypergraph_hyperedges_node_ids ON hypergraph_hyperedges USING gin (node_ids);

-- ============================================================================
-- KNOWLEDGE GRAPH TABLE
-- ============================================================================
-- Structured knowledge representation for domain concepts

CREATE TABLE IF NOT EXISTS knowledge_graph (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    entity_id VARCHAR(100) UNIQUE NOT NULL,
    entity_type VARCHAR(50) NOT NULL CHECK (entity_type IN (
        'ingredient', 'formulation', 'regulation', 'safety_standard',
        'research_method', 'clinical_trial', 'patent', 'guideline',
        'cosmetic_claim', 'adverse_effect', 'efficacy_measure'
    )),
    entity_name TEXT NOT NULL,
    description TEXT,
    attributes JSONB DEFAULT '{}',
    related_entities JSONB DEFAULT '[]',
    confidence_score DECIMAL(5,4) CHECK (confidence_score BETWEEN 0 AND 1),
    source VARCHAR(100),
    embedding VECTOR(768),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_knowledge_graph_entity_id ON knowledge_graph(entity_id);
CREATE INDEX idx_knowledge_graph_type ON knowledge_graph(entity_type);
CREATE INDEX idx_knowledge_graph_name ON knowledge_graph(entity_name);
CREATE INDEX idx_knowledge_graph_name_trgm ON knowledge_graph USING gin (entity_name gin_trgm_ops);
CREATE INDEX idx_knowledge_graph_confidence ON knowledge_graph(confidence_score DESC);
CREATE INDEX idx_knowledge_graph_embedding ON knowledge_graph USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- ============================================================================
-- SEMANTIC SIMILARITY CACHE TABLE
-- ============================================================================
-- Caches computed semantic similarities for performance

CREATE TABLE IF NOT EXISTS semantic_similarity_cache (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    entity1_id VARCHAR(100) NOT NULL,
    entity2_id VARCHAR(100) NOT NULL,
    similarity_score DECIMAL(5,4) NOT NULL CHECK (similarity_score BETWEEN 0 AND 1),
    similarity_type VARCHAR(50) NOT NULL CHECK (similarity_type IN (
        'cosine', 'euclidean', 'jaccard', 'semantic', 'structural'
    )),
    computed_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(entity1_id, entity2_id, similarity_type)
);

CREATE INDEX idx_semantic_cache_entity1 ON semantic_similarity_cache(entity1_id);
CREATE INDEX idx_semantic_cache_entity2 ON semantic_similarity_cache(entity2_id);
CREATE INDEX idx_semantic_cache_score ON semantic_similarity_cache(similarity_score DESC);

-- ============================================================================
-- GRAPH ANALYTICS RESULTS TABLE
-- ============================================================================
-- Stores results of graph analytics algorithms

CREATE TABLE IF NOT EXISTS graph_analytics_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    analysis_id VARCHAR(100) UNIQUE NOT NULL,
    analysis_type VARCHAR(50) NOT NULL CHECK (analysis_type IN (
        'pagerank', 'centrality', 'community_detection', 'path_finding',
        'clustering', 'influence_propagation', 'anomaly_detection'
    )),
    target_node_id VARCHAR(100),
    results JSONB NOT NULL,
    metadata JSONB DEFAULT '{}',
    computed_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_graph_analytics_analysis_id ON graph_analytics_results(analysis_id);
CREATE INDEX idx_graph_analytics_type ON graph_analytics_results(analysis_type);
CREATE INDEX idx_graph_analytics_target ON graph_analytics_results(target_node_id);
CREATE INDEX idx_graph_analytics_computed ON graph_analytics_results(computed_at DESC);

-- ============================================================================
-- TEMPORAL GRAPH SNAPSHOTS TABLE
-- ============================================================================
-- Tracks evolution of the hypergraph over time

CREATE TABLE IF NOT EXISTS temporal_graph_snapshots (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    snapshot_id VARCHAR(100) UNIQUE NOT NULL,
    snapshot_timestamp TIMESTAMP NOT NULL,
    node_count INTEGER NOT NULL,
    edge_count INTEGER NOT NULL,
    hyperedge_count INTEGER NOT NULL,
    graph_statistics JSONB DEFAULT '{}',
    snapshot_data JSONB,  -- Optional: store full snapshot for time-travel queries
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_temporal_snapshots_snapshot_id ON temporal_graph_snapshots(snapshot_id);
CREATE INDEX idx_temporal_snapshots_timestamp ON temporal_graph_snapshots(snapshot_timestamp DESC);

-- ============================================================================
-- REVIEWER EXPERTISE GRAPH TABLE
-- ============================================================================
-- Maps reviewers to their areas of expertise using graph structure

CREATE TABLE IF NOT EXISTS reviewer_expertise_graph (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    reviewer_node_id VARCHAR(100) NOT NULL,
    expertise_node_id VARCHAR(100) NOT NULL,
    proficiency_level DECIMAL(5,4) CHECK (proficiency_level BETWEEN 0 AND 1),
    evidence_count INTEGER DEFAULT 0,
    last_updated TIMESTAMP DEFAULT NOW(),
    FOREIGN KEY (reviewer_node_id) REFERENCES hypergraph_nodes(node_id) ON DELETE CASCADE,
    FOREIGN KEY (expertise_node_id) REFERENCES hypergraph_nodes(node_id) ON DELETE CASCADE,
    UNIQUE(reviewer_node_id, expertise_node_id)
);

CREATE INDEX idx_reviewer_expertise_reviewer ON reviewer_expertise_graph(reviewer_node_id);
CREATE INDEX idx_reviewer_expertise_expertise ON reviewer_expertise_graph(expertise_node_id);
CREATE INDEX idx_reviewer_expertise_proficiency ON reviewer_expertise_graph(proficiency_level DESC);

-- ============================================================================
-- CITATION NETWORK TABLE
-- ============================================================================
-- Specialized table for citation network analysis

CREATE TABLE IF NOT EXISTS citation_network (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    citing_manuscript_id VARCHAR(100) NOT NULL,
    cited_manuscript_id VARCHAR(100) NOT NULL,
    citation_context TEXT,
    citation_type VARCHAR(50) CHECK (citation_type IN (
        'direct', 'indirect', 'supporting', 'contrasting', 'methodological'
    )),
    citation_weight DECIMAL(5,4) DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(citing_manuscript_id, cited_manuscript_id)
);

CREATE INDEX idx_citation_network_citing ON citation_network(citing_manuscript_id);
CREATE INDEX idx_citation_network_cited ON citation_network(cited_manuscript_id);
CREATE INDEX idx_citation_network_type ON citation_network(citation_type);

-- ============================================================================
-- VIEWS FOR COMMON GRAPH QUERIES
-- ============================================================================

-- View: Node degree distribution
CREATE OR REPLACE VIEW node_degree_distribution AS
SELECT 
    n.node_type,
    n.node_id,
    n.entity_name,
    COUNT(DISTINCT e1.id) as out_degree,
    COUNT(DISTINCT e2.id) as in_degree,
    COUNT(DISTINCT e1.id) + COUNT(DISTINCT e2.id) as total_degree
FROM hypergraph_nodes n
LEFT JOIN hypergraph_edges e1 ON n.node_id = e1.source_node_id
LEFT JOIN hypergraph_edges e2 ON n.node_id = e2.target_node_id
GROUP BY n.node_type, n.node_id, n.entity_name;

-- View: Most connected nodes (hubs)
CREATE OR REPLACE VIEW graph_hubs AS
SELECT 
    node_type,
    node_id,
    entity_name,
    total_degree
FROM node_degree_distribution
WHERE total_degree > 5
ORDER BY total_degree DESC;

-- View: Reviewer-manuscript matching scores
CREATE OR REPLACE VIEW reviewer_manuscript_matches AS
SELECT 
    r.node_id as reviewer_id,
    r.entity_name as reviewer_name,
    m.node_id as manuscript_id,
    m.entity_name as manuscript_title,
    (r.embedding <=> m.embedding) as semantic_distance,
    1 - (r.embedding <=> m.embedding) as match_score
FROM hypergraph_nodes r
CROSS JOIN hypergraph_nodes m
WHERE r.node_type = 'reviewer'
  AND m.node_type = 'manuscript'
  AND r.embedding IS NOT NULL
  AND m.embedding IS NOT NULL;

-- ============================================================================
-- FUNCTIONS FOR HYPERGRAPH OPERATIONS
-- ============================================================================

-- Function: Find similar nodes using vector similarity
CREATE OR REPLACE FUNCTION find_similar_nodes(
    p_node_id VARCHAR(100),
    p_limit INTEGER DEFAULT 10,
    p_threshold DECIMAL DEFAULT 0.7
) RETURNS TABLE (
    similar_node_id VARCHAR(100),
    similarity_score DECIMAL,
    node_type VARCHAR(50),
    entity_name TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        n2.node_id,
        (1 - (n1.embedding <=> n2.embedding))::DECIMAL as similarity,
        n2.node_type,
        n2.entity_name
    FROM hypergraph_nodes n1
    CROSS JOIN hypergraph_nodes n2
    WHERE n1.node_id = p_node_id
      AND n2.node_id != p_node_id
      AND n1.embedding IS NOT NULL
      AND n2.embedding IS NOT NULL
      AND (1 - (n1.embedding <=> n2.embedding)) >= p_threshold
    ORDER BY similarity DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- Function: Get node neighborhood (k-hop neighbors)
CREATE OR REPLACE FUNCTION get_node_neighborhood(
    p_node_id VARCHAR(100),
    p_hops INTEGER DEFAULT 1
) RETURNS TABLE (
    neighbor_node_id VARCHAR(100),
    distance INTEGER,
    path_length INTEGER
) AS $$
BEGIN
    RETURN QUERY
    WITH RECURSIVE neighborhood AS (
        -- Base case: direct neighbors
        SELECT 
            e.target_node_id as neighbor_id,
            1 as dist,
            1 as path_len
        FROM hypergraph_edges e
        WHERE e.source_node_id = p_node_id
        
        UNION
        
        -- Recursive case: neighbors of neighbors
        SELECT 
            e.target_node_id,
            n.dist + 1,
            n.path_len + 1
        FROM neighborhood n
        JOIN hypergraph_edges e ON n.neighbor_id = e.source_node_id
        WHERE n.dist < p_hops
    )
    SELECT DISTINCT 
        neighbor_id,
        MIN(dist) as distance,
        MIN(path_len) as path_length
    FROM neighborhood
    GROUP BY neighbor_id
    ORDER BY distance, path_length;
END;
$$ LANGUAGE plpgsql;

-- Function: Compute PageRank for nodes
CREATE OR REPLACE FUNCTION compute_pagerank(
    p_damping_factor DECIMAL DEFAULT 0.85,
    p_iterations INTEGER DEFAULT 20
) RETURNS TABLE (
    node_id VARCHAR(100),
    pagerank_score DECIMAL
) AS $$
DECLARE
    v_node_count INTEGER;
    v_initial_rank DECIMAL;
BEGIN
    -- Get total node count
    SELECT COUNT(*) INTO v_node_count FROM hypergraph_nodes;
    v_initial_rank := 1.0 / v_node_count;
    
    -- Create temporary table for PageRank computation
    CREATE TEMP TABLE IF NOT EXISTS pagerank_temp (
        node_id VARCHAR(100) PRIMARY KEY,
        current_rank DECIMAL,
        next_rank DECIMAL
    ) ON COMMIT DROP;
    
    -- Initialize ranks
    INSERT INTO pagerank_temp (node_id, current_rank, next_rank)
    SELECT n.node_id, v_initial_rank, 0.0
    FROM hypergraph_nodes n;
    
    -- Iterative PageRank computation
    FOR i IN 1..p_iterations LOOP
        -- Compute next rank
        UPDATE pagerank_temp pt
        SET next_rank = (1 - p_damping_factor) / v_node_count + 
                       p_damping_factor * COALESCE(
                           (SELECT SUM(pt2.current_rank / NULLIF(out_deg.degree, 0))
                            FROM hypergraph_edges e
                            JOIN pagerank_temp pt2 ON e.source_node_id = pt2.node_id
                            LEFT JOIN (
                                SELECT source_node_id, COUNT(*) as degree
                                FROM hypergraph_edges
                                GROUP BY source_node_id
                            ) out_deg ON e.source_node_id = out_deg.source_node_id
                            WHERE e.target_node_id = pt.node_id),
                           0
                       );
        
        -- Update current rank
        UPDATE pagerank_temp
        SET current_rank = next_rank;
    END LOOP;
    
    -- Return results
    RETURN QUERY
    SELECT pt.node_id, pt.current_rank
    FROM pagerank_temp pt
    ORDER BY pt.current_rank DESC;
END;
$$ LANGUAGE plpgsql;

-- Function: Detect communities using label propagation
CREATE OR REPLACE FUNCTION detect_communities(
    p_iterations INTEGER DEFAULT 10
) RETURNS TABLE (
    node_id VARCHAR(100),
    community_id INTEGER
) AS $$
BEGIN
    -- Create temporary table for community labels
    CREATE TEMP TABLE IF NOT EXISTS community_labels (
        node_id VARCHAR(100) PRIMARY KEY,
        label INTEGER
    ) ON COMMIT DROP;
    
    -- Initialize: each node is its own community
    INSERT INTO community_labels (node_id, label)
    SELECT node_id, ROW_NUMBER() OVER ()::INTEGER
    FROM hypergraph_nodes;
    
    -- Label propagation iterations
    FOR i IN 1..p_iterations LOOP
        UPDATE community_labels cl
        SET label = (
            SELECT cl2.label
            FROM hypergraph_edges e
            JOIN community_labels cl2 ON e.target_node_id = cl2.node_id
            WHERE e.source_node_id = cl.node_id
            GROUP BY cl2.label
            ORDER BY COUNT(*) DESC, cl2.label
            LIMIT 1
        )
        WHERE EXISTS (
            SELECT 1 FROM hypergraph_edges e WHERE e.source_node_id = cl.node_id
        );
    END LOOP;
    
    -- Return results
    RETURN QUERY
    SELECT cl.node_id, cl.label
    FROM community_labels cl
    ORDER BY cl.label, cl.node_id;
END;
$$ LANGUAGE plpgsql;

-- Function: Update node embedding
CREATE OR REPLACE FUNCTION update_node_embedding(
    p_node_id VARCHAR(100),
    p_embedding VECTOR(768)
) RETURNS VOID AS $$
BEGIN
    UPDATE hypergraph_nodes
    SET embedding = p_embedding,
        updated_at = NOW()
    WHERE node_id = p_node_id;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- TRIGGERS
-- ============================================================================

-- Trigger: Update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_hypergraph_nodes_updated_at BEFORE UPDATE ON hypergraph_nodes
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_hypergraph_edges_updated_at BEFORE UPDATE ON hypergraph_edges
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_hypergraph_hyperedges_updated_at BEFORE UPDATE ON hypergraph_hyperedges
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_knowledge_graph_updated_at BEFORE UPDATE ON knowledge_graph
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- SCHEMA VERSION TRACKING
-- ============================================================================

CREATE TABLE IF NOT EXISTS schema_version (
    version VARCHAR(20) PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT NOW(),
    description TEXT
);

INSERT INTO schema_version (version, description) VALUES
    ('1.0.0', 'Initial Neon hypergraph schema with vector embeddings and graph analytics')
ON CONFLICT (version) DO NOTHING;

-- End of Neon hypergraph schema
