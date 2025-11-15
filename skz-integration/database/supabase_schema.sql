-- OJSCog Supabase Database Schema
-- Purpose: Agent State Management and Workflow Tracking
-- Version: 1.0
-- Date: 2025-11-15

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Enable timestamp functions
CREATE EXTENSION IF NOT EXISTS "btree_gist";

-- ============================================================================
-- AGENTS STATE TABLE
-- ============================================================================
-- Tracks the current state of all 7 autonomous agents

CREATE TABLE IF NOT EXISTS agents_state (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id VARCHAR(50) UNIQUE NOT NULL,
    agent_name VARCHAR(100) NOT NULL,
    agent_type VARCHAR(50) NOT NULL CHECK (agent_type IN (
        'research_discovery',
        'submission_assistant',
        'editorial_orchestration',
        'review_coordination',
        'content_quality',
        'publishing_production',
        'analytics_monitoring'
    )),
    current_phase VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'idle' CHECK (status IN (
        'idle', 'active', 'busy', 'error', 'maintenance'
    )),
    context JSONB DEFAULT '{}',
    memory JSONB DEFAULT '{}',
    performance_metrics JSONB DEFAULT '{}',
    last_active TIMESTAMP DEFAULT NOW(),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_agents_state_agent_id ON agents_state(agent_id);
CREATE INDEX idx_agents_state_status ON agents_state(status);
CREATE INDEX idx_agents_state_type ON agents_state(agent_type);
CREATE INDEX idx_agents_state_last_active ON agents_state(last_active DESC);

-- Trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_agents_state_updated_at BEFORE UPDATE ON agents_state
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- MANUSCRIPT WORKFLOWS TABLE
-- ============================================================================
-- Tracks manuscript progression through the autonomous publishing workflow

CREATE TABLE IF NOT EXISTS manuscript_workflows (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    manuscript_id VARCHAR(100) UNIQUE NOT NULL,
    ojs_submission_id INTEGER,
    title TEXT,
    authors JSONB DEFAULT '[]',
    current_stage VARCHAR(50) NOT NULL CHECK (current_stage IN (
        'initial', 'submission', 'quality-assessment', 'review-assignment',
        'under-review', 'editorial-decision', 'revision-stage', 'production',
        'publication-ready', 'published', 'desk-rejection', 'rejection-notification'
    )),
    workflow_state VARCHAR(50) NOT NULL,
    assigned_agents JSONB DEFAULT '[]',
    stage_history JSONB DEFAULT '[]',
    decision_trail JSONB DEFAULT '[]',
    quality_scores JSONB DEFAULT '{}',
    reviewer_assignments JSONB DEFAULT '[]',
    timeline JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_manuscript_workflows_manuscript_id ON manuscript_workflows(manuscript_id);
CREATE INDEX idx_manuscript_workflows_current_stage ON manuscript_workflows(current_stage);
CREATE INDEX idx_manuscript_workflows_workflow_state ON manuscript_workflows(workflow_state);
CREATE INDEX idx_manuscript_workflows_ojs_id ON manuscript_workflows(ojs_submission_id);
CREATE INDEX idx_manuscript_workflows_created ON manuscript_workflows(created_at DESC);

CREATE TRIGGER update_manuscript_workflows_updated_at BEFORE UPDATE ON manuscript_workflows
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- COGNITIVE LOOP EXECUTIONS TABLE
-- ============================================================================
-- Records each execution of the 12-step cognitive loop

CREATE TABLE IF NOT EXISTS cognitive_loop_executions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    execution_id VARCHAR(100) UNIQUE NOT NULL,
    manuscript_id VARCHAR(100),
    loop_iteration INTEGER NOT NULL,
    phase VARCHAR(20) NOT NULL CHECK (phase IN (
        'expressive', 'reflective', 'anticipatory'
    )),
    step_number INTEGER NOT NULL CHECK (step_number BETWEEN 1 AND 12),
    step_name VARCHAR(100) NOT NULL,
    input_data JSONB DEFAULT '{}',
    output_data JSONB DEFAULT '{}',
    relevance_score DECIMAL(5,4) CHECK (relevance_score BETWEEN 0 AND 1),
    confidence DECIMAL(5,4) CHECK (confidence BETWEEN 0 AND 1),
    execution_time_ms INTEGER,
    status VARCHAR(20) NOT NULL CHECK (status IN (
        'pending', 'running', 'completed', 'failed', 'timeout'
    )),
    error_message TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    FOREIGN KEY (manuscript_id) REFERENCES manuscript_workflows(manuscript_id) ON DELETE CASCADE
);

CREATE INDEX idx_cognitive_loop_manuscript ON cognitive_loop_executions(manuscript_id);
CREATE INDEX idx_cognitive_loop_phase ON cognitive_loop_executions(phase);
CREATE INDEX idx_cognitive_loop_status ON cognitive_loop_executions(status);
CREATE INDEX idx_cognitive_loop_created ON cognitive_loop_executions(created_at DESC);
CREATE INDEX idx_cognitive_loop_execution_id ON cognitive_loop_executions(execution_id);

-- ============================================================================
-- AGENT COMMUNICATIONS TABLE
-- ============================================================================
-- Tracks inter-agent communication messages

CREATE TABLE IF NOT EXISTS agent_communications (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    communication_id VARCHAR(100) UNIQUE NOT NULL,
    sender_agent_id VARCHAR(50),
    receiver_agent_id VARCHAR(50),
    message_type VARCHAR(50) NOT NULL CHECK (message_type IN (
        'REQUEST', 'RESPONSE', 'NOTIFY', 'QUERY', 'COMMAND'
    )),
    message_content JSONB NOT NULL,
    priority INTEGER DEFAULT 5 CHECK (priority BETWEEN 1 AND 10),
    status VARCHAR(20) NOT NULL DEFAULT 'pending' CHECK (status IN (
        'pending', 'sent', 'received', 'processed', 'failed'
    )),
    response JSONB,
    correlation_id VARCHAR(100),
    sent_at TIMESTAMP DEFAULT NOW(),
    received_at TIMESTAMP,
    processed_at TIMESTAMP,
    FOREIGN KEY (sender_agent_id) REFERENCES agents_state(agent_id) ON DELETE SET NULL,
    FOREIGN KEY (receiver_agent_id) REFERENCES agents_state(agent_id) ON DELETE SET NULL
);

CREATE INDEX idx_agent_comms_sender ON agent_communications(sender_agent_id);
CREATE INDEX idx_agent_comms_receiver ON agent_communications(receiver_agent_id);
CREATE INDEX idx_agent_comms_status ON agent_communications(status);
CREATE INDEX idx_agent_comms_priority ON agent_communications(priority DESC);
CREATE INDEX idx_agent_comms_correlation ON agent_communications(correlation_id);
CREATE INDEX idx_agent_comms_sent ON agent_communications(sent_at DESC);

-- ============================================================================
-- PERFORMANCE ANALYTICS TABLE
-- ============================================================================
-- Stores performance metrics for agents and workflows

CREATE TABLE IF NOT EXISTS performance_analytics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_id VARCHAR(100) UNIQUE NOT NULL,
    agent_id VARCHAR(50),
    manuscript_id VARCHAR(100),
    metric_type VARCHAR(50) NOT NULL CHECK (metric_type IN (
        'latency', 'throughput', 'accuracy', 'quality', 'efficiency', 'error_rate'
    )),
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(10,4),
    metric_unit VARCHAR(20),
    metadata JSONB DEFAULT '{}',
    recorded_at TIMESTAMP DEFAULT NOW(),
    FOREIGN KEY (agent_id) REFERENCES agents_state(agent_id) ON DELETE CASCADE
);

CREATE INDEX idx_performance_agent ON performance_analytics(agent_id);
CREATE INDEX idx_performance_manuscript ON performance_analytics(manuscript_id);
CREATE INDEX idx_performance_type ON performance_analytics(metric_type);
CREATE INDEX idx_performance_recorded ON performance_analytics(recorded_at DESC);
CREATE INDEX idx_performance_name ON performance_analytics(metric_name);

-- ============================================================================
-- WORKFLOW EVENTS TABLE
-- ============================================================================
-- Event sourcing for workflow state changes

CREATE TABLE IF NOT EXISTS workflow_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_id VARCHAR(100) UNIQUE NOT NULL,
    manuscript_id VARCHAR(100),
    event_type VARCHAR(50) NOT NULL,
    event_name VARCHAR(100) NOT NULL,
    previous_state VARCHAR(50),
    new_state VARCHAR(50),
    triggered_by VARCHAR(100),
    event_data JSONB DEFAULT '{}',
    timestamp TIMESTAMP DEFAULT NOW(),
    FOREIGN KEY (manuscript_id) REFERENCES manuscript_workflows(manuscript_id) ON DELETE CASCADE
);

CREATE INDEX idx_workflow_events_manuscript ON workflow_events(manuscript_id);
CREATE INDEX idx_workflow_events_type ON workflow_events(event_type);
CREATE INDEX idx_workflow_events_timestamp ON workflow_events(timestamp DESC);

-- ============================================================================
-- REVIEWER POOL TABLE
-- ============================================================================
-- Manages the pool of available reviewers and their expertise

CREATE TABLE IF NOT EXISTS reviewer_pool (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    reviewer_id VARCHAR(100) UNIQUE NOT NULL,
    ojs_user_id INTEGER,
    name VARCHAR(200) NOT NULL,
    email VARCHAR(200) NOT NULL,
    expertise_areas JSONB DEFAULT '[]',
    specializations JSONB DEFAULT '[]',
    availability_status VARCHAR(20) DEFAULT 'available' CHECK (availability_status IN (
        'available', 'busy', 'unavailable', 'on_leave'
    )),
    current_workload INTEGER DEFAULT 0,
    max_workload INTEGER DEFAULT 5,
    performance_score DECIMAL(5,4) DEFAULT 0.5,
    review_history JSONB DEFAULT '[]',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_reviewer_pool_reviewer_id ON reviewer_pool(reviewer_id);
CREATE INDEX idx_reviewer_pool_availability ON reviewer_pool(availability_status);
CREATE INDEX idx_reviewer_pool_workload ON reviewer_pool(current_workload);
CREATE INDEX idx_reviewer_pool_performance ON reviewer_pool(performance_score DESC);

CREATE TRIGGER update_reviewer_pool_updated_at BEFORE UPDATE ON reviewer_pool
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- INITIAL DATA SEEDING
-- ============================================================================
-- Insert the 7 autonomous agents

INSERT INTO agents_state (agent_id, agent_name, agent_type, current_phase, status) VALUES
    ('agent_001', 'Research Discovery Agent', 'research_discovery', 'idle', 'idle'),
    ('agent_002', 'Submission Assistant Agent', 'submission_assistant', 'idle', 'idle'),
    ('agent_003', 'Editorial Orchestration Agent', 'editorial_orchestration', 'idle', 'idle'),
    ('agent_004', 'Review Coordination Agent', 'review_coordination', 'idle', 'idle'),
    ('agent_005', 'Content Quality Agent', 'content_quality', 'idle', 'idle'),
    ('agent_006', 'Publishing Production Agent', 'publishing_production', 'idle', 'idle'),
    ('agent_007', 'Analytics & Monitoring Agent', 'analytics_monitoring', 'idle', 'idle')
ON CONFLICT (agent_id) DO NOTHING;

-- ============================================================================
-- VIEWS FOR COMMON QUERIES
-- ============================================================================

-- Active manuscripts view
CREATE OR REPLACE VIEW active_manuscripts AS
SELECT 
    mw.manuscript_id,
    mw.title,
    mw.current_stage,
    mw.workflow_state,
    mw.created_at,
    mw.updated_at,
    COUNT(DISTINCT cle.id) as cognitive_loop_count,
    AVG(cle.relevance_score) as avg_relevance_score,
    AVG(cle.confidence) as avg_confidence
FROM manuscript_workflows mw
LEFT JOIN cognitive_loop_executions cle ON mw.manuscript_id = cle.manuscript_id
WHERE mw.current_stage NOT IN ('published', 'desk-rejection', 'rejection-notification')
GROUP BY mw.manuscript_id, mw.title, mw.current_stage, mw.workflow_state, mw.created_at, mw.updated_at;

-- Agent performance summary view
CREATE OR REPLACE VIEW agent_performance_summary AS
SELECT 
    a.agent_id,
    a.agent_name,
    a.agent_type,
    a.status,
    COUNT(DISTINCT ac.id) as total_communications,
    AVG(pa.metric_value) FILTER (WHERE pa.metric_type = 'latency') as avg_latency,
    AVG(pa.metric_value) FILTER (WHERE pa.metric_type = 'accuracy') as avg_accuracy,
    MAX(a.last_active) as last_active
FROM agents_state a
LEFT JOIN agent_communications ac ON a.agent_id = ac.sender_agent_id
LEFT JOIN performance_analytics pa ON a.agent_id = pa.agent_id
GROUP BY a.agent_id, a.agent_name, a.agent_type, a.status;

-- Workflow timeline view
CREATE OR REPLACE VIEW workflow_timeline AS
SELECT 
    we.manuscript_id,
    we.event_name,
    we.previous_state,
    we.new_state,
    we.triggered_by,
    we.timestamp,
    LEAD(we.timestamp) OVER (PARTITION BY we.manuscript_id ORDER BY we.timestamp) - we.timestamp as duration
FROM workflow_events we
ORDER BY we.manuscript_id, we.timestamp;

-- ============================================================================
-- FUNCTIONS FOR WORKFLOW MANAGEMENT
-- ============================================================================

-- Function to transition manuscript workflow state
CREATE OR REPLACE FUNCTION transition_manuscript_state(
    p_manuscript_id VARCHAR(100),
    p_event VARCHAR(100),
    p_triggered_by VARCHAR(100)
) RETURNS BOOLEAN AS $$
DECLARE
    v_current_state VARCHAR(50);
    v_new_state VARCHAR(50);
    v_event_id VARCHAR(100);
BEGIN
    -- Get current state
    SELECT current_stage INTO v_current_state
    FROM manuscript_workflows
    WHERE manuscript_id = p_manuscript_id;
    
    -- Determine new state based on current state and event
    v_new_state := CASE
        WHEN v_current_state = 'initial' AND p_event = 'manuscript-received' THEN 'submission'
        WHEN v_current_state = 'submission' AND p_event = 'validated' THEN 'quality-assessment'
        WHEN v_current_state = 'quality-assessment' AND p_event = 'passed' THEN 'review-assignment'
        WHEN v_current_state = 'quality-assessment' AND p_event = 'failed' THEN 'desk-rejection'
        WHEN v_current_state = 'review-assignment' AND p_event = 'reviewers-assigned' THEN 'under-review'
        WHEN v_current_state = 'under-review' AND p_event = 'reviews-completed' THEN 'editorial-decision'
        WHEN v_current_state = 'editorial-decision' AND p_event = 'accepted' THEN 'production'
        WHEN v_current_state = 'editorial-decision' AND p_event = 'rejected' THEN 'rejection-notification'
        WHEN v_current_state = 'editorial-decision' AND p_event = 'revisions-required' THEN 'revision-stage'
        WHEN v_current_state = 'revision-stage' AND p_event = 'revisions-submitted' THEN 'quality-assessment'
        WHEN v_current_state = 'production' AND p_event = 'formatted' THEN 'publication-ready'
        WHEN v_current_state = 'publication-ready' AND p_event = 'published' THEN 'published'
        ELSE v_current_state
    END;
    
    -- If state changed, update and log event
    IF v_new_state != v_current_state THEN
        -- Update manuscript state
        UPDATE manuscript_workflows
        SET current_stage = v_new_state,
            workflow_state = p_event,
            updated_at = NOW()
        WHERE manuscript_id = p_manuscript_id;
        
        -- Log workflow event
        v_event_id := 'evt_' || gen_random_uuid()::text;
        INSERT INTO workflow_events (event_id, manuscript_id, event_type, event_name, previous_state, new_state, triggered_by)
        VALUES (v_event_id, p_manuscript_id, 'state_transition', p_event, v_current_state, v_new_state, p_triggered_by);
        
        RETURN TRUE;
    END IF;
    
    RETURN FALSE;
END;
$$ LANGUAGE plpgsql;

-- Function to update agent status
CREATE OR REPLACE FUNCTION update_agent_status(
    p_agent_id VARCHAR(50),
    p_status VARCHAR(20),
    p_context JSONB DEFAULT NULL
) RETURNS VOID AS $$
BEGIN
    UPDATE agents_state
    SET status = p_status,
        context = COALESCE(p_context, context),
        last_active = NOW(),
        updated_at = NOW()
    WHERE agent_id = p_agent_id;
END;
$$ LANGUAGE plpgsql;

-- Function to record performance metric
CREATE OR REPLACE FUNCTION record_performance_metric(
    p_agent_id VARCHAR(50),
    p_metric_type VARCHAR(50),
    p_metric_name VARCHAR(100),
    p_metric_value DECIMAL(10,4),
    p_metric_unit VARCHAR(20),
    p_metadata JSONB DEFAULT '{}'
) RETURNS VARCHAR(100) AS $$
DECLARE
    v_metric_id VARCHAR(100);
BEGIN
    v_metric_id := 'metric_' || gen_random_uuid()::text;
    
    INSERT INTO performance_analytics (metric_id, agent_id, metric_type, metric_name, metric_value, metric_unit, metadata)
    VALUES (v_metric_id, p_agent_id, p_metric_type, p_metric_name, p_metric_value, p_metric_unit, p_metadata);
    
    RETURN v_metric_id;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- GRANTS AND PERMISSIONS
-- ============================================================================
-- Grant appropriate permissions (adjust based on your security requirements)

-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO ojscog_app;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO ojscog_app;
-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO ojscog_app;

-- ============================================================================
-- SCHEMA VERSION TRACKING
-- ============================================================================

CREATE TABLE IF NOT EXISTS schema_version (
    version VARCHAR(20) PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT NOW(),
    description TEXT
);

INSERT INTO schema_version (version, description) VALUES
    ('1.0.0', 'Initial OJSCog Supabase schema with agent state management and workflow tracking')
ON CONFLICT (version) DO NOTHING;

-- End of Supabase schema
