-- OJS-SKZ Integration Database Schema Extension
-- Version 1.0 - November 2025
-- Purpose: Extend OJS database to support autonomous agent operations

-- ============================================================================
-- Agent State Management Tables
-- ============================================================================

-- Agent State Table: Tracks current state of each autonomous agent
CREATE TABLE IF NOT EXISTS agent_state (
    agent_state_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    agent_id VARCHAR(100) NOT NULL,
    agent_name VARCHAR(255) NOT NULL,
    agent_type ENUM(
        'research_discovery',
        'submission_assistant',
        'editorial_orchestration',
        'review_coordination',
        'content_quality',
        'publishing_production',
        'analytics_monitoring'
    ) NOT NULL,
    status ENUM('active', 'idle', 'processing', 'error', 'maintenance') DEFAULT 'idle',
    current_task TEXT,
    state_data JSON,
    last_heartbeat TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY unique_agent_id (agent_id),
    INDEX idx_agent_type (agent_type),
    INDEX idx_status (status),
    INDEX idx_last_heartbeat (last_heartbeat)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Agent Decision Log: Records all decisions made by agents
CREATE TABLE IF NOT EXISTS agent_decisions (
    decision_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    agent_id VARCHAR(100) NOT NULL,
    agent_type VARCHAR(50) NOT NULL,
    submission_id BIGINT,
    decision_type VARCHAR(100) NOT NULL,
    decision_data JSON NOT NULL,
    confidence_score DECIMAL(5,4),
    reasoning TEXT,
    outcome VARCHAR(50),
    human_override BOOLEAN DEFAULT FALSE,
    override_reason TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_agent_id (agent_id),
    INDEX idx_submission_id (submission_id),
    INDEX idx_decision_type (decision_type),
    INDEX idx_created_at (created_at),
    INDEX idx_confidence_score (confidence_score)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Agent Performance Metrics: Tracks agent performance over time
CREATE TABLE IF NOT EXISTS agent_metrics (
    metric_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    agent_id VARCHAR(100) NOT NULL,
    agent_type VARCHAR(50) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(10,4),
    metric_unit VARCHAR(50),
    metadata JSON,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_agent_id (agent_id),
    INDEX idx_agent_type (agent_type),
    INDEX idx_metric_name (metric_name),
    INDEX idx_recorded_at (recorded_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================================
-- Workflow Integration Tables
-- ============================================================================

-- Workflow Transitions: Tracks manuscript movement through workflow stages
CREATE TABLE IF NOT EXISTS workflow_transitions (
    transition_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    submission_id BIGINT NOT NULL,
    from_stage ENUM('submission', 'review', 'copyediting', 'production', 'published') NOT NULL,
    to_stage ENUM('submission', 'review', 'copyediting', 'production', 'published') NOT NULL,
    transition_type ENUM('automatic', 'manual', 'agent_triggered') NOT NULL,
    triggered_by VARCHAR(255),
    agent_id VARCHAR(100),
    transition_data JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_submission_id (submission_id),
    INDEX idx_from_stage (from_stage),
    INDEX idx_to_stage (to_stage),
    INDEX idx_transition_type (transition_type),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Agent Task Queue: Manages asynchronous agent tasks
CREATE TABLE IF NOT EXISTS agent_task_queue (
    task_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    agent_id VARCHAR(100) NOT NULL,
    agent_type VARCHAR(50) NOT NULL,
    task_type VARCHAR(100) NOT NULL,
    submission_id BIGINT,
    priority INT DEFAULT 5,
    status ENUM('pending', 'processing', 'completed', 'failed', 'cancelled') DEFAULT 'pending',
    task_data JSON,
    result_data JSON,
    error_message TEXT,
    retry_count INT DEFAULT 0,
    max_retries INT DEFAULT 3,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP NULL,
    completed_at TIMESTAMP NULL,
    INDEX idx_agent_id (agent_id),
    INDEX idx_status (status),
    INDEX idx_priority (priority),
    INDEX idx_submission_id (submission_id),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================================
-- Submission Enhancement Tables
-- ============================================================================

-- Submission Analysis: Stores automated analysis results
CREATE TABLE IF NOT EXISTS submission_analysis (
    analysis_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    submission_id BIGINT NOT NULL,
    agent_id VARCHAR(100) NOT NULL,
    analysis_type VARCHAR(100) NOT NULL,
    analysis_results JSON NOT NULL,
    quality_score DECIMAL(5,2),
    flags JSON,
    recommendations TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_submission_id (submission_id),
    INDEX idx_agent_id (agent_id),
    INDEX idx_analysis_type (analysis_type),
    INDEX idx_quality_score (quality_score)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- INCI Validation: Cosmetic ingredient validation results
CREATE TABLE IF NOT EXISTS inci_validation (
    validation_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    submission_id BIGINT NOT NULL,
    ingredient_name VARCHAR(500) NOT NULL,
    inci_name VARCHAR(500),
    cas_number VARCHAR(50),
    validation_status ENUM('valid', 'invalid', 'warning', 'pending') NOT NULL,
    safety_profile JSON,
    regulatory_status JSON,
    validation_notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_submission_id (submission_id),
    INDEX idx_validation_status (validation_status),
    INDEX idx_ingredient_name (ingredient_name(255))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================================
-- Review Enhancement Tables
-- ============================================================================

-- Reviewer Matching: ML-based reviewer recommendations
CREATE TABLE IF NOT EXISTS reviewer_matching (
    matching_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    submission_id BIGINT NOT NULL,
    reviewer_id BIGINT NOT NULL,
    match_score DECIMAL(5,4) NOT NULL,
    expertise_match DECIMAL(5,4),
    workload_score DECIMAL(5,4),
    availability_score DECIMAL(5,4),
    quality_prediction DECIMAL(5,4),
    matching_reasoning JSON,
    status ENUM('recommended', 'invited', 'accepted', 'declined', 'completed') DEFAULT 'recommended',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_submission_id (submission_id),
    INDEX idx_reviewer_id (reviewer_id),
    INDEX idx_match_score (match_score),
    INDEX idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Review Quality Prediction: AI predictions for review outcomes
CREATE TABLE IF NOT EXISTS review_quality_prediction (
    prediction_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    submission_id BIGINT NOT NULL,
    reviewer_id BIGINT NOT NULL,
    predicted_quality DECIMAL(5,2),
    predicted_timeliness INT,
    predicted_recommendation ENUM('accept', 'minor_revision', 'major_revision', 'reject'),
    confidence_score DECIMAL(5,4),
    prediction_factors JSON,
    actual_quality DECIMAL(5,2),
    actual_timeliness INT,
    actual_recommendation VARCHAR(50),
    prediction_accuracy DECIMAL(5,4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_submission_id (submission_id),
    INDEX idx_reviewer_id (reviewer_id),
    INDEX idx_predicted_recommendation (predicted_recommendation)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================================
-- Audit and Compliance Tables
-- ============================================================================

-- Comprehensive Audit Trail: Records all system actions
CREATE TABLE IF NOT EXISTS audit_trail (
    audit_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    event_type VARCHAR(100) NOT NULL,
    entity_type VARCHAR(50) NOT NULL,
    entity_id BIGINT,
    actor_type ENUM('user', 'agent', 'system') NOT NULL,
    actor_id VARCHAR(255) NOT NULL,
    action VARCHAR(100) NOT NULL,
    event_data JSON,
    ip_address VARCHAR(45),
    user_agent TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_event_type (event_type),
    INDEX idx_entity_type (entity_type),
    INDEX idx_entity_id (entity_id),
    INDEX idx_actor_type (actor_type),
    INDEX idx_actor_id (actor_id),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Agent Communication Log: Inter-agent communication tracking
CREATE TABLE IF NOT EXISTS agent_communication (
    communication_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    from_agent_id VARCHAR(100) NOT NULL,
    to_agent_id VARCHAR(100) NOT NULL,
    message_type VARCHAR(100) NOT NULL,
    message_data JSON NOT NULL,
    priority INT DEFAULT 5,
    status ENUM('sent', 'delivered', 'processed', 'failed') DEFAULT 'sent',
    response_data JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    delivered_at TIMESTAMP NULL,
    processed_at TIMESTAMP NULL,
    INDEX idx_from_agent (from_agent_id),
    INDEX idx_to_agent (to_agent_id),
    INDEX idx_message_type (message_type),
    INDEX idx_status (status),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================================
-- Analytics and Reporting Tables
-- ============================================================================

-- System Performance Metrics: Overall system health and performance
CREATE TABLE IF NOT EXISTS system_metrics (
    metric_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    metric_category VARCHAR(100) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(15,4),
    metric_unit VARCHAR(50),
    aggregation_period ENUM('minute', 'hour', 'day', 'week', 'month') DEFAULT 'hour',
    metadata JSON,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_metric_category (metric_category),
    INDEX idx_metric_name (metric_name),
    INDEX idx_aggregation_period (aggregation_period),
    INDEX idx_recorded_at (recorded_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Workflow Analytics: Aggregated workflow performance data
CREATE TABLE IF NOT EXISTS workflow_analytics (
    analytics_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    date_period DATE NOT NULL,
    workflow_stage VARCHAR(50) NOT NULL,
    total_submissions INT DEFAULT 0,
    avg_processing_time DECIMAL(10,2),
    automation_rate DECIMAL(5,4),
    success_rate DECIMAL(5,4),
    bottleneck_score DECIMAL(5,4),
    analytics_data JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY unique_period_stage (date_period, workflow_stage),
    INDEX idx_date_period (date_period),
    INDEX idx_workflow_stage (workflow_stage)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================================
-- Configuration and Settings Tables
-- ============================================================================

-- Agent Configuration: Runtime configuration for each agent
CREATE TABLE IF NOT EXISTS agent_configuration (
    config_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    agent_id VARCHAR(100) NOT NULL,
    agent_type VARCHAR(50) NOT NULL,
    config_key VARCHAR(255) NOT NULL,
    config_value TEXT,
    config_type ENUM('string', 'integer', 'float', 'boolean', 'json') DEFAULT 'string',
    description TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY unique_agent_config (agent_id, config_key),
    INDEX idx_agent_id (agent_id),
    INDEX idx_agent_type (agent_type),
    INDEX idx_is_active (is_active)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Workflow Rules: Configurable automation rules
CREATE TABLE IF NOT EXISTS workflow_rules (
    rule_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    rule_name VARCHAR(255) NOT NULL,
    rule_type VARCHAR(100) NOT NULL,
    workflow_stage VARCHAR(50) NOT NULL,
    conditions JSON NOT NULL,
    actions JSON NOT NULL,
    priority INT DEFAULT 5,
    is_active BOOLEAN DEFAULT TRUE,
    created_by VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_rule_type (rule_type),
    INDEX idx_workflow_stage (workflow_stage),
    INDEX idx_is_active (is_active),
    INDEX idx_priority (priority)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================================
-- Initial Data Population
-- ============================================================================

-- Insert default agent configurations
INSERT INTO agent_state (agent_id, agent_name, agent_type, status) VALUES
('research_discovery_001', 'Research Discovery Agent', 'research_discovery', 'idle'),
('submission_assistant_001', 'Submission Assistant Agent', 'submission_assistant', 'idle'),
('editorial_orchestration_001', 'Editorial Orchestration Agent', 'editorial_orchestration', 'idle'),
('review_coordination_001', 'Review Coordination Agent', 'review_coordination', 'idle'),
('content_quality_001', 'Content Quality Agent', 'content_quality', 'idle'),
('publishing_production_001', 'Publishing Production Agent', 'publishing_production', 'idle'),
('analytics_monitoring_001', 'Analytics & Monitoring Agent', 'analytics_monitoring', 'idle')
ON DUPLICATE KEY UPDATE updated_at = CURRENT_TIMESTAMP;

-- Insert default agent configurations
INSERT INTO agent_configuration (agent_id, agent_type, config_key, config_value, config_type, description) VALUES
('research_discovery_001', 'research_discovery', 'inci_database_url', 'https://api.inci.example.com', 'string', 'INCI database API endpoint'),
('research_discovery_001', 'research_discovery', 'patent_api_enabled', 'true', 'boolean', 'Enable patent landscape analysis'),
('submission_assistant_001', 'submission_assistant', 'quality_threshold', '0.75', 'float', 'Minimum quality score for automatic approval'),
('submission_assistant_001', 'submission_assistant', 'plagiarism_check_enabled', 'true', 'boolean', 'Enable plagiarism detection'),
('review_coordination_001', 'review_coordination', 'min_reviewers', '3', 'integer', 'Minimum number of reviewers per submission'),
('review_coordination_001', 'review_coordination', 'max_workload_per_reviewer', '5', 'integer', 'Maximum concurrent reviews per reviewer'),
('content_quality_001', 'content_quality', 'safety_validation_enabled', 'true', 'boolean', 'Enable safety compliance checking'),
('publishing_production_001', 'publishing_production', 'output_formats', '["pdf", "html", "xml"]', 'json', 'Supported output formats'),
('analytics_monitoring_001', 'analytics_monitoring', 'alert_threshold', '0.90', 'float', 'Performance alert threshold')
ON DUPLICATE KEY UPDATE updated_at = CURRENT_TIMESTAMP;

-- ============================================================================
-- Database Views for Common Queries
-- ============================================================================

-- View: Active Agent Status
CREATE OR REPLACE VIEW v_active_agents AS
SELECT 
    agent_id,
    agent_name,
    agent_type,
    status,
    current_task,
    last_heartbeat,
    TIMESTAMPDIFF(SECOND, last_heartbeat, NOW()) as seconds_since_heartbeat
FROM agent_state
WHERE status IN ('active', 'processing')
ORDER BY last_heartbeat DESC;

-- View: Recent Agent Decisions
CREATE OR REPLACE VIEW v_recent_decisions AS
SELECT 
    ad.decision_id,
    ad.agent_id,
    ad.agent_type,
    ad.submission_id,
    ad.decision_type,
    ad.confidence_score,
    ad.outcome,
    ad.human_override,
    ad.created_at,
    ast.agent_name
FROM agent_decisions ad
LEFT JOIN agent_state ast ON ad.agent_id = ast.agent_id
ORDER BY ad.created_at DESC
LIMIT 100;

-- View: Submission Processing Status
CREATE OR REPLACE VIEW v_submission_processing AS
SELECT 
    sa.submission_id,
    COUNT(DISTINCT sa.agent_id) as agents_processed,
    AVG(sa.quality_score) as avg_quality_score,
    MAX(sa.created_at) as last_analysis_time,
    GROUP_CONCAT(DISTINCT sa.analysis_type) as analysis_types
FROM submission_analysis sa
GROUP BY sa.submission_id;

-- View: Agent Performance Summary
CREATE OR REPLACE VIEW v_agent_performance AS
SELECT 
    agent_id,
    agent_type,
    COUNT(*) as total_decisions,
    AVG(confidence_score) as avg_confidence,
    SUM(CASE WHEN human_override = TRUE THEN 1 ELSE 0 END) as override_count,
    SUM(CASE WHEN human_override = TRUE THEN 1 ELSE 0 END) / COUNT(*) as override_rate
FROM agent_decisions
WHERE created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
GROUP BY agent_id, agent_type;

-- ============================================================================
-- Stored Procedures for Common Operations
-- ============================================================================

DELIMITER //

-- Procedure: Log Agent Decision
CREATE PROCEDURE IF NOT EXISTS sp_log_agent_decision(
    IN p_agent_id VARCHAR(100),
    IN p_agent_type VARCHAR(50),
    IN p_submission_id BIGINT,
    IN p_decision_type VARCHAR(100),
    IN p_decision_data JSON,
    IN p_confidence_score DECIMAL(5,4),
    IN p_reasoning TEXT
)
BEGIN
    INSERT INTO agent_decisions (
        agent_id, agent_type, submission_id, decision_type,
        decision_data, confidence_score, reasoning
    ) VALUES (
        p_agent_id, p_agent_type, p_submission_id, p_decision_type,
        p_decision_data, p_confidence_score, p_reasoning
    );
    
    INSERT INTO audit_trail (
        event_type, entity_type, entity_id, actor_type, actor_id, action, event_data
    ) VALUES (
        'agent_decision', 'submission', p_submission_id, 'agent', p_agent_id,
        p_decision_type, p_decision_data
    );
END//

-- Procedure: Update Agent Heartbeat
CREATE PROCEDURE IF NOT EXISTS sp_update_agent_heartbeat(
    IN p_agent_id VARCHAR(100),
    IN p_status VARCHAR(50),
    IN p_current_task TEXT
)
BEGIN
    UPDATE agent_state
    SET status = p_status,
        current_task = p_current_task,
        last_heartbeat = CURRENT_TIMESTAMP
    WHERE agent_id = p_agent_id;
END//

-- Procedure: Queue Agent Task
CREATE PROCEDURE IF NOT EXISTS sp_queue_agent_task(
    IN p_agent_id VARCHAR(100),
    IN p_agent_type VARCHAR(50),
    IN p_task_type VARCHAR(100),
    IN p_submission_id BIGINT,
    IN p_priority INT,
    IN p_task_data JSON,
    OUT p_task_id BIGINT
)
BEGIN
    INSERT INTO agent_task_queue (
        agent_id, agent_type, task_type, submission_id, priority, task_data
    ) VALUES (
        p_agent_id, p_agent_type, p_task_type, p_submission_id, p_priority, p_task_data
    );
    
    SET p_task_id = LAST_INSERT_ID();
END//

DELIMITER ;

-- ============================================================================
-- Database Triggers for Workflow Automation
-- ============================================================================

-- Note: Triggers will be created in separate migration file to avoid conflicts
-- with existing OJS tables. See workflow_triggers.sql

-- ============================================================================
-- Indexes for Performance Optimization
-- ============================================================================

-- Additional composite indexes for common query patterns
CREATE INDEX idx_agent_submission ON agent_decisions(agent_id, submission_id);
CREATE INDEX idx_submission_analysis_type ON submission_analysis(submission_id, analysis_type);
CREATE INDEX idx_task_status_priority ON agent_task_queue(status, priority, created_at);
CREATE INDEX idx_audit_entity ON audit_trail(entity_type, entity_id, created_at);

-- ============================================================================
-- Schema Version Tracking
-- ============================================================================

CREATE TABLE IF NOT EXISTS schema_version (
    version_id INT AUTO_INCREMENT PRIMARY KEY,
    version_number VARCHAR(20) NOT NULL,
    description TEXT,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY unique_version (version_number)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

INSERT INTO schema_version (version_number, description) VALUES
('1.0.0', 'Initial OJS-SKZ integration schema with agent state management, workflow integration, and analytics tables');

-- ============================================================================
-- End of Schema Extension
-- ============================================================================
