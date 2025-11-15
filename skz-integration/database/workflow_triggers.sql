-- OJS-SKZ Workflow Triggers
-- Version 1.0 - November 2025
-- Purpose: Automate agent activation based on OJS workflow events

-- ============================================================================
-- Trigger Definitions for OJS Workflow Integration
-- ============================================================================

-- Note: These triggers assume OJS 3.x database structure
-- Adjust table names and columns based on actual OJS installation

DELIMITER //

-- ============================================================================
-- Submission Stage Triggers
-- ============================================================================

-- Trigger: After new submission is created
-- Activates Research Discovery and Submission Assistant agents
CREATE TRIGGER IF NOT EXISTS trg_after_submission_insert
AFTER INSERT ON submissions
FOR EACH ROW
BEGIN
    DECLARE v_task_id BIGINT;
    
    -- Queue Research Discovery Agent task
    CALL sp_queue_agent_task(
        'research_discovery_001',
        'research_discovery',
        'analyze_novelty',
        NEW.submission_id,
        8, -- High priority
        JSON_OBJECT(
            'submission_id', NEW.submission_id,
            'title', NEW.title,
            'abstract', NEW.abstract,
            'keywords', NEW.keywords
        ),
        v_task_id
    );
    
    -- Queue Submission Assistant Agent task
    CALL sp_queue_agent_task(
        'submission_assistant_001',
        'submission_assistant',
        'quality_assessment',
        NEW.submission_id,
        8, -- High priority
        JSON_OBJECT(
            'submission_id', NEW.submission_id,
            'check_inci', TRUE,
            'check_safety', TRUE,
            'check_plagiarism', TRUE
        ),
        v_task_id
    );
    
    -- Log workflow transition
    INSERT INTO workflow_transitions (
        submission_id, from_stage, to_stage, transition_type, triggered_by
    ) VALUES (
        NEW.submission_id, 'submission', 'submission', 'automatic', 'system'
    );
    
    -- Log audit trail
    INSERT INTO audit_trail (
        event_type, entity_type, entity_id, actor_type, actor_id, action, event_data
    ) VALUES (
        'submission_created', 'submission', NEW.submission_id, 'system', 'ojs_core',
        'create', JSON_OBJECT('submission_id', NEW.submission_id, 'title', NEW.title)
    );
END//

-- Trigger: After submission metadata is updated
-- Re-triggers analysis if significant changes detected
CREATE TRIGGER IF NOT EXISTS trg_after_submission_update
AFTER UPDATE ON submissions
FOR EACH ROW
BEGIN
    DECLARE v_task_id BIGINT;
    
    -- Check if significant fields changed (title, abstract, keywords)
    IF OLD.title != NEW.title OR 
       OLD.abstract != NEW.abstract OR 
       OLD.keywords != NEW.keywords THEN
        
        -- Queue re-analysis task
        CALL sp_queue_agent_task(
            'submission_assistant_001',
            'submission_assistant',
            'reanalyze_submission',
            NEW.submission_id,
            5, -- Medium priority
            JSON_OBJECT(
                'submission_id', NEW.submission_id,
                'changes', JSON_OBJECT(
                    'title_changed', OLD.title != NEW.title,
                    'abstract_changed', OLD.abstract != NEW.abstract,
                    'keywords_changed', OLD.keywords != NEW.keywords
                )
            ),
            v_task_id
        );
        
        -- Log audit trail
        INSERT INTO audit_trail (
            event_type, entity_type, entity_id, actor_type, actor_id, action, event_data
        ) VALUES (
            'submission_updated', 'submission', NEW.submission_id, 'system', 'ojs_core',
            'update', JSON_OBJECT('submission_id', NEW.submission_id)
        );
    END IF;
END//

-- ============================================================================
-- Review Stage Triggers
-- ============================================================================

-- Trigger: When submission moves to review stage
-- Activates Review Coordination Agent for reviewer matching
CREATE TRIGGER IF NOT EXISTS trg_submission_to_review
AFTER INSERT ON review_rounds
FOR EACH ROW
BEGIN
    DECLARE v_task_id BIGINT;
    
    -- Queue Review Coordination Agent task for reviewer matching
    CALL sp_queue_agent_task(
        'review_coordination_001',
        'review_coordination',
        'match_reviewers',
        NEW.submission_id,
        9, -- Very high priority
        JSON_OBJECT(
            'submission_id', NEW.submission_id,
            'review_round_id', NEW.review_round_id,
            'num_reviewers', 3,
            'urgency', 'normal'
        ),
        v_task_id
    );
    
    -- Log workflow transition
    INSERT INTO workflow_transitions (
        submission_id, from_stage, to_stage, transition_type, 
        triggered_by, agent_id
    ) VALUES (
        NEW.submission_id, 'submission', 'review', 'automatic',
        'system', 'review_coordination_001'
    );
    
    -- Log audit trail
    INSERT INTO audit_trail (
        event_type, entity_type, entity_id, actor_type, actor_id, action, event_data
    ) VALUES (
        'review_round_created', 'submission', NEW.submission_id, 'system', 'ojs_core',
        'create_review_round', JSON_OBJECT('review_round_id', NEW.review_round_id)
    );
END//

-- Trigger: After review assignment is created
-- Activates Review Quality Prediction
CREATE TRIGGER IF NOT EXISTS trg_after_review_assignment
AFTER INSERT ON review_assignments
FOR EACH ROW
BEGIN
    DECLARE v_task_id BIGINT;
    
    -- Queue quality prediction task
    CALL sp_queue_agent_task(
        'review_coordination_001',
        'review_coordination',
        'predict_review_quality',
        NEW.submission_id,
        6, -- Medium-high priority
        JSON_OBJECT(
            'submission_id', NEW.submission_id,
            'reviewer_id', NEW.reviewer_id,
            'review_assignment_id', NEW.review_assignment_id
        ),
        v_task_id
    );
    
    -- Log audit trail
    INSERT INTO audit_trail (
        event_type, entity_type, entity_id, actor_type, actor_id, action, event_data
    ) VALUES (
        'review_assigned', 'submission', NEW.submission_id, 'system', 'ojs_core',
        'assign_reviewer', JSON_OBJECT('reviewer_id', NEW.reviewer_id)
    );
END//

-- Trigger: After review is completed
-- Activates Content Quality Agent for review analysis
CREATE TRIGGER IF NOT EXISTS trg_after_review_completed
AFTER UPDATE ON review_assignments
FOR EACH ROW
BEGIN
    DECLARE v_task_id BIGINT;
    
    -- Check if review was just completed
    IF OLD.date_completed IS NULL AND NEW.date_completed IS NOT NULL THEN
        
        -- Queue Content Quality Agent task
        CALL sp_queue_agent_task(
            'content_quality_001',
            'content_quality',
            'analyze_review',
            NEW.submission_id,
            7, -- High priority
            JSON_OBJECT(
                'submission_id', NEW.submission_id,
                'reviewer_id', NEW.reviewer_id,
                'review_assignment_id', NEW.review_assignment_id,
                'recommendation', NEW.recommendation
            ),
            v_task_id
        );
        
        -- Queue Editorial Orchestration Agent for decision synthesis
        CALL sp_queue_agent_task(
            'editorial_orchestration_001',
            'editorial_orchestration',
            'synthesize_reviews',
            NEW.submission_id,
            7, -- High priority
            JSON_OBJECT(
                'submission_id', NEW.submission_id,
                'check_completion', TRUE
            ),
            v_task_id
        );
        
        -- Log audit trail
        INSERT INTO audit_trail (
            event_type, entity_type, entity_id, actor_type, actor_id, action, event_data
        ) VALUES (
            'review_completed', 'submission', NEW.submission_id, 'system', 'ojs_core',
            'complete_review', JSON_OBJECT('reviewer_id', NEW.reviewer_id)
        );
    END IF;
END//

-- ============================================================================
-- Copyediting Stage Triggers
-- ============================================================================

-- Trigger: When submission moves to copyediting
-- Activates Content Quality Agent for copyediting support
CREATE TRIGGER IF NOT EXISTS trg_submission_to_copyediting
AFTER INSERT ON copyedit_assignments
FOR EACH ROW
BEGIN
    DECLARE v_task_id BIGINT;
    
    -- Queue Content Quality Agent task
    CALL sp_queue_agent_task(
        'content_quality_001',
        'content_quality',
        'copyediting_support',
        NEW.submission_id,
        6, -- Medium-high priority
        JSON_OBJECT(
            'submission_id', NEW.submission_id,
            'copyedit_assignment_id', NEW.copyedit_assignment_id,
            'check_style', TRUE,
            'check_references', TRUE,
            'check_consistency', TRUE
        ),
        v_task_id
    );
    
    -- Log workflow transition
    INSERT INTO workflow_transitions (
        submission_id, from_stage, to_stage, transition_type, 
        triggered_by, agent_id
    ) VALUES (
        NEW.submission_id, 'review', 'copyediting', 'automatic',
        'system', 'content_quality_001'
    );
    
    -- Log audit trail
    INSERT INTO audit_trail (
        event_type, entity_type, entity_id, actor_type, actor_id, action, event_data
    ) VALUES (
        'copyediting_started', 'submission', NEW.submission_id, 'system', 'ojs_core',
        'assign_copyeditor', JSON_OBJECT('copyedit_assignment_id', NEW.copyedit_assignment_id)
    );
END//

-- ============================================================================
-- Production Stage Triggers
-- ============================================================================

-- Trigger: When submission moves to production
-- Activates Publishing Production Agent
CREATE TRIGGER IF NOT EXISTS trg_submission_to_production
AFTER INSERT ON production_assignments
FOR EACH ROW
BEGIN
    DECLARE v_task_id BIGINT;
    
    -- Queue Publishing Production Agent task
    CALL sp_queue_agent_task(
        'publishing_production_001',
        'publishing_production',
        'prepare_publication',
        NEW.submission_id,
        8, -- High priority
        JSON_OBJECT(
            'submission_id', NEW.submission_id,
            'production_assignment_id', NEW.production_assignment_id,
            'formats', JSON_ARRAY('pdf', 'html', 'xml'),
            'generate_metadata', TRUE
        ),
        v_task_id
    );
    
    -- Log workflow transition
    INSERT INTO workflow_transitions (
        submission_id, from_stage, to_stage, transition_type, 
        triggered_by, agent_id
    ) VALUES (
        NEW.submission_id, 'copyediting', 'production', 'automatic',
        'system', 'publishing_production_001'
    );
    
    -- Log audit trail
    INSERT INTO audit_trail (
        event_type, entity_type, entity_id, actor_type, actor_id, action, event_data
    ) VALUES (
        'production_started', 'submission', NEW.submission_id, 'system', 'ojs_core',
        'assign_production', JSON_OBJECT('production_assignment_id', NEW.production_assignment_id)
    );
END//

-- Trigger: When galley is created (publication format ready)
-- Triggers final quality check
CREATE TRIGGER IF NOT EXISTS trg_after_galley_created
AFTER INSERT ON publication_galleys
FOR EACH ROW
BEGIN
    DECLARE v_task_id BIGINT;
    DECLARE v_submission_id BIGINT;
    
    -- Get submission_id from publication
    SELECT submission_id INTO v_submission_id
    FROM publications
    WHERE publication_id = NEW.publication_id;
    
    -- Queue final quality check
    CALL sp_queue_agent_task(
        'content_quality_001',
        'content_quality',
        'final_quality_check',
        v_submission_id,
        9, -- Very high priority
        JSON_OBJECT(
            'submission_id', v_submission_id,
            'publication_id', NEW.publication_id,
            'galley_id', NEW.galley_id,
            'format', NEW.label
        ),
        v_task_id
    );
    
    -- Log audit trail
    INSERT INTO audit_trail (
        event_type, entity_type, entity_id, actor_type, actor_id, action, event_data
    ) VALUES (
        'galley_created', 'submission', v_submission_id, 'system', 'ojs_core',
        'create_galley', JSON_OBJECT('galley_id', NEW.galley_id, 'format', NEW.label)
    );
END//

-- ============================================================================
-- Publication Triggers
-- ============================================================================

-- Trigger: When submission is published
-- Activates Analytics & Monitoring Agent
CREATE TRIGGER IF NOT EXISTS trg_after_publication
AFTER UPDATE ON publications
FOR EACH ROW
BEGIN
    DECLARE v_task_id BIGINT;
    
    -- Check if publication status changed to published
    IF OLD.status != 3 AND NEW.status = 3 THEN
        
        -- Queue Analytics & Monitoring Agent task
        CALL sp_queue_agent_task(
            'analytics_monitoring_001',
            'analytics_monitoring',
            'track_publication',
            NEW.submission_id,
            5, -- Medium priority
            JSON_OBJECT(
                'submission_id', NEW.submission_id,
                'publication_id', NEW.publication_id,
                'date_published', NEW.date_published
            ),
            v_task_id
        );
        
        -- Log workflow transition
        INSERT INTO workflow_transitions (
            submission_id, from_stage, to_stage, transition_type, 
            triggered_by, agent_id
        ) VALUES (
            NEW.submission_id, 'production', 'published', 'automatic',
            'system', 'analytics_monitoring_001'
        );
        
        -- Log audit trail
        INSERT INTO audit_trail (
            event_type, entity_type, entity_id, actor_type, actor_id, action, event_data
        ) VALUES (
            'submission_published', 'submission', NEW.submission_id, 'system', 'ojs_core',
            'publish', JSON_OBJECT('publication_id', NEW.publication_id)
        );
    END IF;
END//

-- ============================================================================
-- Agent Task Processing Triggers
-- ============================================================================

-- Trigger: After agent task is completed
-- Updates agent metrics and checks for dependent tasks
CREATE TRIGGER IF NOT EXISTS trg_after_task_completed
AFTER UPDATE ON agent_task_queue
FOR EACH ROW
BEGIN
    DECLARE v_processing_time INT;
    
    -- Check if task was just completed
    IF OLD.status != 'completed' AND NEW.status = 'completed' THEN
        
        -- Calculate processing time
        SET v_processing_time = TIMESTAMPDIFF(SECOND, NEW.started_at, NEW.completed_at);
        
        -- Record agent metric
        INSERT INTO agent_metrics (
            agent_id, agent_type, metric_name, metric_value, metric_unit, metadata
        ) VALUES (
            NEW.agent_id, NEW.agent_type, 'task_processing_time', v_processing_time, 'seconds',
            JSON_OBJECT('task_id', NEW.task_id, 'task_type', NEW.task_type)
        );
        
        -- Log audit trail
        INSERT INTO audit_trail (
            event_type, entity_type, entity_id, actor_type, actor_id, action, event_data
        ) VALUES (
            'task_completed', 'agent_task', NEW.task_id, 'agent', NEW.agent_id,
            'complete_task', JSON_OBJECT('task_type', NEW.task_type, 'processing_time', v_processing_time)
        );
    END IF;
    
    -- Check if task failed and needs retry
    IF OLD.status != 'failed' AND NEW.status = 'failed' AND NEW.retry_count < NEW.max_retries THEN
        
        -- Reset task for retry
        UPDATE agent_task_queue
        SET status = 'pending',
            retry_count = retry_count + 1,
            started_at = NULL
        WHERE task_id = NEW.task_id;
        
        -- Log retry attempt
        INSERT INTO audit_trail (
            event_type, entity_type, entity_id, actor_type, actor_id, action, event_data
        ) VALUES (
            'task_retry', 'agent_task', NEW.task_id, 'system', 'task_manager',
            'retry_task', JSON_OBJECT('retry_count', NEW.retry_count + 1, 'error', NEW.error_message)
        );
    END IF;
END//

DELIMITER ;

-- ============================================================================
-- Event Scheduler for Periodic Tasks
-- ============================================================================

-- Enable event scheduler
SET GLOBAL event_scheduler = ON;

-- Event: Clean up old completed tasks (runs daily)
CREATE EVENT IF NOT EXISTS evt_cleanup_completed_tasks
ON SCHEDULE EVERY 1 DAY
STARTS CURRENT_TIMESTAMP
DO
BEGIN
    -- Archive tasks older than 90 days
    DELETE FROM agent_task_queue
    WHERE status IN ('completed', 'cancelled')
    AND completed_at < DATE_SUB(NOW(), INTERVAL 90 DAY);
    
    -- Log cleanup
    INSERT INTO audit_trail (
        event_type, entity_type, entity_id, actor_type, actor_id, action, event_data
    ) VALUES (
        'maintenance', 'system', NULL, 'system', 'event_scheduler',
        'cleanup_tasks', JSON_OBJECT('timestamp', NOW())
    );
END;

-- Event: Update agent heartbeat monitoring (runs every minute)
CREATE EVENT IF NOT EXISTS evt_monitor_agent_heartbeats
ON SCHEDULE EVERY 1 MINUTE
STARTS CURRENT_TIMESTAMP
DO
BEGIN
    -- Mark agents as error if no heartbeat for 5 minutes
    UPDATE agent_state
    SET status = 'error'
    WHERE status IN ('active', 'processing')
    AND TIMESTAMPDIFF(SECOND, last_heartbeat, NOW()) > 300;
    
    -- Log stale agents
    INSERT INTO audit_trail (
        event_type, entity_type, entity_id, actor_type, actor_id, action, event_data
    )
    SELECT 
        'agent_timeout', 'agent', NULL, 'system', 'heartbeat_monitor',
        'mark_error', JSON_OBJECT('agent_id', agent_id, 'last_heartbeat', last_heartbeat)
    FROM agent_state
    WHERE status = 'error'
    AND TIMESTAMPDIFF(SECOND, last_heartbeat, NOW()) BETWEEN 300 AND 310;
END;

-- Event: Calculate daily workflow analytics (runs daily at midnight)
CREATE EVENT IF NOT EXISTS evt_calculate_workflow_analytics
ON SCHEDULE EVERY 1 DAY
STARTS (CURRENT_DATE + INTERVAL 1 DAY)
DO
BEGIN
    DECLARE v_date DATE;
    SET v_date = DATE_SUB(CURRENT_DATE, INTERVAL 1 DAY);
    
    -- Calculate submission stage analytics
    INSERT INTO workflow_analytics (
        date_period, workflow_stage, total_submissions, 
        avg_processing_time, automation_rate, success_rate
    )
    SELECT 
        v_date,
        'submission',
        COUNT(DISTINCT wt.submission_id),
        AVG(TIMESTAMPDIFF(SECOND, wt.created_at, 
            (SELECT MIN(created_at) FROM workflow_transitions 
             WHERE submission_id = wt.submission_id AND to_stage = 'review'))),
        COUNT(DISTINCT CASE WHEN wt.transition_type = 'automatic' THEN wt.submission_id END) / COUNT(DISTINCT wt.submission_id),
        COUNT(DISTINCT CASE WHEN EXISTS (
            SELECT 1 FROM workflow_transitions 
            WHERE submission_id = wt.submission_id AND to_stage = 'review'
        ) THEN wt.submission_id END) / COUNT(DISTINCT wt.submission_id)
    FROM workflow_transitions wt
    WHERE DATE(wt.created_at) = v_date
    AND wt.from_stage = 'submission'
    ON DUPLICATE KEY UPDATE
        total_submissions = VALUES(total_submissions),
        avg_processing_time = VALUES(avg_processing_time),
        automation_rate = VALUES(automation_rate),
        success_rate = VALUES(success_rate);
END;

-- ============================================================================
-- Trigger Version Tracking
-- ============================================================================

INSERT INTO schema_version (version_number, description) VALUES
('1.0.1', 'Workflow triggers for OJS-SKZ integration with automated agent activation')
ON DUPLICATE KEY UPDATE description = VALUES(description);

-- ============================================================================
-- End of Workflow Triggers
-- ============================================================================
