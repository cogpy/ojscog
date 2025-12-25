#!/bin/bash
#
# Database Setup Script
# Configures OJS database connection and creates necessary tables/indexes
#
# Usage: ./setup_database.sh
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Load environment variables
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
    log_info "Loaded environment from .env"
fi

# Database configuration
DB_HOST="${OJS_DB_HOST:-localhost}"
DB_PORT="${OJS_DB_PORT:-3306}"
DB_NAME="${OJS_DB_NAME:-ojs}"
DB_USER="${OJS_DB_USER:-ojs_user}"
DB_PASSWORD="${OJS_DB_PASSWORD}"

log_info "Database Setup for OJS Cognitive Enhancement"
echo ""

# Check if MySQL client is installed
if ! command -v mysql &> /dev/null; then
    log_error "MySQL client not found. Installing..."
    sudo apt-get update
    sudo apt-get install -y mysql-client
fi

# Test database connection
log_info "Testing database connection..."

if [ -z "$DB_PASSWORD" ]; then
    log_warning "Database password not set in environment"
    read -sp "Enter database password: " DB_PASSWORD
    echo ""
fi

# Test connection
if mysql -h "$DB_HOST" -P "$DB_PORT" -u "$DB_USER" -p"$DB_PASSWORD" -e "USE $DB_NAME;" 2>/dev/null; then
    log_success "Database connection successful"
else
    log_error "Failed to connect to database"
    log_error "Host: $DB_HOST:$DB_PORT, Database: $DB_NAME, User: $DB_USER"
    exit 1
fi

# Create extension tables for agent data
log_info "Creating extension tables..."

mysql -h "$DB_HOST" -P "$DB_PORT" -u "$DB_USER" -p"$DB_PASSWORD" "$DB_NAME" <<EOF

-- Agent execution log
CREATE TABLE IF NOT EXISTS agent_execution_log (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    agent_name VARCHAR(100) NOT NULL,
    submission_id INT,
    action VARCHAR(100),
    status VARCHAR(50),
    execution_time_ms INT,
    result_data JSON,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_agent_name (agent_name),
    INDEX idx_submission_id (submission_id),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Quality assessment cache
CREATE TABLE IF NOT EXISTS quality_assessment_cache (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    submission_id INT NOT NULL UNIQUE,
    quality_score DECIMAL(3,2),
    clarity_score DECIMAL(3,2),
    methodology_score DECIMAL(3,2),
    novelty_score DECIMAL(3,2),
    feedback TEXT,
    assessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_submission_id (submission_id),
    INDEX idx_quality_score (quality_score)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Reviewer matching scores
CREATE TABLE IF NOT EXISTS reviewer_matching_scores (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    submission_id INT NOT NULL,
    reviewer_id INT NOT NULL,
    match_score DECIMAL(3,2),
    expertise_match DECIMAL(3,2),
    availability_score DECIMAL(3,2),
    workload_score DECIMAL(3,2),
    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY unique_match (submission_id, reviewer_id),
    INDEX idx_submission_id (submission_id),
    INDEX idx_reviewer_id (reviewer_id),
    INDEX idx_match_score (match_score)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Semantic embeddings cache
CREATE TABLE IF NOT EXISTS semantic_embeddings (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    content_type VARCHAR(50) NOT NULL,
    content_id INT NOT NULL,
    embedding_model VARCHAR(100),
    embedding_vector JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY unique_embedding (content_type, content_id, embedding_model),
    INDEX idx_content_type (content_type),
    INDEX idx_content_id (content_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Workflow automation state
CREATE TABLE IF NOT EXISTS workflow_automation_state (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    submission_id INT NOT NULL UNIQUE,
    current_stage VARCHAR(50),
    automation_enabled BOOLEAN DEFAULT TRUE,
    last_agent_action VARCHAR(100),
    next_scheduled_action VARCHAR(100),
    next_action_time TIMESTAMP NULL,
    state_data JSON,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_submission_id (submission_id),
    INDEX idx_current_stage (current_stage),
    INDEX idx_next_action_time (next_action_time)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Performance metrics
CREATE TABLE IF NOT EXISTS performance_metrics (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(10,2),
    metric_labels JSON,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_metric_name (metric_name),
    INDEX idx_recorded_at (recorded_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

EOF

if [ $? -eq 0 ]; then
    log_success "Extension tables created successfully"
else
    log_error "Failed to create extension tables"
    exit 1
fi

# Create indexes on OJS tables for performance
log_info "Creating performance indexes..."

mysql -h "$DB_HOST" -P "$DB_PORT" -u "$DB_USER" -p"$DB_PASSWORD" "$DB_NAME" <<EOF

-- Add indexes if they don't exist
CREATE INDEX IF NOT EXISTS idx_submissions_status ON submissions(status);
CREATE INDEX IF NOT EXISTS idx_submissions_stage ON submissions(stage_id);
CREATE INDEX IF NOT EXISTS idx_submissions_date ON submissions(date_submitted);
CREATE INDEX IF NOT EXISTS idx_review_assignments_reviewer ON review_assignments(reviewer_id);
CREATE INDEX IF NOT EXISTS idx_review_assignments_submission ON review_assignments(submission_id);

EOF

if [ $? -eq 0 ]; then
    log_success "Performance indexes created"
else
    log_warning "Some indexes may already exist"
fi

# Create database views for agent access
log_info "Creating database views..."

mysql -h "$DB_HOST" -P "$DB_PORT" -u "$DB_USER" -p"$DB_PASSWORD" "$DB_NAME" <<EOF

-- View for pending submissions with full details
CREATE OR REPLACE VIEW v_pending_submissions AS
SELECT 
    s.submission_id,
    s.context_id,
    s.status,
    s.stage_id,
    s.date_submitted,
    ps.setting_value as title,
    ps2.setting_value as abstract,
    GROUP_CONCAT(CONCAT(a.first_name, ' ', a.last_name) SEPARATOR ', ') as authors
FROM submissions s
LEFT JOIN publication_settings ps ON s.current_publication_id = ps.publication_id 
    AND ps.setting_name = 'title'
LEFT JOIN publication_settings ps2 ON s.current_publication_id = ps2.publication_id 
    AND ps2.setting_name = 'abstract'
LEFT JOIN authors a ON a.publication_id = s.current_publication_id
WHERE s.status = 1
GROUP BY s.submission_id;

-- View for reviewer performance
CREATE OR REPLACE VIEW v_reviewer_performance AS
SELECT 
    u.user_id,
    u.username,
    u.email,
    CONCAT(u.first_name, ' ', u.last_name) as full_name,
    COUNT(ra.review_id) as total_reviews,
    SUM(CASE WHEN ra.date_completed IS NOT NULL THEN 1 ELSE 0 END) as completed_reviews,
    AVG(DATEDIFF(ra.date_completed, ra.date_assigned)) as avg_completion_days,
    MAX(ra.date_completed) as last_review_date
FROM users u
JOIN user_user_groups uug ON u.user_id = uug.user_id
JOIN user_groups ug ON uug.user_group_id = ug.user_group_id
LEFT JOIN review_assignments ra ON u.user_id = ra.reviewer_id
WHERE ug.role_id = 4096
GROUP BY u.user_id;

EOF

if [ $? -eq 0 ]; then
    log_success "Database views created"
else
    log_warning "Some views may already exist"
fi

# Test agent database access
log_info "Testing agent database access..."

python3.11 << 'PYTHON_EOF'
import sys
sys.path.insert(0, '/home/ubuntu/ojscog/skz-integration')

try:
    from ojs_database_integration import OJSIntegrationManager
    
    with OJSIntegrationManager() as manager:
        # Test manuscript retrieval
        manuscripts = manager.manuscript_repo.get_pending_submissions()
        print(f"✓ Successfully accessed {len(manuscripts)} pending submissions")
        
        # Test reviewer retrieval
        reviewers = manager.reviewer_repo.get_available_reviewers(1)
        print(f"✓ Successfully accessed {len(reviewers)} available reviewers")
        
    print("✓ Agent database access test passed")
    sys.exit(0)
    
except Exception as e:
    print(f"✗ Agent database access test failed: {e}")
    sys.exit(1)

PYTHON_EOF

if [ $? -eq 0 ]; then
    log_success "Agent database access verified"
else
    log_error "Agent database access test failed"
    exit 1
fi

# Summary
echo ""
echo "=========================================="
echo "  Database Setup Complete"
echo "=========================================="
echo ""
echo "Database: $DB_NAME@$DB_HOST:$DB_PORT"
echo ""
echo "Created Tables:"
echo "  - agent_execution_log"
echo "  - quality_assessment_cache"
echo "  - reviewer_matching_scores"
echo "  - semantic_embeddings"
echo "  - workflow_automation_state"
echo "  - performance_metrics"
echo ""
echo "Created Views:"
echo "  - v_pending_submissions"
echo "  - v_reviewer_performance"
echo ""
echo "=========================================="

log_success "Database setup completed successfully!"
