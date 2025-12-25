#!/bin/bash
#
# Production Deployment Script for OJS Cognitive Enhancement
# Deploys the enhanced autonomous agents system to ARM64 production environment
#
# Usage: ./deploy_production.sh [--env production|staging] [--skip-models] [--skip-db]
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEPLOYMENT_ENV="${DEPLOYMENT_ENV:-production}"
SKIP_MODELS=false
SKIP_DB=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --env)
            DEPLOYMENT_ENV="$2"
            shift 2
            ;;
        --skip-models)
            SKIP_MODELS=true
            shift
            ;;
        --skip-db)
            SKIP_DB=true
            shift
            ;;
        --help)
            echo "Usage: $0 [--env production|staging] [--skip-models] [--skip-db]"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Logging functions
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

# Check if running on ARM64
check_architecture() {
    log_info "Checking system architecture..."
    ARCH=$(uname -m)
    
    if [[ "$ARCH" == "aarch64" ]] || [[ "$ARCH" == "arm64" ]]; then
        log_success "Running on ARM64 architecture: $ARCH"
        return 0
    else
        log_warning "Not running on ARM64 (detected: $ARCH)"
        log_warning "Native libraries will not function properly"
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# Check system requirements
check_requirements() {
    log_info "Checking system requirements..."
    
    # Check Python version
    if ! command -v python3.11 &> /dev/null; then
        log_error "Python 3.11+ is required but not found"
        exit 1
    fi
    log_success "Python 3.11+ found"
    
    # Check required system libraries
    REQUIRED_LIBS=("libgomp1" "libopenblas0")
    for lib in "${REQUIRED_LIBS[@]}"; do
        if ! dpkg -l | grep -q "$lib"; then
            log_warning "$lib not found, attempting to install..."
            sudo apt-get update
            sudo apt-get install -y "$lib"
        fi
    done
    log_success "Required system libraries present"
    
    # Check disk space (need at least 10GB for models)
    AVAILABLE_SPACE=$(df -BG "$PROJECT_ROOT" | awk 'NR==2 {print $4}' | sed 's/G//')
    if [ "$AVAILABLE_SPACE" -lt 10 ]; then
        log_error "Insufficient disk space. Need at least 10GB, have ${AVAILABLE_SPACE}GB"
        exit 1
    fi
    log_success "Sufficient disk space: ${AVAILABLE_SPACE}GB available"
    
    # Check memory (need at least 8GB)
    TOTAL_MEM=$(free -g | awk 'NR==2 {print $2}')
    if [ "$TOTAL_MEM" -lt 8 ]; then
        log_warning "Low memory detected: ${TOTAL_MEM}GB (recommended: 8GB+)"
    else
        log_success "Sufficient memory: ${TOTAL_MEM}GB"
    fi
}

# Create directory structure
create_directories() {
    log_info "Creating directory structure..."
    
    mkdir -p "$PROJECT_ROOT/models"
    mkdir -p "$PROJECT_ROOT/logs"
    mkdir -p "$PROJECT_ROOT/data/cache"
    mkdir -p "$PROJECT_ROOT/data/embeddings"
    mkdir -p "$PROJECT_ROOT/config"
    mkdir -p "$PROJECT_ROOT/backups"
    
    log_success "Directory structure created"
}

# Install Python dependencies
install_dependencies() {
    log_info "Installing Python dependencies..."
    
    cd "$PROJECT_ROOT"
    
    if [ -f "requirements.txt" ]; then
        python3.11 -m pip install --upgrade pip
        python3.11 -m pip install -r requirements.txt
        log_success "Python dependencies installed"
    else
        log_warning "requirements.txt not found, skipping"
    fi
}

# Download and setup models
setup_models() {
    if [ "$SKIP_MODELS" = true ]; then
        log_warning "Skipping model download (--skip-models flag)"
        return 0
    fi
    
    log_info "Setting up AI models..."
    
    # Run model download script
    if [ -f "$SCRIPT_DIR/download_models.sh" ]; then
        bash "$SCRIPT_DIR/download_models.sh"
        log_success "Models downloaded and configured"
    else
        log_warning "Model download script not found, skipping"
    fi
}

# Setup database integration
setup_database() {
    if [ "$SKIP_DB" = true ]; then
        log_warning "Skipping database setup (--skip-db flag)"
        return 0
    fi
    
    log_info "Setting up database integration..."
    
    # Run database setup script
    if [ -f "$SCRIPT_DIR/setup_database.sh" ]; then
        bash "$SCRIPT_DIR/setup_database.sh"
        log_success "Database integration configured"
    else
        log_warning "Database setup script not found, skipping"
    fi
}

# Configure environment
configure_environment() {
    log_info "Configuring environment for $DEPLOYMENT_ENV..."
    
    # Create environment file
    ENV_FILE="$PROJECT_ROOT/.env.$DEPLOYMENT_ENV"
    
    if [ ! -f "$ENV_FILE" ]; then
        log_info "Creating environment configuration..."
        
        cat > "$ENV_FILE" << EOF
# OJS Cognitive Enhancement Environment Configuration
# Environment: $DEPLOYMENT_ENV
# Generated: $(date)

# Deployment
DEPLOYMENT_ENV=$DEPLOYMENT_ENV
PROJECT_ROOT=$PROJECT_ROOT

# Model Paths
LLM_MODEL_PATH=$PROJECT_ROOT/models/llama-7b-q4.gguf
VISION_MODEL_PATH=$PROJECT_ROOT/models/stable-diffusion
EMBEDDING_MODEL_PATH=$PROJECT_ROOT/models/embeddings

# Performance Settings
LLM_NUM_THREADS=4
LLM_CONTEXT_LENGTH=2048
VISION_USE_GPU=false
SPEECH_SAMPLE_RATE=22050

# Backend Selection
LLM_BACKEND=ggml-cpu
VISION_BACKEND=ncnn
SPEECH_BACKEND=kaldi

# Database Configuration
OJS_DB_HOST=localhost
OJS_DB_PORT=3306
OJS_DB_NAME=ojs
OJS_DB_USER=ojs_user
# OJS_DB_PASSWORD=  # Set this securely

# Cache Configuration
CACHE_DIR=$PROJECT_ROOT/data/cache
CACHE_SIZE_MB=1024
ENABLE_CACHE=true

# Logging
LOG_LEVEL=INFO
LOG_DIR=$PROJECT_ROOT/logs
LOG_ROTATION=daily
LOG_RETENTION_DAYS=30

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
ENABLE_TRACING=false

# Security
ENABLE_AUTH=true
JWT_SECRET_KEY=  # Generate secure key
API_RATE_LIMIT=100

# Feature Flags
ENABLE_VOICE_COMMANDS=true
ENABLE_AUDIO_NOTIFICATIONS=true
ENABLE_FIGURE_GENERATION=true
ENABLE_SEMANTIC_SEARCH=true
ENABLE_AUTO_REVIEWER_MATCHING=true
EOF
        
        log_success "Environment configuration created: $ENV_FILE"
        log_warning "Please review and update sensitive values (DB password, JWT secret, etc.)"
    else
        log_info "Environment file already exists: $ENV_FILE"
    fi
    
    # Create symlink to active environment
    ln -sf "$ENV_FILE" "$PROJECT_ROOT/.env"
    log_success "Active environment: $DEPLOYMENT_ENV"
}

# Setup systemd services
setup_services() {
    log_info "Setting up systemd services..."
    
    # Create service file
    SERVICE_FILE="/tmp/ojscog-agents.service"
    
    cat > "$SERVICE_FILE" << EOF
[Unit]
Description=OJS Cognitive Enhancement - Autonomous Agents
After=network.target mysql.service

[Service]
Type=simple
User=$USER
WorkingDirectory=$PROJECT_ROOT
EnvironmentFile=$PROJECT_ROOT/.env
ExecStart=/usr/bin/python3.11 $PROJECT_ROOT/skz-integration/autonomous_workflow_orchestrator.py
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
    
    log_info "Service file created: $SERVICE_FILE"
    log_info "To install: sudo cp $SERVICE_FILE /etc/systemd/system/"
    log_info "To enable: sudo systemctl enable ojscog-agents"
    log_info "To start: sudo systemctl start ojscog-agents"
}

# Run tests
run_tests() {
    log_info "Running integration tests..."
    
    cd "$PROJECT_ROOT"
    
    if [ -f "scripts/testing/test_enhanced_integration.py" ]; then
        python3.11 scripts/testing/test_enhanced_integration.py
        
        if [ $? -eq 0 ]; then
            log_success "All tests passed"
        else
            log_error "Tests failed"
            exit 1
        fi
    else
        log_warning "Test script not found, skipping"
    fi
}

# Create backup
create_backup() {
    log_info "Creating backup..."
    
    BACKUP_DIR="$PROJECT_ROOT/backups"
    BACKUP_FILE="$BACKUP_DIR/ojscog-backup-$(date +%Y%m%d-%H%M%S).tar.gz"
    
    tar -czf "$BACKUP_FILE" \
        --exclude='*.pyc' \
        --exclude='__pycache__' \
        --exclude='node_modules' \
        --exclude='models' \
        --exclude='logs' \
        --exclude='backups' \
        -C "$PROJECT_ROOT" .
    
    log_success "Backup created: $BACKUP_FILE"
}

# Print deployment summary
print_summary() {
    echo ""
    echo "=========================================="
    echo "  Deployment Summary"
    echo "=========================================="
    echo ""
    echo "Environment: $DEPLOYMENT_ENV"
    echo "Project Root: $PROJECT_ROOT"
    echo "Architecture: $(uname -m)"
    echo ""
    echo "Next Steps:"
    echo "1. Review environment configuration: $PROJECT_ROOT/.env"
    echo "2. Set sensitive values (DB password, JWT secret)"
    echo "3. Install systemd service: sudo cp /tmp/ojscog-agents.service /etc/systemd/system/"
    echo "4. Enable service: sudo systemctl enable ojscog-agents"
    echo "5. Start service: sudo systemctl start ojscog-agents"
    echo "6. Check status: sudo systemctl status ojscog-agents"
    echo "7. View logs: sudo journalctl -u ojscog-agents -f"
    echo ""
    echo "Monitoring:"
    echo "- Metrics endpoint: http://localhost:9090/metrics"
    echo "- Logs directory: $PROJECT_ROOT/logs"
    echo ""
    echo "=========================================="
}

# Main deployment flow
main() {
    log_info "Starting OJS Cognitive Enhancement Deployment"
    log_info "Environment: $DEPLOYMENT_ENV"
    echo ""
    
    check_architecture
    check_requirements
    create_directories
    install_dependencies
    setup_models
    setup_database
    configure_environment
    setup_services
    run_tests
    create_backup
    
    echo ""
    log_success "Deployment completed successfully!"
    print_summary
}

# Run main function
main
