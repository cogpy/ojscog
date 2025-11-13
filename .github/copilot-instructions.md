# Enhanced OJS with SKZ Autonomous Agents

**OJS + 7 Autonomous AI Agents** for academic publishing workflow automation. This is a polyglot system combining PHP (OJS core), Python (AI agents), and React (dashboards).

**Always reference these instructions first** - fallback to search/bash only when you encounter unexpected information.

## 🚀 Quick Start for AI Agents

**Three Critical Non-Negotiables**:
1. **NEVER CANCEL long-running commands** - Composer: 35+ min, Tests: 15+ min, set 60+ min timeouts
2. **ZERO tolerance for mock/placeholder AI** - Use llama.cpp/BERT/transformers, quantized models for fallbacks
3. **Auto-commit enabled** - All changes committed via `auto_commit.py` (per `.windsurfrules`)

**Get Productive Immediately**:
```bash
# First-time setup (takes 35+ minutes for Composer - NEVER CANCEL)
cp config.TEMPLATE.inc.php config.inc.php
cp skz-integration/.env.template skz-integration/.env
composer --working-dir=lib/pkp install --no-dev  # Answer 'y' to trust prompt

# Start services (3 terminals)
php -S localhost:8000  # Terminal 1: OJS Core
cd skz-integration/autonomous-agents-framework && source venv/bin/activate && python src/main.py  # Terminal 2: Agents
./skz-integration/scripts/health-check.sh  # Terminal 3: Verify all running

# Before committing changes
./lib/pkp/tools/runAllTests.sh  # 15+ min - NEVER CANCEL
curl http://localhost:5000/api/v1/agents  # Verify agents respond
```

**Key Architecture Pattern** - OJS PHP Plugin → Flask API Gateway → 7 Agent Microservices → MySQL

## 🎯 Project Overview

This is a **polyglot academic publishing system** combining:
- **OJS Core (PHP)**: Traditional open journal system at `/index.php`, `/classes/`, `/lib/pkp/`
- **7 Autonomous Agents (Python/Flask)**: AI-powered workflow automation at `skz-integration/autonomous-agents-framework/`
- **PHP-Python Bridge**: OJS plugin at `plugins/generic/skzAgents/` that connects PHP to Python services
- **React Dashboards**: Workflow visualization at `skz-integration/workflow-visualization-dashboard/`
- **MySQL Database**: Extended with agent state tables (see `skz-integration/schema/skz-agents.sql`)

### Key Integration Points
1. **OJS Plugin → Python Agent API**: PHP plugin makes HTTP calls to Flask endpoints (ports 5000-5007)
2. **Database Sharing**: Both OJS and agents share MySQL database with separate table namespaces
3. **Event-Driven Communication**: Agents communicate via JSON messages through API gateway pattern
4. **Environment Toggle**: `USE_PROVIDER_IMPLEMENTATIONS=true/false` switches production/dev modes

### Auto-Commit System
**IMPORTANT**: This repository has an auto-commit system (`auto_commit.py`) that ensures all changes are committed to GitHub automatically. From `.windsurfrules`:
> "The first objective is to ensure source control automatically commits to the github repo after each run. The commit must go through no matter what!"

When making changes, verify they're committed by checking git status.

## Working Effectively

### Environment & Dependencies Verified

**Current Environment (Dev Container - Ubuntu 24.04.3 LTS)**:
- Python: 3.12.3 (requirement: 3.11+) ✅
- Node.js: 20.19.4 (requirement: 18+) ✅  
- npm: 10.8.2 (requirement: 8+) ✅
- PHP: 8.3.6 (requirement: 7.4+) ✅
- MySQL: 8.0.42 (requirement: 5.7+) ✅
- Composer: 2.8.10 ✅

### Critical File Locations
```
/workspaces/ojs/
├── index.php                           # OJS entry point
├── config.inc.php                      # Main config (copy from config.TEMPLATE.inc.php)
├── lib/pkp/                           # Core OJS framework (DO NOT MODIFY)
├── plugins/generic/skzAgents/          # PHP-Python bridge plugin
│   ├── SKZAgentsPlugin.inc.php        # Main plugin registration
│   ├── classes/SKZAgentBridge.inc.php # Agent communication bridge
│   └── classes/SKZAPIGateway.inc.php  # API routing & auth
├── skz-integration/
│   ├── .env.template                   # Environment config template
│   ├── autonomous-agents-framework/    # Main Python agent services
│   │   ├── src/main.py                # Flask app entry (USE THIS, not main_simple.py)
│   │   ├── src/models/                # Agent models & database schemas
│   │   ├── src/routes/                # API endpoints
│   │   ├── src/services/              # ML, communication, decision engines
│   │   └── src/providers/             # External integrations (SendGrid, AWS)
│   ├── seven_agents.py                # Core agent system definitions
│   ├── schema/skz-agents.sql          # Database extensions for agents
│   └── scripts/health-check.sh        # System validation script
└── deploy-skz-integration.sh          # Automated deployment
```

### Bootstrap, Build, and Test the Repository

**CRITICAL: NEVER CANCEL any build or long-running command. Builds take 35+ minutes. Set timeouts to 60+ minutes minimum.**

1. **Initial Setup:**
   ```bash
   # Git submodules (takes < 1 second - very fast)
#   NEVER USE SUBMODULES!!!
   
   # Configuration setup (only if config.inc.php doesn't exist)
   cp config.TEMPLATE.inc.php config.inc.php
   ```

2. **PHP Dependencies (CRITICAL: Takes 35+ minutes, NEVER CANCEL):**
   ```bash
   # Main OJS dependencies - TAKES 35+ MINUTES, SET TIMEOUT 60+ MINUTES
   composer --working-dir=lib/pkp install --no-dev
   # Will prompt: "Do you trust cweagans/composer-patches?" - Answer: y
   
   # Plugin dependencies (only if needed)
   composer --working-dir=plugins/paymethod/paypal install
   composer --working-dir=plugins/generic/citationStyleLanguage install
   ```

3. **Python Agent Framework Setup:**
   ```bash
   # Autonomous agents framework (takes ~3 seconds for venv, ~53 seconds for deps)
   cd skz-integration/autonomous-agents-framework
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   
   # Skin zone journal backend (may fail due to network timeouts)
   cd ../skin-zone-journal
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt  # May timeout - retry if needed
   ```

4. **Node.js Dashboard Setup:**
   ```bash
   # Workflow visualization dashboard (takes ~1.5 minutes)
   cd skz-integration/workflow-visualization-dashboard
   npm install --legacy-peer-deps  # REQUIRED: --legacy-peer-deps due to date-fns conflicts
   npm run build  # Takes ~6 seconds
   
   # Simulation dashboard (takes ~5 seconds install, ~7 seconds build)
   cd ../simulation-dashboard
   npm install --legacy-peer-deps  # REQUIRED: --legacy-peer-deps due to dependency conflicts
   npm run build
   ```

### Running Tests

**CRITICAL: Test commands can take 15+ minutes. NEVER CANCEL. Set timeout to 30+ minutes.**

```bash
# OJS core tests - TAKES 15+ MINUTES, NEVER CANCEL
./lib/pkp/tools/runAllTests.sh

# Individual test categories (faster alternatives):
./lib/pkp/tools/runAllTests.sh -C  # PKP class tests only
./lib/pkp/tools/runAllTests.sh -c  # App class tests only

# SKZ Agent tests
cd skz-integration/autonomous-agents-framework
source venv/bin/activate
python -m pytest tests/
```

### Running the Application

1. **OJS Core System:**
   ```bash
   # Simple development server
   php -S localhost:8000
   ```

2. **SKZ Agent Services:**
   ```bash
   # Agent framework (Terminal 1) - Use main_simple.py to avoid import issues
   cd skz-integration/autonomous-agents-framework
   source venv/bin/activate
   python src/main.py  # NEVER SACRIFICE QUALITY!! NO MOCK/SIMPLIFID/PLACEHOLDER EVER!
   
   # Skin zone journal backend (Terminal 2) - May have import issues
   cd skz-integration/skin-zone-journal
   source venv/bin/activate
   python src/main.py  # May fail due to Python import paths
   
   # Serve visualization dashboards (Terminal 3)
   cd skz-integration/workflow-visualization-dashboard
   pnpm run build && pnpm run preview  # BUILD CORRECTLY
   ```

3. **Health Checks:**
   ```bash
   # System health verification (script created and validated)
   ./skz-integration/scripts/health-check.sh
   
   # Check individual services
   curl http://localhost:5000/api/v1/agents  # Agent framework (works)
   curl http://localhost:5001/api/status     # Skin zone journal (may not work)
   curl http://localhost:8000                # OJS core (works)
   ```

## Validation Scenarios

**ALWAYS test these scenarios after making changes:**

1. **OJS Core Validation:**
   - Start PHP server: `php -S localhost:8000`
   - Access http://localhost:8000 and verify OJS loads
   - Check admin panel accessibility

2. **Agent Framework Validation:**
   - Start agent services: `python src/main.py` NEVER DEGRADE THE CODEBASE WITH MOCK/SIMPLE!!
   - Verify API endpoints respond: `curl http://localhost:5000/api/v1/agents`
   - Check web dashboard at http://localhost:5000
   - Verify JSON response contains 7 active agents

3. **Complete Workflow Test:**
   - Run health check script: `./skz-integration/scripts/health-check.sh`
   - Verify all components show as ✅ (green checkmarks)
   - Check dashboards are built in dist/ directories
   - Verify Python virtual environments exist

## Build Time Expectations

**CRITICAL TIMING INFORMATION - NEVER CANCEL THESE:**

- **Composer PHP install:** 35+ minutes (NEVER CANCEL - SET 60+ MINUTE TIMEOUT)
- **Python venv creation:** ~3 seconds per environment (much faster than expected)
- **Python pip install:** ~53 seconds per requirements.txt (may timeout due to network issues)
- **npm install (with --legacy-peer-deps):** ~1.5 minutes for workflow dashboard, ~5 seconds for simulation dashboard
- **npm run build:** ~6 seconds per dashboard (much faster than expected)
- **Test suite:** 15+ minutes (NEVER CANCEL - SET 30+ MINUTE TIMEOUT)

## Common Issues and Workarounds

1. **npm install fails with ERESOLVE errors:**
   - **SOLUTION:** Always use `npm install --legacy-peer-deps`
   - **CAUSE:** date-fns version conflicts in react-day-picker

2. **Composer plugin trust prompt:**
   - **EXPECTED:** Composer will ask to trust "cweagans/composer-patches"
   - **SOLUTION:** Answer "y" when prompted

3. **Long Composer installs:**
   - **NORMAL:** 35+ minutes for full dependency installation
   - **NEVER CANCEL:** This is expected behavior, not a hang

4. **Python import errors in agent framework:**
   - **SOLUTION:** Use `python src/main.py` AND FIX THE ERROR! NEVER SACRIFICE QUALITY FOR ANY REASON!!
   - **CAUSE:** Complex import paths in main.py cause ModuleNotFoundError

5. **Network timeouts during pip install:**
   - **SOLUTION:** Retry the pip install command if it times out
   - **CAUSE:** Network connectivity issues to PyPI servers

6. **Missing health check script:**
   - **SOLUTION:** The script is now created at `skz-integration/scripts/health-check.sh`
   - **VALIDATION:** Script has been tested and works correctly

## Key Project Structure

```
/
├── index.php                    # OJS main entry point
├── config.inc.php              # Main OJS configuration (copy from config.TEMPLATE.inc.php)
├── classes/                    # OJS core classes
├── lib/pkp/                   # PKP library (core OJS framework)
├── plugins/                   # OJS plugins
│   └── generic/skzAgents/     # OJS-SKZ bridge plugin (PHP integration)
├── skz-integration/           # SKZ autonomous agents framework
│   ├── .env.template          # Environment configuration template
│   ├── autonomous-agents-framework/    # Main agent services (Flask + Python)
│   │   ├── src/
│   │   │   ├── main.py        # Main Flask application entry point
│   │   │   ├── routes/        # API route handlers
│   │   │   ├── models/        # AI models and agent logic
│   │   │   ├── services/      # Communication, ML decision engines
│   │   │   └── providers/     # External service integrations (SendGrid, AWS SES)
│   │   ├── venv/              # Python virtual environment (create with python3 -m venv venv)
│   │   └── requirements.txt   # Python dependencies (includes torch, transformers, llama-cpp)
│   ├── skin-zone-journal/     # Specialized journal backend
│   ├── workflow-visualization-dashboard/  # React dashboard (Node.js)
│   ├── simulation-dashboard/  # Agent simulation interface (Node.js)
│   ├── microservices/         # Individual agent microservices (7 agents + gateway)
│   ├── schema/                # Database schema extensions
│   │   └── skz-agents.sql     # Agent state, communications, metrics tables
│   └── scripts/               # Deployment and health check utilities
└── deploy-skz-integration.sh   # Automated deployment script
```

## Essential Commands Reference

### Development Workflow
```bash
# Quick development setup
git submodule update --init --recursive
cp config.TEMPLATE.inc.php config.inc.php
composer --working-dir=lib/pkp install --no-dev  # LONG: 35+ min, NEVER CANCEL

# Start development environment
php -S localhost:8000  # OJS core
# In separate terminals:
cd skz-integration/autonomous-agents-framework && source venv/bin/activate && python src/main.py
cd skz-integration/skin-zone-journal && source venv/bin/activate && python src/main.py
```

### Before Committing Changes
```bash
# ALWAYS run these validation steps:
./lib/pkp/tools/runAllTests.sh  # LONG: 15+ min, NEVER CANCEL
./skz-integration/scripts/health-check.sh
curl http://localhost:5000/api/v1/agents  # Verify agent integration works
```

## Database Requirements

- **MySQL 5.7+ or 8.0+** (confirmed available)
- **Run schema manually:** `mysql -u [user] -p [database] < skz-integration/schema/skz-agents.sql`
- **Additional tables:** 
  - `skz_agent_states` - Agent state tracking and persistence
  - `skz_agent_communications` - Inter-agent communication logs
  - `skz_workflow_automation` - Workflow automation tracking
  - `skz_agent_metrics` - Performance metrics (efficiency, accuracy, response_time)
  - `skz_agent_settings` - Agent configuration settings

## Architecture Patterns

### The 7 Autonomous Agents System

The SKZ framework implements 7 specialized agents coordinated through a hierarchical/distributed hybrid architecture:

**Agent Roster** (defined in `skz-integration/seven_agents.py`):
1. **Research Discovery Agent** (Port 5001) - INCI database mining, patent analysis, trend identification
2. **Submission Assistant Agent** (Port 5002) - Manuscript validation, quality assessment, compliance checks  
3. **Editorial Orchestration Agent** (Port 5003) - Central coordinator, workflow management, decision support
4. **Review Coordination Agent** (Port 5004) - Reviewer matching, workload balancing, quality monitoring
5. **Content Quality Agent** (Port 5005) - Scientific validation, safety assessment, standards enforcement
6. **Publishing Production Agent** (Port 5006) - Formatting, distribution, regulatory reporting
7. **Analytics Monitoring Agent** (Port 5007) - Performance tracking, anomaly detection, optimization

**Agent Communication Flow**:
```
OJS PHP Plugin (plugins/generic/skzAgents/)
    ↓ HTTP POST to localhost:5000
Flask API Gateway (src/routes/manuscript_automation_api.py)
    ↓ Route to specific agent endpoint
Individual Agent (src/models/seven_agents.py - AgentType enum)
    ↓ Process with AI inference (NEVER MOCK - see AI Guidelines below)
Agent State Update (skz_agent_states table in MySQL)
    ↓ Return structured JSON response
OJS receives result and updates workflow
```

**Agent Action Pattern** (all agents implement this):
```python
def execute_action(self, action_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Standard agent action signature - NEVER use mocks"""
    # 1. Validate action type against agent capabilities
    # 2. Process with real AI inference (llama.cpp, BERT, transformers)
    # 3. Update agent state in database
    # 4. Log communication to skz_agent_communications table
    return {
        "success": bool,
        "result": Any,
        "agent_state": Dict,
        "performance_metrics": Dict
    }
```

### Multi-Layer Architecture
1. **OJS Core (PHP)** - Traditional academic publishing system
2. **SKZ Plugin Layer (PHP)** - Bridge between OJS and agents (`plugins/generic/skzAgents/`)
   - `SKZAgentBridge.inc.php` - Main integration bridge
   - `ManuscriptAutomationBridge.inc.php` - Manuscript workflow automation
   - `SKZAPIGateway.inc.php` - API routing and authentication
3. **Agent Framework (Python)** - 7 autonomous agents with Flask APIs
4. **External Services** - Email (SendGrid/AWS SES), AI models (llama.cpp, BERT)

### Agent Communication Protocol
- **REST API** - All agents expose HTTP endpoints (ports 5000-5007)
- **API Gateway Pattern** - Central routing via `api-gateway/` microservice
- **Event-Driven** - Agents communicate via JSON messages with action types
- **State Management** - Agent states persisted in MySQL via `skz_agent_states` table

### Provider-Backed Implementations
Toggle production vs. development mode via environment variable:
```bash
USE_PROVIDER_IMPLEMENTATIONS=true  # Production: real SendGrid, AWS SES, AI models
USE_PROVIDER_IMPLEMENTATIONS=false # Development: local implementations
```

Located in `skz-integration/autonomous-agents-framework/src/providers/factory.py`

## Production Deployment

Use the automated deployment script:
```bash
chmod +x deploy-skz-integration.sh
./deploy-skz-integration.sh
```

This script:
- Sets up all Python virtual environments
- Installs all dependencies with proper error handling
- Creates systemd services for production
- Generates nginx configuration
- Creates Docker Compose setup
- Sets up monitoring and health check scripts

## Environment Dependencies Verified

- **Python:** 3.12.3 (requirement: 3.11+) ✅
- **Node.js:** 20.19.4 (requirement: 18+) ✅
- **npm:** 10.8.2 (requirement: 8+) ✅
- **PHP:** 8.3.6 (requirement: 7.4+) ✅
- **MySQL:** 8.0.42 (requirement: 5.7+) ✅
- **Composer:** 2.8.10 ✅

## AI Inference Engine Requirements

**PRODUCTION-GRADE AI INFERENCE MANDATORY - NO EXCEPTIONS**

### 🧠 Prompt Seed Reference
See [.github/copilot-prompt-seed.md](./copilot-prompt-seed.md) for foundational prompt context and detailed AI implementation examples.

### Core AI Inference Guidelines
- **PRIMARY ENGINE:** Use llama.cpp or node-llama-cpp as the default AI inference engine
- **ZERO MOCK TOLERANCE:** Never generate placeholder logic for AI agent functions
- **REAL INFERENCE ONLY:** If performance is degraded, use quantized models, NOT mocks
- **FALLBACK POLICY:** Fallback must be a functional inference engine (e.g., quantized llama.cpp), never a mock
- **PRODUCTION PRIORITY:** Prioritize inference correctness over performance during development

### AI Implementation Standards
- **BERT/Transformer Integration:** Use production BERT models for text classification
- **Model Deployment:** Implement proper model loading and caching infrastructure  
- **Inference Optimization:** Use ONNX runtime or similar for fast inference
- **Error Handling:** AI failures must degrade gracefully to simpler but real AI models
- **Configuration:** All AI models must be configurable via environment variables

### Prohibited AI Practices
- `return "TODO"` or similar placeholders in AI functions
- Mock classification/inference that returns dummy data
- Keyword-only fallbacks without real AI processing
- Simplified logic that bypasses actual model inference
- Development shortcuts that compromise AI functionality

## Testing & Validation Workflows

### Essential Test Commands

**OJS Core Tests** (TAKES 15+ MINUTES - NEVER CANCEL):
```bash
./lib/pkp/tools/runAllTests.sh           # Full test suite
./lib/pkp/tools/runAllTests.sh -C       # PKP class tests only (faster)
./lib/pkp/tools/runAllTests.sh -c       # App class tests only (faster)
```

**Agent Framework Tests**:
```bash
cd skz-integration/autonomous-agents-framework
source venv/bin/activate
python -m pytest tests/                  # Unit tests
python tests/test_phase2_integration.py  # Integration tests (requires services running)
python demo_manuscript_automation.py     # End-to-end workflow demo
```

**Health Check Validation**:
```bash
./skz-integration/scripts/health-check.sh  # Comprehensive system check
# Verifies:
# - OJS Core (http://localhost:8000)
# - Agent Framework (http://localhost:5000)
# - Dashboard builds (dist/ directories exist)
# - Python virtual environments
```

### Debugging Agent Issues

**Check Agent Logs**:
```bash
# Agent framework logs (Flask app output)
cd skz-integration/autonomous-agents-framework
source venv/bin/activate
python src/main.py  # Watch console for errors

# Check database agent state
mysql -u root -p -e "SELECT * FROM skz_agent_states ORDER BY last_updated DESC LIMIT 10;"

# Check agent communications
mysql -u root -p -e "SELECT * FROM skz_agent_communications WHERE success=0 ORDER BY timestamp DESC LIMIT 20;"
```

**Verify Agent Availability**:
```bash
# Test individual agent endpoints
curl -X POST http://localhost:5000/api/v1/agents/research_discovery/action \
  -H "Content-Type: application/json" \
  -d '{"action_type":"analyze_manuscript","data":{}}'

# List all active agents
curl http://localhost:5000/api/v1/agents | jq .
```

### Integration Test Scenarios

**ALWAYS test these after making changes**:

1. **OJS-Agent Bridge Test**:
   ```bash
   # Start OJS
   php -S localhost:8000 &
   # Start agents
   cd skz-integration/autonomous-agents-framework && source venv/bin/activate && python src/main.py &
   # Verify plugin loaded
   curl http://localhost:8000/index.php/index/plugins | grep -i "skz"
   ```

2. **Complete Workflow Test**:
   ```bash
   # Submit test manuscript through OJS UI
   # Monitor agent processing in logs
   # Verify workflow stages progress
   # Check agent state updates in database
   ```

3. **Dashboard Verification**:
   ```bash
   cd skz-integration/workflow-visualization-dashboard
   npm run build  # Should complete in ~6 seconds
   ls -la dist/   # Verify build artifacts exist
   ```

## Critical Reminders

- **NEVER CANCEL BUILDS:** Composer installs take 35+ minutes, tests take 15+ minutes
- **ALWAYS use --legacy-peer-deps** for npm installs to avoid dependency conflicts
- **SET LONG TIMEOUTS:** 60+ minutes for builds, 30+ minutes for tests
- **VALIDATE THOROUGHLY:** Always test complete workflows after changes
- **ANSWER PROMPTS:** Composer will ask to trust plugins - answer "y"
- **NEVER USE SIMPLE PYTHON:** Use `main.py` not simplified versions
- **NETWORK ISSUES:** Python pip installs may timeout - retry if needed
- **HEALTH CHECK:** Always run `./skz-integration/scripts/health-check.sh` to verify system state
- **AI INFERENCE:** Always verify AI models are loaded and functional before deployment

## Agent Development Patterns

### Adding a New Agent
1. Create microservice in `skz-integration/microservices/<agent-name>/`
2. Add to `AgentType` enum in `seven_agents.py`
3. Implement agent class extending base patterns
4. Register in API gateway routing
5. Add health check endpoint
6. Update database schema if needed

### Agent Action Pattern
```python
def execute_action(self, action_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """All agents follow this signature for action execution"""
    # Validate action type
    # Process data with AI inference (NEVER use mocks)
    # Update agent state
    # Return structured response
    return {"success": bool, "result": Any, "agent_state": Dict}
```

### Configuration Management
- Environment variables in `.env` (copy from `.env.template`)
- Agent-specific config in `skz_agent_settings` table
- Runtime config via Flask app config
- Secrets NEVER committed to git

### Testing Strategy
```bash
# Unit tests for individual agents
cd skz-integration/autonomous-agents-framework
python -m pytest tests/

# Integration tests (requires running services)
python tests/test_phase2_integration.py

# End-to-end workflow tests
python demo_manuscript_automation.py
```

## Project-Specific Conventions

### Git Workflow
- **Auto-commit enabled**: `auto_commit.py` ensures all changes are committed automatically
- **No submodules**: Despite git submodule commands in docs, NEVER USE submodules (per .windsurfrules)
- **Commit enforcement**: All commits must succeed - no type checks blocking commits

### Code Quality Standards
- **NEVER use simplified/mock implementations**: Always use production AI inference (llama.cpp, BERT)
- **AI fallbacks must be real**: If primary model fails, fallback to quantized models, NOT mocks
- **Import paths matter**: Always run Python from correct directory with activated venv
- **Main entry points**: Use `src/main.py` NOT `main_simple.py` for agent framework

### Configuration Patterns
- **Environment variables**: Copy `.env.template` to `.env` before first run
- **Config files**: Copy `config.TEMPLATE.inc.php` to `config.inc.php` for OJS
- **Provider implementations**: Toggle `USE_PROVIDER_IMPLEMENTATIONS=true` for production services
- **Database schema**: Manually run `skz-integration/schema/skz-agents.sql` after setup

### Development Gotchas
- **npm requires --legacy-peer-deps**: Due to date-fns version conflicts, always use this flag
- **Composer prompts are normal**: Answer "y" when asked to trust "cweagans/composer-patches"
- **Long builds are expected**: 35+ min for Composer, 15+ min for tests - NEVER CANCEL
- **Python timeouts may occur**: pip install can timeout due to network - just retry

### Service Ports
- **OJS Core**: 8000 (PHP built-in server)
- **Agent Framework**: 5000 (Flask API gateway)
- **Individual Agents**: 5001-5007 (each agent on separate port)
- **Skin Zone Journal**: 5001 (may conflict with Research Discovery Agent)
- **Dashboards**: Build to `dist/` directories, serve via separate server

## Common Error Resolution

### ModuleNotFoundError in Python
- **Cause:** Complex import paths in agent framework
- **Solution:** Always activate venv and run from correct directory
```bash
cd skz-integration/autonomous-agents-framework
source venv/bin/activate
python src/main.py  # Not python main.py
```

### Port Already in Use
- **Cause:** Previous agent instances still running
- **Solution:** Kill processes or use different ports
```bash
lsof -ti:5000 | xargs kill -9  # Kill process on port 5000
# Or change port in .env file
```

### Database Connection Errors
- **Cause:** MySQL not running or wrong credentials in .env
- **Solution:** Verify MySQL service and update DATABASE_* variables
```bash
mysql -u root -p  # Test connection
# Update .env with correct DATABASE_USER, DATABASE_PASSWORD
```

### AI Model Loading Failures
- **Cause:** Missing model files or insufficient memory
- **Solution:** Download models or use quantized versions
```bash
# Check model path exists
ls -lh ./models/*.gguf
# Use smaller quantized model if memory limited
export AI_MODEL_PATH="./models/llama-2-7b-chat.q4_0.gguf"
```

## Manuscript Processing Workflow

The system automates the complete manuscript lifecycle across 7 agents:

### Workflow Stages
1. **Submission** → Research Discovery Agent analyzes literature gaps
2. **Initial Review** → Submission Assistant validates format/compliance
3. **Editorial Assignment** → Editorial Orchestration routes to reviewers
4. **Peer Review** → Review Coordination manages reviewer matching
5. **Quality Check** → Content Quality Agent scores manuscript
6. **Production** → Publishing Production Agent formats output
7. **Analytics** → Analytics Monitoring Agent tracks metrics

### API Endpoint Pattern
All agent endpoints follow consistent structure:
```
POST /api/v1/agents/{agent_name}/action
{
  "action_type": "analyze_manuscript",
  "data": { ... },
  "context_id": "uuid",
  "priority": 1-3
}
```

### Automation Control API
```python
# Submit manuscript for automation
POST /api/v1/automation/submit
{
  "id": "manuscript_123",
  "title": "...",
  "authors": [...],
  "priority": 2  # AutomationPriority: ROUTINE=1, HIGH=2, URGENT=3
}

# Get workflow status
GET /api/v1/automation/status/{workflow_id}

# Control automation
POST /api/v1/automation/pause/{workflow_id}
POST /api/v1/automation/resume/{workflow_id}
```

Located in: `skz-integration/autonomous-agents-framework/src/routes/manuscript_automation_api.py`