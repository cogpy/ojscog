# OJSCog Improvement Design Document

**Date**: November 15, 2025  
**Version**: 1.0  
**Purpose**: Design comprehensive improvements for autonomous research journal integration

---

## 1. Database Schema Design

### 1.1 Supabase Schema (Agent State Management)

The Supabase database will manage agent states, workflow tracking, and real-time coordination between the 7 autonomous agents.

#### Tables Structure

**agents_state**
```sql
CREATE TABLE agents_state (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id VARCHAR(50) UNIQUE NOT NULL,
    agent_name VARCHAR(100) NOT NULL,
    agent_type VARCHAR(50) NOT NULL,
    current_phase VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'idle',
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
```

**manuscript_workflows**
```sql
CREATE TABLE manuscript_workflows (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    manuscript_id VARCHAR(100) UNIQUE NOT NULL,
    ojs_submission_id INTEGER,
    current_stage VARCHAR(50) NOT NULL,
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
```

**cognitive_loop_executions**
```sql
CREATE TABLE cognitive_loop_executions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    execution_id VARCHAR(100) UNIQUE NOT NULL,
    manuscript_id VARCHAR(100) REFERENCES manuscript_workflows(manuscript_id),
    loop_iteration INTEGER NOT NULL,
    phase VARCHAR(20) NOT NULL,
    step_number INTEGER NOT NULL,
    step_name VARCHAR(100) NOT NULL,
    input_data JSONB DEFAULT '{}',
    output_data JSONB DEFAULT '{}',
    relevance_score DECIMAL(5,4),
    confidence DECIMAL(5,4),
    execution_time_ms INTEGER,
    status VARCHAR(20) NOT NULL,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_cognitive_loop_manuscript ON cognitive_loop_executions(manuscript_id);
CREATE INDEX idx_cognitive_loop_phase ON cognitive_loop_executions(phase);
CREATE INDEX idx_cognitive_loop_status ON cognitive_loop_executions(status);
```

**agent_communications**
```sql
CREATE TABLE agent_communications (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    communication_id VARCHAR(100) UNIQUE NOT NULL,
    sender_agent_id VARCHAR(50) REFERENCES agents_state(agent_id),
    receiver_agent_id VARCHAR(50) REFERENCES agents_state(agent_id),
    message_type VARCHAR(50) NOT NULL,
    message_content JSONB NOT NULL,
    priority INTEGER DEFAULT 5,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    response JSONB,
    sent_at TIMESTAMP DEFAULT NOW(),
    received_at TIMESTAMP,
    processed_at TIMESTAMP
);

CREATE INDEX idx_agent_comms_sender ON agent_communications(sender_agent_id);
CREATE INDEX idx_agent_comms_receiver ON agent_communications(receiver_agent_id);
CREATE INDEX idx_agent_comms_status ON agent_communications(status);
```

**performance_analytics**
```sql
CREATE TABLE performance_analytics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_id VARCHAR(100) UNIQUE NOT NULL,
    agent_id VARCHAR(50) REFERENCES agents_state(agent_id),
    manuscript_id VARCHAR(100),
    metric_type VARCHAR(50) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(10,4),
    metric_unit VARCHAR(20),
    metadata JSONB DEFAULT '{}',
    recorded_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_performance_agent ON performance_analytics(agent_id);
CREATE INDEX idx_performance_type ON performance_analytics(metric_type);
CREATE INDEX idx_performance_recorded ON performance_analytics(recorded_at);
```

### 1.2 Neon Database Schema (Hypergraph Dynamics)

The Neon database will implement hypergraph dynamics for tracking complex relationships between manuscripts, reviewers, agents, and knowledge entities.

#### Hypergraph Tables

**hypergraph_nodes**
```sql
CREATE TABLE hypergraph_nodes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    node_id VARCHAR(100) UNIQUE NOT NULL,
    node_type VARCHAR(50) NOT NULL,
    entity_id VARCHAR(100) NOT NULL,
    properties JSONB DEFAULT '{}',
    embedding VECTOR(768),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_hypergraph_nodes_type ON hypergraph_nodes(node_type);
CREATE INDEX idx_hypergraph_nodes_entity ON hypergraph_nodes(entity_id);
CREATE INDEX idx_hypergraph_nodes_embedding ON hypergraph_nodes USING ivfflat (embedding vector_cosine_ops);
```

**hypergraph_edges**
```sql
CREATE TABLE hypergraph_edges (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    edge_id VARCHAR(100) UNIQUE NOT NULL,
    edge_type VARCHAR(50) NOT NULL,
    source_node_id VARCHAR(100) REFERENCES hypergraph_nodes(node_id),
    target_node_id VARCHAR(100) REFERENCES hypergraph_nodes(node_id),
    weight DECIMAL(5,4) DEFAULT 1.0,
    properties JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_hypergraph_edges_type ON hypergraph_edges(edge_type);
CREATE INDEX idx_hypergraph_edges_source ON hypergraph_edges(source_node_id);
CREATE INDEX idx_hypergraph_edges_target ON hypergraph_edges(target_node_id);
```

**hypergraph_hyperedges**
```sql
CREATE TABLE hypergraph_hyperedges (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    hyperedge_id VARCHAR(100) UNIQUE NOT NULL,
    hyperedge_type VARCHAR(50) NOT NULL,
    node_ids JSONB NOT NULL,
    properties JSONB DEFAULT '{}',
    weight DECIMAL(5,4) DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_hypergraph_hyperedges_type ON hypergraph_hyperedges(hyperedge_type);
```

**knowledge_graph**
```sql
CREATE TABLE knowledge_graph (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    entity_id VARCHAR(100) UNIQUE NOT NULL,
    entity_type VARCHAR(50) NOT NULL,
    entity_name TEXT NOT NULL,
    description TEXT,
    attributes JSONB DEFAULT '{}',
    related_entities JSONB DEFAULT '[]',
    confidence_score DECIMAL(5,4),
    source VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_knowledge_graph_type ON knowledge_graph(entity_type);
CREATE INDEX idx_knowledge_graph_name ON knowledge_graph(entity_name);
```

---

## 2. Metamodel Expansion

### 2.1 Python-Scheme Bridge Implementation

Create a Foreign Function Interface (FFI) to connect Python agents with the Scheme metamodel.

**Architecture:**
- Python wrapper module: `metamodel_bridge.py`
- Guile Scheme subprocess communication
- JSON-based message passing
- Asynchronous execution support

**Key Functions:**
- `execute_cognitive_loop()` - Run 12-step cognitive loop
- `workflow_transition()` - Execute state transitions
- `compute_relevance()` - Calculate relevance realization
- `integrate_agent_results()` - Combine multi-agent outputs

### 2.2 Parallel Tensor Thread Fibers

Implement parallel processing capabilities for concurrent agent execution.

**Components:**
- Thread pool manager for parallel agent tasks
- Fiber synchronization primitives
- Load balancing across agents
- Result aggregation mechanisms

**Implementation:**
- Use Python `asyncio` for concurrent execution
- Implement work-stealing scheduler
- Add backpressure handling
- Monitor fiber health and performance

### 2.3 Ontogenetic Loom Placement

Design the ontogenetic loom system for optimal cognitive inference engine weaving.

**Loom Positions:**
1. **Input Loom** - Manuscript reception and parsing
2. **Quality Loom** - Assessment and validation weaving
3. **Coordination Loom** - Multi-agent orchestration
4. **Decision Loom** - Editorial decision synthesis
5. **Production Loom** - Publication preparation
6. **Learning Loom** - Continuous improvement weaving

**Weaving Patterns:**
- Serial weaving for sequential dependencies
- Parallel weaving for independent tasks
- Hierarchical weaving for nested workflows
- Recursive weaving for iterative refinement

---

## 3. Agent Coordination Enhancement

### 3.1 Inter-Agent Communication Protocol

**Message Types:**
- `REQUEST` - Agent requests action from another agent
- `RESPONSE` - Agent responds to request
- `NOTIFY` - Agent broadcasts status update
- `QUERY` - Agent queries for information
- `COMMAND` - Orchestrator commands agent action

**Protocol Specification:**
```python
{
    "message_id": "uuid",
    "message_type": "REQUEST|RESPONSE|NOTIFY|QUERY|COMMAND",
    "sender_agent_id": "agent_id",
    "receiver_agent_id": "agent_id",
    "timestamp": "iso8601",
    "priority": 1-10,
    "payload": {
        "action": "action_name",
        "parameters": {},
        "context": {}
    },
    "metadata": {
        "correlation_id": "uuid",
        "workflow_id": "uuid",
        "retry_count": 0
    }
}
```

### 3.2 Workflow Orchestration Layer

**Orchestrator Responsibilities:**
- Coordinate 7 agents across manuscript lifecycle
- Manage workflow state transitions
- Handle error recovery and retries
- Monitor agent health and performance
- Optimize task scheduling and resource allocation

**Orchestration Patterns:**
- Pipeline pattern for sequential stages
- Scatter-gather for parallel reviews
- Saga pattern for distributed transactions
- Circuit breaker for fault tolerance

### 3.3 Agent State Synchronization

**Synchronization Mechanisms:**
- Real-time state updates via WebSocket
- Event-driven state change notifications
- Distributed locking for critical sections
- Optimistic concurrency control
- Eventual consistency guarantees

---

## 4. OJS Integration Enhancement

### 4.1 PHP-Python API Bridge

**Bridge Architecture:**
- RESTful API gateway (Flask)
- PHP client library for OJS
- Request/response transformation
- Authentication token management
- Error handling and retry logic

**API Endpoints:**
```
POST /api/v1/manuscripts/submit
GET  /api/v1/manuscripts/{id}/status
POST /api/v1/manuscripts/{id}/assign-reviewers
POST /api/v1/manuscripts/{id}/editorial-decision
GET  /api/v1/agents/status
POST /api/v1/agents/{agent_id}/execute
GET  /api/v1/workflows/{manuscript_id}
POST /api/v1/cognitive-loop/execute
```

### 4.2 Real-time Event Streaming

**Event Types:**
- Manuscript submitted
- Quality assessment completed
- Reviewers assigned
- Reviews received
- Editorial decision made
- Production started
- Publication completed

**Implementation:**
- Server-Sent Events (SSE) for real-time updates
- Redis Pub/Sub for event distribution
- WebSocket for bidirectional communication
- Event sourcing for audit trail

### 4.3 Authentication Integration

**Authentication Flow:**
- OJS user authentication via JWT
- Agent service authentication via API keys
- Role-based access control (RBAC)
- Permission mapping between OJS and agents
- Session management and token refresh

---

## 5. Forensic Study and Component Mapping

### 5.1 Repository Component Analysis

**Component Categories:**
1. **Core Infrastructure** - OJS base, database, configuration
2. **Agent Framework** - 7 autonomous agents, microservices
3. **Metamodel** - Scheme implementation, cognitive architecture
4. **Integration Layer** - API bridges, communication protocols
5. **Frontend** - Dashboards, visualizations, UI components
6. **Documentation** - Technical docs, API specs, guides

### 5.2 MetaModel Mapping

**Mapping Framework:**

| Repository Component | MetaModel Element | Tensor Thread Type | Loom Position |
|---------------------|-------------------|-------------------|---------------|
| Research Discovery Agent | Input Processing | Serial | Input Loom |
| Submission Assistant | Quality Assessment | Serial | Quality Loom |
| Editorial Orchestration | Coordination Engine | Parallel | Coordination Loom |
| Review Coordination | Task Distribution | Parallel | Coordination Loom |
| Content Quality Agent | Validation Engine | Serial | Quality Loom |
| Publishing Production | Output Generation | Serial | Production Loom |
| Analytics Monitoring | Learning System | Parallel | Learning Loom |
| Cognitive Loop (Scheme) | Inference Engine | Serial + Parallel | All Looms |
| Workflow State Machine | State Transitions | Serial | Coordination Loom |
| Agent Communications | Message Passing | Parallel | All Looms |
| Hypergraph Dynamics | Knowledge Representation | Parallel | Learning Loom |

### 5.3 Cognitive Inference Engine Optimization

**Optimization Strategies:**
1. **Serial Optimization** - Minimize latency in sequential workflows
2. **Parallel Optimization** - Maximize throughput in concurrent tasks
3. **Load Balancing** - Distribute work evenly across agents
4. **Caching** - Reduce redundant computations
5. **Prefetching** - Anticipate future data needs
6. **Batch Processing** - Group similar operations
7. **Adaptive Scheduling** - Adjust priorities dynamically

---

## 6. Advanced Features Implementation

### 6.1 Hypergraph Dynamics Engine

**Capabilities:**
- Model complex multi-way relationships
- Track knowledge evolution over time
- Identify emerging patterns and trends
- Support semantic search and reasoning
- Enable graph-based machine learning

**Use Cases:**
- Reviewer-manuscript matching via graph embeddings
- Citation network analysis
- Research trend identification
- Expertise mapping
- Collaboration network discovery

### 6.2 Continuous Learning System

**Learning Components:**
- Online learning from editorial decisions
- Reinforcement learning for workflow optimization
- Transfer learning across manuscript types
- Meta-learning for agent adaptation
- Federated learning for privacy preservation

**Feedback Loops:**
- Decision quality assessment
- Reviewer performance tracking
- Manuscript outcome prediction
- Process efficiency metrics
- User satisfaction monitoring

### 6.3 Advanced Analytics Dashboard

**Analytics Features:**
- Real-time performance monitoring
- Predictive analytics for manuscript outcomes
- Trend analysis and forecasting
- Comparative benchmarking
- Anomaly detection and alerting

**Visualizations:**
- Workflow state diagrams
- Agent activity heatmaps
- Performance trend charts
- Network graphs for relationships
- Interactive dashboards

---

## 7. Implementation Roadmap

### Phase 1: Database Setup (Priority: HIGH)
- Create Supabase tables and indexes
- Set up Neon hypergraph database
- Implement database migration scripts
- Create seed data for testing

### Phase 2: Metamodel Enhancement (Priority: HIGH)
- Build Python-Scheme bridge
- Implement parallel tensor thread fibers
- Design ontogenetic loom system
- Create forensic study document

### Phase 3: Agent Coordination (Priority: MEDIUM)
- Implement inter-agent communication protocol
- Build workflow orchestration layer
- Add agent state synchronization
- Create monitoring and health checks

### Phase 4: OJS Integration (Priority: MEDIUM)
- Enhance PHP-Python API bridge
- Implement real-time event streaming
- Integrate authentication systems
- Add comprehensive error handling

### Phase 5: Advanced Features (Priority: LOW)
- Build hypergraph dynamics engine
- Implement continuous learning system
- Create advanced analytics dashboard
- Add predictive capabilities

### Phase 6: Testing and Optimization (Priority: HIGH)
- Comprehensive integration testing
- Performance benchmarking
- Security auditing
- Documentation updates

---

## 8. Success Metrics

**Technical Metrics:**
- API response time < 200ms (p95)
- Cognitive loop execution < 5 seconds
- Agent availability > 99.9%
- Database query time < 50ms (p95)
- Event processing latency < 100ms

**Business Metrics:**
- Manuscript processing time reduction > 70%
- Editorial decision quality improvement > 50%
- Reviewer assignment accuracy > 90%
- User satisfaction score > 4.5/5
- System adoption rate > 80%

**Research Metrics:**
- Publication throughput increase > 100%
- Review turnaround time reduction > 60%
- Quality score improvement > 40%
- Rejection rate optimization
- Citation impact prediction accuracy > 75%

---

## Conclusion

This improvement design provides a comprehensive roadmap for evolving the OJSCog repository toward a fully autonomous research journal system. The design emphasizes the integration of OJS workflows with the 7 autonomous agents, supported by robust database schemas, an expanded metamodel, and advanced features like hypergraph dynamics and continuous learning. The forensic study framework ensures optimal placement of cognitive inference engines through proper tensor thread fiber implementation and ontogenetic loom positioning.
