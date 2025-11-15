# OJSCog Improvement Plan: Evolution Toward Autonomous Research Journal

**Date**: November 15, 2025  
**Repository**: cogpy/ojscog  
**Goal**: Integrate OJS workflows with 7 autonomous agents for fully autonomous research journal operation

---

## Executive Summary

This document outlines the comprehensive improvement plan for evolving the ojscog repository toward a fully autonomous research journal. The plan integrates cognitive architecture principles, MetaModel mapping, and the 7 specialized agents with OJS workflows to create a self-organizing, adaptive publishing system.

---

## Cognitive Architecture Framework

### MetaModel Mapping for Autonomous Journal System

Following the forensic study framework for MetaModel mapping, we map each component to cognitive inference engine elements:

#### 1. **Serial Tensor Thread Fibers** (Sequential Processing)
- **OJS Workflow Stages**: Submission → Review → Decision → Production → Publication
- **Agent Coordination**: Editorial Orchestration Agent manages sequential dependencies
- **Implementation**: State machine with deterministic transitions

#### 2. **Parallel Tensor Thread Fibers** (Concurrent Processing)
- **Multi-Agent Collaboration**: 7 agents operate concurrently on different aspects
- **Reviewer Coordination**: Multiple reviewers process manuscripts simultaneously
- **Quality Checks**: Parallel validation across multiple dimensions

#### 3. **Ontogenetic Looms** (Learning and Adaptation)
- **Agent Learning Framework**: Reinforcement learning for decision optimization
- **Pattern Recognition**: Historical data analysis for trend identification
- **Adaptive Workflows**: Self-modifying processes based on performance metrics

### 12-Step Cognitive Loop Architecture

Implementing the Echobeats-style cognitive loop with 3 concurrent inference engines:

#### **Phase 1: Expressive Mode (Steps 1-4)**
1. **Manuscript Reception** - Initial submission intake
2. **Quality Assessment** - Automated validation and scoring
3. **Expertise Matching** - Reviewer identification
4. **Task Distribution** - Work allocation to agents

#### **Phase 2: Reflective Mode (Steps 5-8)**
5. **Pivotal Relevance Realization** - Editorial decision point (orienting present commitment)
6. **Review Aggregation** - Synthesize reviewer feedback (conditioning past performance)
7. **Conflict Resolution** - Handle disagreements
8. **Quality Validation** - Final checks before decision

#### **Phase 3: Anticipatory Mode (Steps 9-12)**
9. **Pivotal Relevance Realization** - Publication decision point (orienting present commitment)
10. **Production Planning** - Format and distribution strategy (anticipating future potential)
11. **Impact Prediction** - Forecast citation and engagement (virtual salience simulation)
12. **Continuous Learning** - Update models and optimize (virtual salience simulation)

---

## Priority Improvements

### Phase 1: Critical Infrastructure (Immediate)

#### 1.1 Scheme-Based MetaModel Foundation

**Rationale**: Following user preference for Scheme implementation as foundational component.

**Implementation**:
```scheme
; Core MetaModel for Autonomous Journal System
(define-module (ojscog metamodel core)
  #:export (agent-state
            workflow-transition
            cognitive-loop
            relevance-realization))

; Agent state representation
(define-record-type <agent-state>
  (make-agent-state id phase context memory)
  agent-state?
  (id agent-id)
  (phase agent-phase set-agent-phase!)
  (context agent-context set-agent-context!)
  (memory agent-memory set-agent-memory!))

; Workflow transition function
(define (workflow-transition current-state event)
  "Deterministic state transition for OJS workflows"
  (match (cons current-state event)
    [('submission . 'validated) 'review-assignment]
    [('review-assignment . 'assigned) 'under-review]
    [('under-review . 'completed) 'editorial-decision]
    [('editorial-decision . 'accepted) 'production]
    [('production . 'formatted) 'publication]
    [_ current-state]))

; 12-step cognitive loop
(define (cognitive-loop agents manuscript)
  "Execute 12-step cognitive loop across 3 inference engines"
  (let* ([expressive (expressive-phase agents manuscript)]
         [reflective (reflective-phase agents expressive)]
         [anticipatory (anticipatory-phase agents reflective)])
    (integrate-results expressive reflective anticipatory)))

; Relevance realization (pivotal steps 5 and 9)
(define (relevance-realization context history future-potential)
  "Compute relevance for decision-making at pivotal points"
  (weighted-sum
    (* 0.4 (context-salience context))
    (* 0.3 (historical-performance history))
    (* 0.3 (anticipated-impact future-potential))))
```

**Files to Create**:
- `skz-integration/metamodel/scheme/core.scm`
- `skz-integration/metamodel/scheme/agents.scm`
- `skz-integration/metamodel/scheme/workflows.scm`
- `skz-integration/metamodel/scheme/cognitive-loop.scm`

#### 1.2 Enhanced Agent Communication Protocol

**Current Issue**: Agents communicate via HTTP REST API only.

**Improvement**: Implement message bus for asynchronous agent coordination.

**Implementation**:
```python
# skz-integration/autonomous-agents-framework/src/communication/message_bus.py

import asyncio
import json
from typing import Dict, Callable, Any
from dataclasses import dataclass
from datetime import datetime

@dataclass
class AgentMessage:
    """Message structure for inter-agent communication"""
    sender: str
    receiver: str
    message_type: str
    payload: Dict[str, Any]
    timestamp: datetime
    correlation_id: str
    priority: int = 5

class MessageBus:
    """Asynchronous message bus for agent communication"""
    
    def __init__(self):
        self.subscribers: Dict[str, list[Callable]] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.running = False
        
    async def publish(self, message: AgentMessage):
        """Publish message to bus"""
        await self.message_queue.put(message)
        
    async def subscribe(self, message_type: str, callback: Callable):
        """Subscribe to message type"""
        if message_type not in self.subscribers:
            self.subscribers[message_type] = []
        self.subscribers[message_type].append(callback)
        
    async def start(self):
        """Start message bus processing"""
        self.running = True
        while self.running:
            message = await self.message_queue.get()
            await self._route_message(message)
            
    async def _route_message(self, message: AgentMessage):
        """Route message to subscribers"""
        if message.message_type in self.subscribers:
            tasks = [callback(message) for callback in self.subscribers[message.message_type]]
            await asyncio.gather(*tasks)
```

#### 1.3 Unified State Management

**Integration**: Connect agent SQLite databases with OJS MySQL database.

**Implementation**:
```python
# skz-integration/autonomous-agents-framework/src/state/unified_state_manager.py

from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker
import redis
import json

class UnifiedStateManager:
    """Manage state across OJS database and agent memory stores"""
    
    def __init__(self, ojs_db_url: str, redis_url: str):
        self.ojs_engine = create_engine(ojs_db_url)
        self.ojs_session = sessionmaker(bind=self.ojs_engine)()
        self.redis_client = redis.from_url(redis_url)
        self.metadata = MetaData()
        
    def sync_agent_state(self, agent_id: str, state_data: dict):
        """Sync agent state to both OJS database and Redis cache"""
        # Update OJS database
        self.ojs_session.execute(
            """
            INSERT INTO skz_agent_states (agent_id, state_data, last_updated)
            VALUES (:agent_id, :state_data, NOW())
            ON DUPLICATE KEY UPDATE 
                state_data = :state_data,
                last_updated = NOW()
            """,
            {"agent_id": agent_id, "state_data": json.dumps(state_data)}
        )
        self.ojs_session.commit()
        
        # Update Redis cache
        self.redis_client.setex(
            f"agent_state:{agent_id}",
            300,  # 5 minute TTL
            json.dumps(state_data)
        )
        
    def get_agent_state(self, agent_id: str) -> dict:
        """Retrieve agent state with cache-first strategy"""
        # Try cache first
        cached = self.redis_client.get(f"agent_state:{agent_id}")
        if cached:
            return json.loads(cached)
            
        # Fallback to database
        result = self.ojs_session.execute(
            "SELECT state_data FROM skz_agent_states WHERE agent_id = :agent_id",
            {"agent_id": agent_id}
        ).fetchone()
        
        if result:
            state_data = json.loads(result[0])
            # Populate cache
            self.redis_client.setex(
                f"agent_state:{agent_id}",
                300,
                json.dumps(state_data)
            )
            return state_data
            
        return {}
```

#### 1.4 Docker Compose for Microservices

**File**: `skz-integration/docker-compose.yml`

```yaml
version: '3.8'

services:
  # OJS Core
  ojs:
    build: .
    ports:
      - "8000:80"
    volumes:
      - ./:/var/www/html
    environment:
      - OJS_DB_HOST=mysql
      - OJS_DB_USER=ojs
      - OJS_DB_PASSWORD=ojs
      - OJS_DB_NAME=ojs
    depends_on:
      - mysql
      - redis
      
  # MySQL Database
  mysql:
    image: mysql:8.0
    environment:
      - MYSQL_ROOT_PASSWORD=root
      - MYSQL_DATABASE=ojs
      - MYSQL_USER=ojs
      - MYSQL_PASSWORD=ojs
    volumes:
      - mysql_data:/var/lib/mysql
      - ./plugins/generic/skzAgents/schema.sql:/docker-entrypoint-initdb.d/skz_schema.sql
    ports:
      - "3306:3306"
      
  # Redis Cache
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
      
  # API Gateway
  api-gateway:
    build: ./skz-integration/autonomous-agents-framework
    command: python src/api_gateway.py
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=development
      - OJS_DB_URL=mysql+pymysql://ojs:ojs@mysql:3306/ojs
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - mysql
      - redis
      
  # Agent 1: Research Discovery
  agent-research-discovery:
    build: ./skz-integration/autonomous-agents-framework
    command: python src/agents/research_discovery_agent.py
    ports:
      - "5001:5001"
    environment:
      - AGENT_ID=research-discovery
      - API_GATEWAY_URL=http://api-gateway:5000
    depends_on:
      - api-gateway
      
  # Agent 2: Submission Assistant
  agent-submission-assistant:
    build: ./skz-integration/autonomous-agents-framework
    command: python src/agents/submission_assistant_agent.py
    ports:
      - "5002:5002"
    environment:
      - AGENT_ID=submission-assistant
      - API_GATEWAY_URL=http://api-gateway:5000
    depends_on:
      - api-gateway
      
  # Agent 3: Editorial Orchestration
  agent-editorial-orchestration:
    build: ./skz-integration/autonomous-agents-framework
    command: python src/agents/editorial_orchestration_agent.py
    ports:
      - "5003:5003"
    environment:
      - AGENT_ID=editorial-orchestration
      - API_GATEWAY_URL=http://api-gateway:5000
    depends_on:
      - api-gateway
      
  # Agent 4: Review Coordination
  agent-review-coordination:
    build: ./skz-integration/autonomous-agents-framework
    command: python src/agents/review_coordination_agent.py
    ports:
      - "5004:5004"
    environment:
      - AGENT_ID=review-coordination
      - API_GATEWAY_URL=http://api-gateway:5000
    depends_on:
      - api-gateway
      
  # Agent 5: Content Quality
  agent-content-quality:
    build: ./skz-integration/autonomous-agents-framework
    command: python src/agents/content_quality_agent.py
    ports:
      - "5005:5005"
    environment:
      - AGENT_ID=content-quality
      - API_GATEWAY_URL=http://api-gateway:5000
    depends_on:
      - api-gateway
      
  # Agent 6: Publishing Production
  agent-publishing-production:
    build: ./skz-integration/autonomous-agents-framework
    command: python src/agents/publishing_production_agent.py
    ports:
      - "5006:5006"
    environment:
      - AGENT_ID=publishing-production
      - API_GATEWAY_URL=http://api-gateway:5000
    depends_on:
      - api-gateway
      
  # Agent 7: Analytics & Monitoring
  agent-analytics-monitoring:
    build: ./skz-integration/autonomous-agents-framework
    command: python src/agents/analytics_monitoring_agent.py
    ports:
      - "5007:5007"
    environment:
      - AGENT_ID=analytics-monitoring
      - API_GATEWAY_URL=http://api-gateway:5000
    depends_on:
      - api-gateway

volumes:
  mysql_data:
  redis_data:
```

---

### Phase 2: Advanced Agent Intelligence (High Priority)

#### 2.1 LLM Integration for Content Analysis

**Integration with OpenAI API** (available via environment variables):

```python
# skz-integration/autonomous-agents-framework/src/intelligence/llm_analyzer.py

from openai import OpenAI
import os

class LLMContentAnalyzer:
    """Use LLM for advanced manuscript analysis"""
    
    def __init__(self):
        self.client = OpenAI()  # API key pre-configured
        
    async def analyze_manuscript_quality(self, manuscript_text: str) -> dict:
        """Analyze manuscript quality using LLM"""
        
        prompt = f"""
        Analyze the following academic manuscript for quality and provide scores:
        
        Manuscript:
        {manuscript_text[:4000]}  # Limit context
        
        Provide analysis in JSON format:
        {{
            "scientific_rigor": <score 0-10>,
            "methodology_clarity": <score 0-10>,
            "novelty": <score 0-10>,
            "writing_quality": <score 0-10>,
            "strengths": ["list of strengths"],
            "weaknesses": ["list of weaknesses"],
            "recommendations": ["list of recommendations"]
        }}
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are an expert academic manuscript reviewer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)
        
    async def suggest_reviewers(self, manuscript_abstract: str, expertise_db: list) -> list:
        """Suggest reviewers based on manuscript content"""
        
        prompt = f"""
        Given this manuscript abstract, identify the top 5 most relevant reviewers from the database.
        
        Abstract:
        {manuscript_abstract}
        
        Reviewer Database:
        {json.dumps(expertise_db[:50])}  # Limit to top 50
        
        Return JSON array of reviewer IDs with relevance scores.
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)
```

#### 2.2 Vector Database for Semantic Search

**Implementation using Supabase (available via environment variables)**:

```python
# skz-integration/autonomous-agents-framework/src/intelligence/semantic_search.py

from supabase import create_client
import os
import numpy as np
from openai import OpenAI

class SemanticSearchEngine:
    """Semantic search for manuscripts and reviewers using vector embeddings"""
    
    def __init__(self):
        self.supabase = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_KEY")
        )
        self.openai_client = OpenAI()
        
    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding for text"""
        response = self.openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
        
    async def index_manuscript(self, manuscript_id: int, title: str, abstract: str):
        """Index manuscript for semantic search"""
        embedding = await self.embed_text(f"{title}\n\n{abstract}")
        
        self.supabase.table("manuscript_embeddings").insert({
            "manuscript_id": manuscript_id,
            "embedding": embedding,
            "title": title,
            "abstract": abstract
        }).execute()
        
    async def find_similar_manuscripts(self, query_text: str, limit: int = 10) -> list:
        """Find manuscripts similar to query"""
        query_embedding = await self.embed_text(query_text)
        
        # Use Supabase vector similarity search
        result = self.supabase.rpc(
            "match_manuscripts",
            {
                "query_embedding": query_embedding,
                "match_threshold": 0.7,
                "match_count": limit
            }
        ).execute()
        
        return result.data
        
    async def match_reviewer_expertise(self, manuscript_abstract: str) -> list:
        """Match reviewers based on semantic similarity to manuscript"""
        manuscript_embedding = await self.embed_text(manuscript_abstract)
        
        result = self.supabase.rpc(
            "match_reviewer_expertise",
            {
                "query_embedding": manuscript_embedding,
                "match_threshold": 0.75,
                "match_count": 20
            }
        ).execute()
        
        return result.data
```

**Supabase Schema Setup**:
```sql
-- Create extension for vector operations
CREATE EXTENSION IF NOT EXISTS vector;

-- Manuscript embeddings table
CREATE TABLE manuscript_embeddings (
    id BIGSERIAL PRIMARY KEY,
    manuscript_id INTEGER NOT NULL,
    embedding vector(1536),
    title TEXT,
    abstract TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create index for fast similarity search
CREATE INDEX ON manuscript_embeddings USING ivfflat (embedding vector_cosine_ops);

-- Reviewer expertise embeddings
CREATE TABLE reviewer_expertise_embeddings (
    id BIGSERIAL PRIMARY KEY,
    reviewer_id INTEGER NOT NULL,
    expertise_area TEXT,
    embedding vector(1536),
    publications_count INTEGER,
    avg_review_quality FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX ON reviewer_expertise_embeddings USING ivfflat (embedding vector_cosine_ops);

-- Function to match manuscripts
CREATE OR REPLACE FUNCTION match_manuscripts(
    query_embedding vector(1536),
    match_threshold FLOAT,
    match_count INT
)
RETURNS TABLE (
    manuscript_id INTEGER,
    title TEXT,
    abstract TEXT,
    similarity FLOAT
)
LANGUAGE SQL STABLE
AS $$
    SELECT 
        manuscript_id,
        title,
        abstract,
        1 - (embedding <=> query_embedding) AS similarity
    FROM manuscript_embeddings
    WHERE 1 - (embedding <=> query_embedding) > match_threshold
    ORDER BY similarity DESC
    LIMIT match_count;
$$;

-- Function to match reviewer expertise
CREATE OR REPLACE FUNCTION match_reviewer_expertise(
    query_embedding vector(1536),
    match_threshold FLOAT,
    match_count INT
)
RETURNS TABLE (
    reviewer_id INTEGER,
    expertise_area TEXT,
    publications_count INTEGER,
    avg_review_quality FLOAT,
    similarity FLOAT
)
LANGUAGE SQL STABLE
AS $$
    SELECT 
        reviewer_id,
        expertise_area,
        publications_count,
        avg_review_quality,
        1 - (embedding <=> query_embedding) AS similarity
    FROM reviewer_expertise_embeddings
    WHERE 1 - (embedding <=> query_embedding) > match_threshold
    ORDER BY similarity DESC
    LIMIT match_count;
$$;
```

#### 2.3 Reinforcement Learning for Decision Optimization

```python
# skz-integration/autonomous-agents-framework/src/intelligence/rl_optimizer.py

import numpy as np
from collections import defaultdict
import json

class EditorialDecisionOptimizer:
    """Reinforcement learning for optimizing editorial decisions"""
    
    def __init__(self, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: defaultdict(float))
        
    def get_state_representation(self, manuscript_data: dict) -> str:
        """Convert manuscript data to state representation"""
        return json.dumps({
            "quality_score": round(manuscript_data.get("quality_score", 0), 1),
            "reviewer_consensus": manuscript_data.get("reviewer_consensus", "neutral"),
            "novelty_score": round(manuscript_data.get("novelty_score", 0), 1),
            "methodology_score": round(manuscript_data.get("methodology_score", 0), 1)
        }, sort_keys=True)
        
    def choose_action(self, state: str, available_actions: list) -> str:
        """Choose action using epsilon-greedy policy"""
        if np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.choice(available_actions)
        else:
            # Exploit: best known action
            q_values = {action: self.q_table[state][action] for action in available_actions}
            return max(q_values, key=q_values.get)
            
    def update_q_value(self, state: str, action: str, reward: float, next_state: str):
        """Update Q-value using Q-learning algorithm"""
        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state].values()) if self.q_table[next_state] else 0
        
        # Q-learning update rule
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q
        
    def get_reward(self, action: str, outcome: dict) -> float:
        """Calculate reward based on outcome"""
        rewards = {
            "accept": {
                "high_citations": 10.0,
                "medium_citations": 5.0,
                "low_citations": -2.0,
                "retracted": -20.0
            },
            "reject": {
                "correct_rejection": 3.0,
                "missed_opportunity": -5.0
            },
            "revise": {
                "improved_quality": 7.0,
                "abandoned": -1.0
            }
        }
        
        outcome_type = outcome.get("type", "unknown")
        return rewards.get(action, {}).get(outcome_type, 0.0)
```

---

### Phase 3: Complete OJS Workflow Integration (High Priority)

#### 3.1 Automated Submission Processing

**File**: `plugins/generic/skzAgents/classes/AutomatedSubmissionProcessor.inc.php`

```php
<?php

import('plugins.generic.skzAgents.classes.SKZAgentBridge');

class AutomatedSubmissionProcessor {
    
    private $agentBridge;
    private $plugin;
    
    public function __construct($plugin) {
        $this->plugin = $plugin;
        $this->agentBridge = new SKZAgentBridge($plugin);
    }
    
    /**
     * Process new submission automatically
     */
    public function processNewSubmission($submission) {
        $contextId = $submission->getContextId();
        
        // Check if automation is enabled
        if (!$this->plugin->getSetting($contextId, 'enableAutoSubmissionProcessing')) {
            return false;
        }
        
        try {
            // Step 1: Research Discovery Agent - Check for similar work
            $similarityCheck = $this->agentBridge->callAgent(
                'research-discovery',
                'check-similarity',
                [
                    'submission_id' => $submission->getId(),
                    'title' => $submission->getTitle(),
                    'abstract' => $submission->getAbstract()
                ]
            );
            
            // Step 2: Submission Assistant Agent - Quality assessment
            $qualityAssessment = $this->agentBridge->callAgent(
                'submission-assistant',
                'assess-quality',
                [
                    'submission_id' => $submission->getId(),
                    'manuscript_text' => $this->getManuscriptText($submission)
                ]
            );
            
            // Step 3: Content Quality Agent - Initial validation
            $validation = $this->agentBridge->callAgent(
                'content-quality',
                'validate-submission',
                [
                    'submission_id' => $submission->getId(),
                    'quality_score' => $qualityAssessment['quality_score']
                ]
            );
            
            // Step 4: Editorial Orchestration Agent - Routing decision
            $routingDecision = $this->agentBridge->callAgent(
                'editorial-orchestration',
                'route-submission',
                [
                    'submission_id' => $submission->getId(),
                    'similarity_check' => $similarityCheck,
                    'quality_assessment' => $qualityAssessment,
                    'validation' => $validation
                ]
            );
            
            // Apply routing decision
            $this->applyRoutingDecision($submission, $routingDecision);
            
            // Log automation
            $this->logAutomation($submission->getId(), 'submission_processing', [
                'similarity_check' => $similarityCheck,
                'quality_assessment' => $qualityAssessment,
                'validation' => $validation,
                'routing_decision' => $routingDecision
            ]);
            
            return true;
            
        } catch (Exception $e) {
            error_log("Automated submission processing failed: " . $e->getMessage());
            return false;
        }
    }
    
    /**
     * Apply routing decision to submission
     */
    private function applyRoutingDecision($submission, $routingDecision) {
        if ($routingDecision['action'] === 'desk_reject') {
            $this->deskReject($submission, $routingDecision['reason']);
        } elseif ($routingDecision['action'] === 'assign_editor') {
            $this->assignEditor($submission, $routingDecision['editor_id']);
        } elseif ($routingDecision['action'] === 'request_revisions') {
            $this->requestInitialRevisions($submission, $routingDecision['revision_notes']);
        }
    }
    
    /**
     * Get manuscript text from submission files
     */
    private function getManuscriptText($submission) {
        // Extract text from submission files
        $submissionFileDao = DAORegistry::getDAO('SubmissionFileDAO');
        $files = $submissionFileDao->getBySubmissionId($submission->getId());
        
        $text = '';
        foreach ($files as $file) {
            if ($file->getFileStage() == SUBMISSION_FILE_SUBMISSION) {
                $text .= $this->extractTextFromFile($file);
            }
        }
        
        return $text;
    }
}
```

#### 3.2 Intelligent Reviewer Assignment

**File**: `plugins/generic/skzAgents/classes/IntelligentReviewerMatcher.inc.php`

```php
<?php

class IntelligentReviewerMatcher {
    
    private $agentBridge;
    private $plugin;
    
    public function __construct($plugin) {
        $this->plugin = $plugin;
        $this->agentBridge = new SKZAgentBridge($plugin);
    }
    
    /**
     * Find and assign reviewers automatically
     */
    public function assignReviewers($submission, $numReviewers = 3) {
        try {
            // Call Review Coordination Agent
            $reviewerMatches = $this->agentBridge->callAgent(
                'review-coordination',
                'match-reviewers',
                [
                    'submission_id' => $submission->getId(),
                    'title' => $submission->getTitle(),
                    'abstract' => $submission->getAbstract(),
                    'keywords' => $submission->getKeywords(),
                    'num_reviewers' => $numReviewers
                ]
            );
            
            // Assign matched reviewers
            $assigned = [];
            foreach ($reviewerMatches['reviewers'] as $reviewer) {
                if ($this->assignReviewer($submission, $reviewer)) {
                    $assigned[] = $reviewer;
                }
            }
            
            // Log assignment
            $this->logReviewerAssignment($submission->getId(), $assigned);
            
            return $assigned;
            
        } catch (Exception $e) {
            error_log("Intelligent reviewer matching failed: " . $e->getMessage());
            return [];
        }
    }
    
    /**
     * Assign individual reviewer
     */
    private function assignReviewer($submission, $reviewerData) {
        $reviewAssignmentDao = DAORegistry::getDAO('ReviewAssignmentDAO');
        
        $reviewAssignment = $reviewAssignmentDao->newDataObject();
        $reviewAssignment->setSubmissionId($submission->getId());
        $reviewAssignment->setReviewerId($reviewerData['reviewer_id']);
        $reviewAssignment->setDateAssigned(Core::getCurrentDate());
        $reviewAssignment->setStageId(WORKFLOW_STAGE_ID_EXTERNAL_REVIEW);
        $reviewAssignment->setReviewRoundId($submission->getCurrentReviewRound());
        $reviewAssignment->setReviewMethod(SUBMISSION_REVIEW_METHOD_DOUBLEBLIND);
        
        $reviewAssignmentDao->insertObject($reviewAssignment);
        
        // Send invitation email
        $this->sendReviewerInvitation($submission, $reviewerData);
        
        return true;
    }
}
```

---

### Phase 4: Integration with OpenCog AGI Framework (Medium Priority)

Following the user's interest in integrating `occ` (OpenCog AGI framework ecosystem), `hurdcog` (modified GNU Hurd OS), and `cognumach` (GNU Mach microkernel):

#### 4.1 OpenCog AtomSpace Integration

**Purpose**: Use OpenCog's AtomSpace for knowledge representation and reasoning.

```python
# skz-integration/opencog-integration/atomspace_bridge.py

from opencog.atomspace import AtomSpace, TruthValue
from opencog.type_constructors import *
from opencog.utilities import initialize_opencog

class OpenCogKnowledgeBase:
    """Integrate OpenCog AtomSpace for knowledge representation"""
    
    def __init__(self):
        self.atomspace = AtomSpace()
        initialize_opencog(self.atomspace)
        
    def add_manuscript_knowledge(self, manuscript_id: int, metadata: dict):
        """Add manuscript to knowledge base"""
        
        # Create manuscript node
        manuscript_node = ConceptNode(f"Manuscript_{manuscript_id}")
        
        # Add metadata as predicates
        for key, value in metadata.items():
            predicate = PredicateNode(key)
            value_node = ConceptNode(str(value))
            
            # Create evaluation link
            EvaluationLink(
                predicate,
                ListLink(manuscript_node, value_node),
                tv=TruthValue(1.0, 1.0)
            )
            
    def query_similar_manuscripts(self, query_metadata: dict) -> list:
        """Query for similar manuscripts using pattern matching"""
        
        # Build pattern matching query
        # Use OpenCog's pattern matcher for complex queries
        
        results = []
        # Pattern matching logic here
        
        return results
        
    def infer_editorial_decision(self, manuscript_id: int) -> dict:
        """Use OpenCog reasoning to infer editorial decision"""
        
        # Use PLN (Probabilistic Logic Networks) for inference
        # Combine multiple sources of evidence
        
        decision = {
            "action": "accept",  # or "reject", "revise"
            "confidence": 0.85,
            "reasoning": []
        }
        
        return decision
```

#### 4.2 Cognitive Architecture Roadmap

**Document**: `skz-integration/opencog-integration/COGNITIVE_ARCHITECTURE_ROADMAP.md`

```markdown
# Cognitive Architecture Integration Roadmap

## Phase 1: AtomSpace Knowledge Base
- Represent manuscripts, reviewers, and decisions as Atoms
- Use OpenCog's pattern matcher for similarity search
- Implement PLN for probabilistic reasoning

## Phase 2: ECAN Attention Allocation
- Use Economic Attention Networks for prioritizing manuscripts
- Allocate agent resources based on importance
- Implement attention spreading for related concepts

## Phase 3: OpenPsi Motivational System
- Define goals for autonomous agents (e.g., maximize quality, minimize time)
- Implement goal-driven behavior selection
- Balance multiple objectives (quality vs. speed)

## Phase 4: Hurdcog OS Integration
- Run agents as Hurd translators
- Implement microkernel-based isolation
- Use Mach IPC for inter-agent communication

## Phase 5: Full AGI Integration
- Combine symbolic reasoning (OpenCog) with neural networks (LLMs)
- Implement meta-learning across all agents
- Achieve true autonomous decision-making
```

---

## Implementation Priority Matrix

| Priority | Component | Complexity | Impact | Timeline |
|----------|-----------|------------|--------|----------|
| CRITICAL | Scheme MetaModel Foundation | Medium | High | Week 1 |
| CRITICAL | Docker Compose Setup | Low | High | Week 1 |
| CRITICAL | Unified State Management | Medium | High | Week 1-2 |
| HIGH | Message Bus Communication | Medium | High | Week 2 |
| HIGH | LLM Content Analysis | Low | High | Week 2 |
| HIGH | Automated Submission Processing | Medium | High | Week 2-3 |
| HIGH | Intelligent Reviewer Matching | Medium | High | Week 3 |
| MEDIUM | Vector Database Semantic Search | Medium | Medium | Week 3-4 |
| MEDIUM | Reinforcement Learning Optimizer | High | Medium | Week 4-5 |
| MEDIUM | OpenCog AtomSpace Integration | High | Medium | Week 5-6 |
| LOW | Hurdcog OS Integration | Very High | Low | Future |

---

## Success Metrics

### Technical Metrics
- **Agent Response Time**: < 500ms average
- **System Uptime**: > 99.5%
- **Automated Decision Accuracy**: > 90%
- **Test Coverage**: > 85%

### Operational Metrics
- **Submission Processing Time**: 60% reduction
- **Reviewer Assignment Time**: 70% reduction
- **Editorial Decision Time**: 50% reduction
- **Publication Cycle Time**: 40% reduction

### Cognitive Architecture Metrics
- **Relevance Realization Accuracy**: > 85%
- **Cognitive Loop Completion Time**: < 2 hours per manuscript
- **Learning Rate**: Continuous improvement of 5% per quarter
- **Agent Collaboration Efficiency**: > 90% successful interactions

---

## Conclusion

This improvement plan provides a comprehensive roadmap for evolving ojscog into a fully autonomous research journal system. By implementing the Scheme-based MetaModel foundation, integrating advanced AI capabilities, and following the 12-step cognitive loop architecture, the system will achieve true autonomy in academic publishing while maintaining high quality standards.

The integration with OpenCog AGI framework positions the system for future evolution toward general artificial intelligence capabilities, making it a pioneering platform in autonomous academic publishing.
