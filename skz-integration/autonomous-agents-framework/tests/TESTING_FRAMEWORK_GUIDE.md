# 🧪 SKZ Agents Comprehensive Testing Framework

**Created:** November 13, 2025  
**Purpose:** Rigorous testing strategy for all 7 autonomous agents  
**Target Coverage:** 90%+ with production-grade validation  
**Status:** Active Development

---

## 📋 Table of Contents

1. [Testing Philosophy](#testing-philosophy)
2. [Testing Architecture](#testing-architecture)
3. [Test Coverage Matrix](#test-coverage-matrix)
4. [Test Types & Strategies](#test-types--strategies)
5. [Running Tests](#running-tests)
6. [Writing New Tests](#writing-new-tests)
7. [CI/CD Integration](#cicd-integration)

---

## 🎯 Testing Philosophy

### Core Principles

1. **ZERO Mock ML Components**: All ML inference must use real models
2. **Production-Grade Testing**: Tests must validate actual behavior, not placeholders
3. **Comprehensive Coverage**: Target 90%+ code coverage across all modules
4. **Fast Feedback**: Unit tests <100ms, integration tests <5s, E2E <30s
5. **Isolation**: Each test must be independent and repeatable

### Testing Pyramid

```
        /\
       /E2E\        10% - End-to-End (16 tests)
      /------\      - Complete workflows
     /Integr.\     30% - Integration (48 tests)
    /----------\    - Multi-component interactions
   /Unit Tests \   60% - Unit (96 tests)
  /--------------\  - Individual functions/classes
  
Total Target: 160 comprehensive tests
Current: 48 tests (30% of target)
Gap: 112 tests needed
```

---

## 🏗️ Testing Architecture

### Directory Structure

```
tests/
├── conftest.py                    # Shared fixtures and configuration
├── TESTING_FRAMEWORK_GUIDE.md     # This document
│
├── unit/                          # Unit tests (96 tests target)
│   ├── agents/                    # Agent-specific unit tests
│   │   ├── test_research_discovery_unit.py      # 12 tests
│   │   ├── test_submission_assistant_unit.py    # 12 tests
│   │   ├── test_editorial_orchestration_unit.py # 12 tests
│   │   ├── test_review_coordination_unit.py     # 12 tests
│   │   ├── test_content_quality_unit.py         # 12 tests
│   │   ├── test_publishing_production_unit.py   # 12 tests
│   │   └── test_analytics_monitoring_unit.py    # 12 tests
│   │
│   ├── ml/                        # ML infrastructure unit tests
│   │   ├── test_vector_memory_unit.py          # 8 tests
│   │   ├── test_nlp_pipeline_unit.py           # 8 tests
│   │   ├── test_rl_framework_unit.py           # 8 tests
│   │   └── test_decision_engine_unit.py        # 8 tests
│   │
│   └── services/                  # Service unit tests
│       ├── test_communication_unit.py           # 4 tests
│       ├── test_data_sync_unit.py              # 4 tests
│       └── test_api_gateway_unit.py            # 4 tests
│
├── integration/                   # Integration tests (48 tests target)
│   ├── test_agent_communication.py             # 8 tests
│   ├── test_workflow_automation.py             # 8 tests
│   ├── test_ojs_bridge_integration.py          # 8 tests
│   ├── test_database_integration.py            # 8 tests
│   ├── test_ml_pipeline_integration.py         # 8 tests
│   └── test_external_services.py               # 8 tests
│
├── e2e/                           # End-to-end tests (16 tests target)
│   ├── test_complete_manuscript_workflow.py    # 4 tests
│   ├── test_multi_agent_coordination.py        # 4 tests
│   ├── test_error_recovery.py                  # 4 tests
│   └── test_performance_load.py                # 4 tests
│
└── performance/                   # Performance benchmarks
    ├── test_response_time.py
    ├── test_throughput.py
    └── test_scalability.py
```

### Fixture Hierarchy

```python
# conftest.py - Shared fixtures available to all tests

# Level 1: Infrastructure Fixtures
- event_loop: Async event loop
- temp_dir: Temporary directory for test data
- mock_database: Database connection (for external DB only)

# Level 2: ML Fixtures (REAL IMPLEMENTATIONS)
- nlp_pipeline: Real NLP pipeline with transformers
- vector_memory: Real ChromaDB instance
- rl_framework: Real Q-learning framework
- decision_engine: Real ML decision engine

# Level 3: Agent Fixtures (REAL IMPLEMENTATIONS)
- research_agent: Real enhanced research discovery agent
- submission_agent: Real submission assistant
- editorial_agent: Real editorial orchestration
- review_agent: Real review coordination
- quality_agent: Real content quality agent
- publishing_agent: Real publishing production
- analytics_agent: Real analytics monitoring

# Level 4: Data Fixtures
- sample_manuscript: Complete manuscript data
- sample_authors: Author information
- sample_reviewers: Reviewer profiles
- sample_journal_metadata: Journal configuration
```

---

## 📊 Test Coverage Matrix

### Current Coverage (November 13, 2025)

| Component | Unit Tests | Integration Tests | E2E Tests | Total | Target | Gap |
|-----------|-----------|------------------|-----------|-------|--------|-----|
| **Research Discovery Agent** | 4/12 | 2/6 | 0/2 | 6/20 | 20 | -14 ❌ |
| **Submission Assistant** | 8/12 | 1/6 | 0/2 | 9/20 | 20 | -11 ❌ |
| **Editorial Orchestration** | 0/12 | 0/6 | 0/2 | 0/20 | 20 | -20 ❌ |
| **Review Coordination** | 0/12 | 0/6 | 0/2 | 0/20 | 20 | -20 ❌ |
| **Content Quality** | 0/12 | 0/6 | 0/2 | 0/20 | 20 | -20 ❌ |
| **Publishing Production** | 0/12 | 0/6 | 0/2 | 0/20 | 20 | -20 ❌ |
| **Analytics Monitoring** | 2/12 | 1/6 | 0/2 | 3/20 | 20 | -17 ❌ |
| **Vector Memory System** | 3/8 | 2/4 | - | 5/12 | 12 | -7 ❌ |
| **NLP Pipeline** | 5/8 | 2/4 | - | 7/12 | 12 | -5 ⚠️ |
| **RL Framework** | 2/8 | 1/4 | - | 3/12 | 12 | -9 ❌ |
| **Decision Engine** | 1/8 | 1/4 | - | 2/12 | 12 | -10 ❌ |
| **Communication Services** | 3/4 | 2/4 | 0/2 | 5/10 | 10 | -5 ⚠️ |
| **Database Integration** | 2/4 | 3/8 | 0/2 | 5/14 | 14 | -9 ❌ |
| **API Gateway** | 4/4 | 4/8 | 0/2 | 8/14 | 14 | -6 ⚠️ |
| **TOTAL** | **34/96** | **19/48** | **0/16** | **53/160** | **160** | **-107** ❌ |

**Overall Coverage: 33%** (Target: 90%)

---

## 🧬 Test Types & Strategies

### 1. Unit Tests (60% of tests)

**Scope:** Individual functions, classes, methods in isolation

**Characteristics:**
- Run in <100ms per test
- No external dependencies (except real ML models)
- Test single responsibility
- High code coverage

**Example Structure:**
```python
# tests/unit/agents/test_research_discovery_unit.py

class TestResearchDiscoveryAgent:
    """Unit tests for Research Discovery Agent"""
    
    def test_initialization(self, research_agent):
        """Test agent initializes correctly"""
        assert research_agent.agent_id == "research_discovery"
        assert research_agent.memory_system is not None
        assert research_agent.nlp_pipeline is not None
    
    def test_analyze_manuscript_structure(self, research_agent, sample_manuscript):
        """Test manuscript analysis returns correct structure"""
        result = research_agent.analyze_manuscript(
            manuscript_id="TEST-001",
            title=sample_manuscript['title'],
            abstract=sample_manuscript['abstract'],
            full_text=sample_manuscript['full_text'],
            metadata=sample_manuscript['metadata']
        )
        
        # Validate structure
        assert 'research_gaps' in result
        assert 'novelty_score' in result
        assert 'trend_alignment' in result
        assert 'impact_prediction' in result
        
        # Validate types
        assert isinstance(result['research_gaps'], list)
        assert isinstance(result['novelty_score'], float)
        assert 0 <= result['novelty_score'] <= 1
    
    def test_identify_research_gaps_real_nlp(self, research_agent):
        """Test research gap identification uses real NLP"""
        abstract = "Novel approach to skin barrier function using ceramides."
        full_text = "Previous work on lipid barriers. Our method improves ceramide delivery."
        
        result = research_agent._identify_research_gaps(abstract, full_text, {})
        
        # MUST use real NLP - no mocks
        assert len(result) > 0
        assert all('gap_description' in gap for gap in result)
        assert all('significance' in gap for gap in result)
    
    # ... 9 more unit tests per agent
```

### 2. Integration Tests (30% of tests)

**Scope:** Multiple components working together

**Characteristics:**
- Run in <5s per test
- Test component interactions
- May use real databases (test instances)
- Validate data flow

**Example Structure:**
```python
# tests/integration/test_agent_communication.py

class TestAgentCommunication:
    """Integration tests for inter-agent communication"""
    
    @pytest.mark.asyncio
    async def test_research_to_submission_handoff(
        self, research_agent, submission_agent, sample_manuscript
    ):
        """Test manuscript handoff from Research to Submission agent"""
        
        # Step 1: Research agent analyzes
        research_result = research_agent.analyze_manuscript(
            manuscript_id="TEST-001",
            title=sample_manuscript['title'],
            abstract=sample_manuscript['abstract'],
            full_text=sample_manuscript['full_text'],
            metadata=sample_manuscript['metadata']
        )
        
        # Step 2: Pass to Submission agent
        submission_result = await submission_agent.assess_quality(
            manuscript_id="TEST-001",
            manuscript_text=sample_manuscript['full_text'],
            research_analysis=research_result  # Integration point
        )
        
        # Validate handoff
        assert submission_result['manuscript_id'] == "TEST-001"
        assert 'quality_score' in submission_result
        assert 'research_alignment' in submission_result
        
        # Validate research context used
        assert submission_result['research_alignment']['novelty_considered']
    
    @pytest.mark.asyncio
    async def test_multi_agent_workflow(
        self, research_agent, submission_agent, editorial_agent
    ):
        """Test complete workflow through 3 agents"""
        # ... test 3-agent coordination
    
    # ... 6 more integration tests
```

### 3. End-to-End Tests (10% of tests)

**Scope:** Complete user workflows from start to finish

**Characteristics:**
- Run in <30s per test
- Test entire system
- Use real services (test environment)
- Validate business outcomes

**Example Structure:**
```python
# tests/e2e/test_complete_manuscript_workflow.py

class TestManuscriptWorkflow:
    """End-to-end tests for complete manuscript lifecycle"""
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_manuscript_submission_to_publication(
        self, full_system_fixture, sample_manuscript
    ):
        """Test complete workflow: submission → review → publication"""
        
        # Setup
        manuscript_id = "E2E-TEST-001"
        
        # Step 1: Submission
        submission_response = await full_system_fixture.submit_manuscript(
            manuscript_data=sample_manuscript
        )
        assert submission_response['status'] == 'submitted'
        
        # Step 2: Research Analysis (Agent 1)
        await asyncio.sleep(2)  # Allow async processing
        research_status = await full_system_fixture.get_agent_status(
            manuscript_id, agent="research_discovery"
        )
        assert research_status['completed']
        assert research_status['novelty_score'] > 0
        
        # Step 3: Quality Assessment (Agent 2)
        quality_status = await full_system_fixture.get_agent_status(
            manuscript_id, agent="submission_assistant"
        )
        assert quality_status['quality_score'] > 0.6  # Threshold
        
        # Step 4: Editorial Decision (Agent 3)
        editorial_decision = await full_system_fixture.get_agent_status(
            manuscript_id, agent="editorial_orchestration"
        )
        assert editorial_decision['decision'] in ['accept', 'reject', 'revise']
        
        # Step 5: If accepted, reviewer assignment (Agent 4)
        if editorial_decision['decision'] == 'accept':
            review_status = await full_system_fixture.get_agent_status(
                manuscript_id, agent="review_coordination"
            )
            assert len(review_status['assigned_reviewers']) >= 2
        
        # Validate complete workflow
        final_status = await full_system_fixture.get_manuscript_status(manuscript_id)
        assert final_status['workflow_stage'] in [
            'under_review', 'accepted', 'revision_required'
        ]
        
        # Validate all agents participated
        agent_logs = await full_system_fixture.get_agent_activity_log(manuscript_id)
        assert len(agent_logs) >= 4  # At least 4 agents involved
    
    # ... 3 more E2E tests
```

---

## 🎯 Test-Driven Development (TDD) Workflow

### Red-Green-Refactor Cycle

```
1. RED: Write failing test
   └─> Define expected behavior
   └─> Test fails (no implementation)

2. GREEN: Make test pass
   └─> Implement minimum code
   └─> Use real ML, no mocks
   └─> Test passes

3. REFACTOR: Improve code
   └─> Optimize performance
   └─> Improve readability
   └─> Test still passes
```

### Example TDD Session

```python
# STEP 1: RED - Write failing test
def test_predict_review_turnaround_time(review_agent, sample_manuscript):
    """Test ML prediction of review turnaround time"""
    prediction = review_agent.predict_turnaround_time(
        manuscript_complexity=0.7,
        reviewer_workload=3,
        journal_priority='high'
    )
    
    assert isinstance(prediction, dict)
    assert 'estimated_days' in prediction
    assert 'confidence' in prediction
    assert 0 <= prediction['confidence'] <= 1
# TEST FAILS - method doesn't exist

# STEP 2: GREEN - Implement real ML prediction
class ReviewCoordinationAgent:
    def predict_turnaround_time(self, manuscript_complexity, reviewer_workload, journal_priority):
        # REAL ML using historical data and regression
        features = np.array([[
            manuscript_complexity,
            reviewer_workload,
            1 if journal_priority == 'high' else 0
        ]])
        
        # Use trained model (NOT a mock)
        estimated_days = self.ml_model.predict(features)[0]
        confidence = self._calculate_confidence(features)
        
        return {
            'estimated_days': float(estimated_days),
            'confidence': float(confidence)
        }
# TEST PASSES

# STEP 3: REFACTOR - Improve implementation
class ReviewCoordinationAgent:
    def predict_turnaround_time(self, manuscript_complexity, reviewer_workload, journal_priority):
        """Predict review turnaround time using gradient boosting regression"""
        # Improved feature engineering
        features = self._extract_turnaround_features(
            manuscript_complexity, reviewer_workload, journal_priority
        )
        
        # Use ensemble model for better accuracy
        prediction = self.turnaround_predictor.predict(features)
        confidence = self._calculate_prediction_confidence(prediction, features)
        
        return {
            'estimated_days': float(prediction['days']),
            'confidence': float(confidence),
            'factors': prediction['contributing_factors']
        }
# TEST STILL PASSES (plus more detailed output)
```

---

## 🚀 Running Tests

### Quick Start

```bash
# Navigate to agent framework
cd skz-integration/autonomous-agents-framework

# Activate virtual environment
source venv/bin/activate

# Install test dependencies
pip install pytest pytest-asyncio pytest-cov pytest-mock

# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src --cov-report=html tests/

# Run specific test categories
pytest tests/unit/                    # Unit tests only
pytest tests/integration/             # Integration tests only
pytest tests/e2e/ -m e2e              # E2E tests only

# Run specific agent tests
pytest tests/unit/agents/test_research_discovery_unit.py
pytest tests/integration/test_agent_communication.py

# Run with verbose output
pytest -v tests/

# Run with detailed output and print statements
pytest -vv -s tests/
```

### Test Markers

```python
# In test files, use markers to categorize tests

@pytest.mark.unit
def test_something():
    """Unit test"""
    pass

@pytest.mark.integration
async def test_integration():
    """Integration test"""
    pass

@pytest.mark.e2e
@pytest.mark.slow
async def test_full_workflow():
    """End-to-end test (slow)"""
    pass

@pytest.mark.skip(reason="Requires external API key")
def test_external_service():
    """Skip if external dependency unavailable"""
    pass
```

**Run by marker:**
```bash
pytest -m unit              # Run only unit tests
pytest -m integration       # Run only integration tests
pytest -m "e2e and not slow"  # Run E2E tests except slow ones
pytest -m "not skip"        # Run all tests except skipped
```

### Continuous Testing (Watch Mode)

```bash
# Install pytest-watch
pip install pytest-watch

# Run tests automatically on file changes
ptw tests/
```

---

## ✍️ Writing New Tests

### Test Naming Conventions

```python
# Pattern: test_<what>_<condition>_<expected>

# Good examples:
def test_analyze_manuscript_valid_input_returns_structure()
def test_predict_impact_missing_abstract_raises_error()
def test_match_reviewers_insufficient_pool_returns_partial()

# Bad examples:
def test_1()  # Not descriptive
def test_function()  # Too vague
def test_research_agent()  # What about it?
```

### AAA Pattern: Arrange-Act-Assert

```python
def test_quality_assessment_real_ml():
    """Test quality assessment uses real ML scoring"""
    
    # ARRANGE: Set up test data
    manuscript_text = "Sample manuscript with methodology and results..."
    expected_score_range = (0.5, 1.0)
    
    # ACT: Execute the function
    quality_agent = create_submission_assistant_agent()
    result = quality_agent.assess_quality(
        manuscript_id="TEST-001",
        manuscript_text=manuscript_text
    )
    
    # ASSERT: Verify expectations
    assert result is not None
    assert 'quality_score' in result
    assert expected_score_range[0] <= result['quality_score'] <= expected_score_range[1]
    assert result['ml_model_used']  # Verify real ML was used
```

### Parametrized Tests

```python
@pytest.mark.parametrize("manuscript_quality,expected_decision", [
    ("high", "accept"),
    ("medium", "revise"),
    ("low", "reject"),
])
def test_editorial_decision_by_quality(manuscript_quality, expected_decision, editorial_agent):
    """Test editorial decisions based on quality tiers"""
    
    quality_scores = {
        "high": 0.85,
        "medium": 0.65,
        "low": 0.35
    }
    
    decision = editorial_agent.make_decision(
        manuscript_id="TEST-001",
        quality_score=quality_scores[manuscript_quality]
    )
    
    assert decision['recommendation'] == expected_decision
```

### Async Test Pattern

```python
@pytest.mark.asyncio
async def test_concurrent_agent_processing(research_agent, submission_agent):
    """Test multiple agents processing concurrently"""
    
    # Create tasks
    task1 = research_agent.analyze_manuscript_async(manuscript_id="M1")
    task2 = submission_agent.assess_quality_async(manuscript_id="M2")
    
    # Execute concurrently
    results = await asyncio.gather(task1, task2)
    
    # Verify both completed
    assert len(results) == 2
    assert results[0]['agent'] == 'research_discovery'
    assert results[1]['agent'] == 'submission_assistant'
```

---

## 📈 Coverage Goals & Metrics

### Coverage Targets by Component

| Component Type | Target Coverage | Current | Priority |
|---------------|----------------|---------|----------|
| **Core Agents** | 95% | 35% | 🔴 Critical |
| **ML Infrastructure** | 90% | 45% | 🔴 Critical |
| **API Routes** | 85% | 60% | 🟡 High |
| **Database Layer** | 80% | 55% | 🟡 High |
| **Utility Functions** | 90% | 70% | 🟢 Medium |
| **Configuration** | 70% | 50% | 🟢 Medium |

### Quality Metrics

```bash
# Generate coverage report
pytest --cov=src --cov-report=term-missing --cov-report=html tests/

# View detailed HTML report
open htmlcov/index.html

# Check coverage threshold (fail if below 90%)
pytest --cov=src --cov-fail-under=90 tests/
```

**Coverage Report Example:**
```
Name                                    Stmts   Miss  Cover   Missing
---------------------------------------------------------------------
src/models/enhanced_research_discovery_agent.py   150     25    83%   45-52, 78-85
src/services/vector_memory_system.py             120     10    92%   234-238
src/services/nlp_pipeline.py                     180     15    92%   156-163
src/services/reinforcement_learning.py           200     30    85%   89-95, 145-158
src/services/autonomous_decision_engine.py       170     20    88%   123-131
---------------------------------------------------------------------
TOTAL                                            820     100   88%
```

---

## 🔧 CI/CD Integration

### GitHub Actions Workflow

```yaml
# .github/workflows/test.yml

name: SKZ Agents Test Suite

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    strategy:
      matrix:
        python-version: [3.11, 3.12]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        cd skz-integration/autonomous-agents-framework
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-asyncio pytest-cov
    
    - name: Run unit tests
      run: |
        cd skz-integration/autonomous-agents-framework
        pytest tests/unit/ --cov=src --cov-report=xml
    
    - name: Run integration tests
      run: |
        cd skz-integration/autonomous-agents-framework
        pytest tests/integration/ --cov=src --cov-append --cov-report=xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        fail_ci_if_error: true
    
    - name: Check coverage threshold
      run: |
        cd skz-integration/autonomous-agents-framework
        pytest --cov=src --cov-fail-under=80 tests/
```

### Pre-commit Hooks

```bash
# .git/hooks/pre-commit

#!/bin/bash
# Run tests before allowing commit

cd skz-integration/autonomous-agents-framework
source venv/bin/activate

echo "Running unit tests..."
pytest tests/unit/ -q

if [ $? -ne 0 ]; then
    echo "❌ Unit tests failed. Commit aborted."
    exit 1
fi

echo "✅ Tests passed. Proceeding with commit."
exit 0
```

---

## 📝 Test Documentation Requirements

### Every Test Must Have:

1. **Docstring** explaining what is tested
2. **Clear assertion messages** for failures
3. **Arrange-Act-Assert** structure
4. **Real ML components** (no mocks for core logic)

**Example:**
```python
def test_reviewer_matching_uses_real_ml(review_agent, sample_manuscript, sample_reviewers):
    """
    Test that reviewer matching uses real ML similarity scoring
    
    Validates:
    - ML embeddings generated for manuscript and reviewer expertise
    - Cosine similarity calculated using real vectors
    - Top matches ranked by similarity score
    - NO mock implementations used
    """
    # Arrange
    manuscript_embedding = review_agent._generate_manuscript_embedding(
        sample_manuscript['title'],
        sample_manuscript['abstract']
    )
    assert manuscript_embedding is not None, "Failed to generate manuscript embedding"
    assert len(manuscript_embedding) > 0, "Embedding is empty"
    
    # Act
    matches = review_agent.match_reviewers(
        manuscript_id="TEST-001",
        manuscript_data=sample_manuscript,
        reviewer_pool=sample_reviewers
    )
    
    # Assert
    assert len(matches) > 0, "No reviewer matches found"
    assert all('similarity_score' in match for match in matches), \
        "Missing similarity scores in matches"
    assert all(0 <= match['similarity_score'] <= 1 for match in matches), \
        "Similarity scores out of valid range [0,1]"
    assert matches[0]['ml_method'] == 'sentence_transformers', \
        "Not using real ML model for matching"
```

---

## 🎯 Next Steps

### Immediate (Week 1)
- [ ] Deploy all services to enable test execution
- [ ] Run existing 48 tests and document results
- [ ] Create 12 unit tests for Editorial Orchestration Agent
- [ ] Create 12 unit tests for Review Coordination Agent

### Short-term (Weeks 2-4)
- [ ] Complete all 96 unit tests
- [ ] Implement 48 integration tests
- [ ] Achieve 60%+ code coverage
- [ ] Set up CI/CD with automated testing

### Long-term (Weeks 5-12)
- [ ] Implement 16 end-to-end tests
- [ ] Achieve 90%+ code coverage
- [ ] Performance benchmarking suite
- [ ] Load testing infrastructure

---

## 📊 Progress Tracking

**Last Updated:** November 13, 2025

| Week | Goal | Status | Tests Added | Coverage |
|------|------|--------|-------------|----------|
| Week 1 | Deploy & Run Existing | ⏳ In Progress | 0 | 33% |
| Week 2 | Complete Unit Tests | ⏳ Planned | +48 | 60% |
| Week 3 | Integration Tests | ⏳ Planned | +48 | 75% |
| Week 4 | E2E Tests | ⏳ Planned | +16 | 90% |

---

**Document maintained by:** SKZ Agents Development Team  
**Review frequency:** Weekly  
**Next review:** November 20, 2025
