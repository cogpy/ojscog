"""
Comprehensive Unit Tests for Agent 5: Content Quality Agent
Tests scientific validation, safety assessment, and standards enforcement
"""

import pytest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from models.content_quality import (
    ContentQualityAgent,
    ScientificValidator,
    SafetyAssessor,
    StandardsEnforcer,
    QualityScore,
    ValidationResult,
    SafetyAssessment
)


class TestContentQualityAgentInitialization:
    """Test suite for agent initialization"""
    
    def test_agent_initialization(self):
        """Test agent initializes correctly"""
        agent = ContentQualityAgent(agent_id="quality_test")
        
        assert agent.agent_id == "quality_test"
        assert agent.scientific_validator is not None
        assert agent.safety_assessor is not None
        assert agent.standards_enforcer is not None
    
    def test_ml_models_loaded(self):
        """Test ML models are loaded, not mocked"""
        agent = ContentQualityAgent()
        
        assert hasattr(agent, 'quality_scoring_model')
        assert agent.quality_scoring_model is not None
        assert hasattr(agent, 'nlp_pipeline')


class TestScientificValidation:
    """Test suite for scientific validation"""
    
    def test_validate_methodology_structure(self):
        """Test methodology validation returns correct structure"""
        agent = ContentQualityAgent()
        
        manuscript_text = """
        Methods: This was a randomized, double-blind, placebo-controlled trial.
        Sample size: n=120. Statistical analysis: t-tests with p<0.05.
        """
        
        validation = agent.validate_methodology(manuscript_text)
        
        assert 'methodology_score' in validation
        assert 'strengths' in validation
        assert 'weaknesses' in validation
        assert 0 <= validation['methodology_score'] <= 1
    
    def test_statistical_validation_uses_real_ml(self):
        """Test statistical validation uses real NLP, not mocks"""
        agent = ContentQualityAgent()
        
        text_with_stats = """
        Results showed significant improvement (p<0.001, 95% CI: 0.5-0.9).
        Effect size: d=0.8. Power analysis: 80% power at alpha=0.05.
        """
        
        text_without_stats = "Results showed improvement. Data looked good."
        
        score_with = agent.validate_statistical_rigor(text_with_stats)
        score_without = agent.validate_statistical_rigor(text_without_stats)
        
        # Should detect statistical reporting
        assert score_with['statistical_rigor_score'] > score_without['statistical_rigor_score']
        assert score_with['has_p_values']
        assert score_with['has_confidence_intervals']
        assert score_with['nlp_detection_used']
    
    def test_detect_research_design_quality(self):
        """Test research design quality detection"""
        agent = ContentQualityAgent()
        
        rct_text = "Randomized controlled trial with double-blind methodology"
        observational_text = "Observational study with convenience sampling"
        
        rct_score = agent.assess_research_design(rct_text)
        obs_score = agent.assess_research_design(observational_text)
        
        # RCT should score higher than observational
        assert rct_score['design_quality'] > obs_score['design_quality']
        assert rct_score['design_type'] == 'rct'


class TestSafetyAssessment:
    """Test suite for safety assessment"""
    
    def test_safety_assessment_structure(self):
        """Test safety assessment returns correct structure"""
        agent = ContentQualityAgent()
        
        formulation_data = {
            'ingredients': ['water', 'glycerin', 'hyaluronic acid'],
            'concentrations': [70, 15, 2]
        }
        
        safety = agent.assess_formulation_safety(formulation_data)
        
        assert 'safety_score' in safety
        assert 'identified_concerns' in safety
        assert 'regulatory_compliance' in safety
    
    def test_ingredient_safety_analysis_real_ml(self):
        """Test ingredient safety uses real ML classification"""
        agent = ContentQualityAgent()
        
        safe_ingredient = {'name': 'water', 'concentration': 70}
        risky_ingredient = {'name': 'unknown_chemical_x', 'concentration': 15}
        
        safe_result = agent.analyze_ingredient_safety(safe_ingredient)
        risky_result = agent.analyze_ingredient_safety(risky_ingredient)
        
        # Should differentiate safety levels
        assert safe_result['safety_classification'] in ['safe', 'generally_safe']
        assert 'classification_confidence' in safe_result
        assert safe_result['ml_classifier_used']
    
    def test_adverse_event_detection(self):
        """Test adverse event detection in study results"""
        agent = ContentQualityAgent()
        
        text_with_events = """
        Adverse events: 3 participants reported mild skin irritation.
        One case of allergic reaction. All events resolved within 48 hours.
        """
        
        text_without = "No adverse events reported. Excellent safety profile."
        
        detection_with = agent.detect_adverse_events(text_with_events)
        detection_without = agent.detect_adverse_events(text_without)
        
        assert detection_with['adverse_events_found']
        assert len(detection_with['event_descriptions']) > 0
        assert not detection_without['adverse_events_found']


class TestStandardsEnforcement:
    """Test suite for standards enforcement"""
    
    def test_check_reporting_standards(self):
        """Test reporting standards compliance check"""
        agent = ContentQualityAgent()
        
        compliant_text = """
        Abstract: [Complete]
        Introduction: [Complete]
        Methods: Randomized trial, n=100, power analysis conducted
        Results: Statistical analysis with p-values and confidence intervals
        Discussion: Limitations acknowledged, clinical implications discussed
        Conclusion: [Complete]
        References: [40 cited]
        """
        
        compliance = agent.check_reporting_standards(compliant_text)
        
        assert 'compliance_score' in compliance
        assert 'missing_sections' in compliance
        assert 'quality_indicators' in compliance
        assert compliance['compliance_score'] > 0.7
    
    def test_citation_quality_assessment(self):
        """Test citation quality assessment"""
        agent = ContentQualityAgent()
        
        references = [
            "Smith J, et al. J Cosmet Sci. 2020;45(3):123-135.",
            "Doe J. Personal communication.",
            "Website: www.example.com"
        ]
        
        quality = agent.assess_citation_quality(references)
        
        assert 'citation_quality_score' in quality
        assert 'peer_reviewed_count' in quality
        assert 'low_quality_sources' in quality


class TestQualityScoring:
    """Test suite for overall quality scoring"""
    
    def test_comprehensive_quality_score_structure(self):
        """Test comprehensive quality scoring structure"""
        agent = ContentQualityAgent()
        
        manuscript_data = {
            'title': 'Test Manuscript',
            'abstract': 'Comprehensive abstract...',
            'full_text': 'Full manuscript text...',
            'references': ['Ref 1', 'Ref 2']
        }
        
        score = agent.calculate_quality_score(manuscript_data)
        
        assert 'overall_quality' in score
        assert 'component_scores' in score
        assert 'recommendations' in score
        assert 0 <= score['overall_quality'] <= 1
    
    def test_quality_scoring_uses_real_ml(self):
        """Test quality scoring uses real ML models"""
        agent = ContentQualityAgent()
        
        high_quality = {
            'title': 'Rigorous Scientific Study',
            'full_text': 'RCT with 200 participants, p<0.001, 95% CI reported',
            'methodology_keywords': ['randomized', 'double-blind', 'controlled']
        }
        
        low_quality = {
            'title': 'Study',
            'full_text': 'We did a study. Results were good.',
            'methodology_keywords': []
        }
        
        high_score = agent.calculate_quality_score(high_quality)
        low_score = agent.calculate_quality_score(low_quality)
        
        assert high_score['overall_quality'] > low_score['overall_quality']
        assert high_score['ml_models_used']
        assert 'methodology' in high_score['component_scores']


class TestPlagiarismDetection:
    """Test suite for plagiarism detection"""
    
    def test_plagiarism_check_structure(self):
        """Test plagiarism check returns correct structure"""
        agent = ContentQualityAgent()
        
        text = "Sample manuscript text for testing."
        
        result = agent.check_plagiarism(text)
        
        assert 'plagiarism_score' in result
        assert 'similar_documents' in result
        assert 0 <= result['plagiarism_score'] <= 1
    
    def test_similarity_detection_uses_embeddings(self):
        """Test similarity detection uses real embeddings"""
        agent = ContentQualityAgent()
        
        original_text = "Novel approach to cosmetic formulation using peptides."
        similar_text = "Innovative method for cosmetic formulation with peptides."
        dissimilar_text = "Statistical analysis of economic data trends."
        
        sim_result = agent.calculate_text_similarity(original_text, similar_text)
        dissim_result = agent.calculate_text_similarity(original_text, dissimilar_text)
        
        # Similar texts should have higher similarity
        assert sim_result['similarity_score'] > dissim_result['similarity_score']
        assert sim_result['embedding_method'] == 'sentence_transformers'


# Fixtures
@pytest.fixture
def sample_manuscript():
    return {
        'title': 'Efficacy of Novel Anti-Aging Cream',
        'abstract': 'Randomized controlled trial of 120 participants...',
        'full_text': '''
        Introduction: Skin aging is a complex process.
        Methods: RCT, n=120, double-blind, power=0.8
        Results: Significant improvement (p<0.001, 95% CI: 0.4-0.8)
        Discussion: Findings align with previous research.
        ''',
        'references': [
            'Smith J. J Cosmet Sci. 2020.',
            'Doe J. Dermatology. 2019.'
        ]
    }
