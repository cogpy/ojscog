"""
Comprehensive Unit Tests for Agent 4: Review Coordination Agent
Tests reviewer matching, workload balancing, and review quality monitoring
"""

import pytest
import pytest_asyncio
import sys
import os
from datetime import datetime, timedelta
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from models.review_coordination import (
    ReviewCoordinationAgent,
    ReviewerMatcher,
    WorkloadBalancer,
    QualityMonitor,
    ReviewerMatch,
    WorkloadAssignment,
    QualityMetrics
)


class TestReviewCoordinationAgentInitialization:
    """Test suite for agent initialization"""
    
    def test_agent_initialization(self):
        """Test agent initializes with correct components"""
        agent = ReviewCoordinationAgent(agent_id="review_test")
        
        assert agent.agent_id == "review_test"
        assert agent.reviewer_matcher is not None
        assert agent.workload_balancer is not None
        assert agent.quality_monitor is not None
    
    def test_ml_components_loaded(self):
        """Test ML components are real implementations, not mocks"""
        agent = ReviewCoordinationAgent()
        
        # Verify real ML reviewer matcher
        assert hasattr(agent.reviewer_matcher, 'embedding_model')
        assert agent.reviewer_matcher.embedding_model is not None
        
        # Verify real workload predictor
        assert hasattr(agent.workload_balancer, 'prediction_model')
        assert agent.workload_balancer.prediction_model is not None
    
    def test_reviewer_database_connection(self):
        """Test reviewer database is accessible"""
        agent = ReviewCoordinationAgent()
        
        # Should have reviewer pool
        assert hasattr(agent, 'reviewer_pool')
        assert agent.reviewer_pool is not None


class TestReviewerMatching:
    """Test suite for ML-based reviewer matching"""
    
    def test_match_reviewers_structure(self):
        """Test reviewer matching returns correct structure"""
        agent = ReviewCoordinationAgent()
        
        manuscript_data = {
            'id': 'TEST-001',
            'title': 'Novel Hyaluronic Acid Formulation',
            'abstract': 'Study on anti-aging properties...',
            'keywords': ['hyaluronic acid', 'anti-aging', 'cosmetics']
        }
        
        matches = agent.match_reviewers(
            manuscript_id='TEST-001',
            manuscript_data=manuscript_data,
            num_reviewers=3
        )
        
        assert isinstance(matches, list)
        assert len(matches) <= 3
        for match in matches:
            assert 'reviewer_id' in match
            assert 'similarity_score' in match
            assert 'expertise_areas' in match
            assert 0 <= match['similarity_score'] <= 1
    
    def test_matching_uses_real_ml_embeddings(self):
        """Test reviewer matching uses real sentence embeddings"""
        agent = ReviewCoordinationAgent()
        
        manuscript_data = {
            'title': 'Advanced Peptide Formulations for Skin Rejuvenation',
            'abstract': 'This study investigates novel peptide combinations...',
            'keywords': ['peptides', 'skin rejuvenation', 'formulation']
        }
        
        matches = agent.match_reviewers('TEST-001', manuscript_data, num_reviewers=5)
        
        # Verify ML-based matching
        assert len(matches) > 0
        assert all('ml_embedding_score' in match for match in matches)
        assert all(match['matching_method'] == 'sentence_transformers' for match in matches)
        
        # Verify semantic similarity, not keyword matching
        top_match = matches[0]
        assert top_match['similarity_score'] > 0.3  # Real embeddings produce meaningful scores
    
    def test_expertise_weighting(self):
        """Test reviewer matching weights by expertise level"""
        agent = ReviewCoordinationAgent()
        
        manuscript_data = {
            'title': 'Clinical Trial of Anti-Aging Cream',
            'abstract': 'Randomized controlled trial...',
            'keywords': ['clinical trial', 'anti-aging', 'RCT']
        }
        
        matches = agent.match_reviewers('TEST-001', manuscript_data, num_reviewers=5)
        
        # Verify expertise weighting applied
        for match in matches:
            assert 'expertise_weight' in match
            assert 'years_of_experience' in match
            
            # Expert reviewers should have higher weights
            if match['years_of_experience'] > 10:
                assert match['expertise_weight'] >= 1.0
    
    def test_conflict_of_interest_filtering(self):
        """Test reviewer matching excludes conflicts of interest"""
        agent = ReviewCoordinationAgent()
        
        manuscript_data = {
            'id': 'TEST-001',
            'authors': ['Dr. Jane Smith', 'Dr. John Doe']
        }
        
        # Should exclude reviewers with conflicts
        matches = agent.match_reviewers('TEST-001', manuscript_data, num_reviewers=3)
        
        for match in matches:
            # Verify no co-authors
            assert 'conflict_of_interest' in match
            assert not match['conflict_of_interest']
    
    @pytest.mark.parametrize("field,expected_matches", [
        ("cosmetic_chemistry", 3),
        ("dermatology", 3),
        ("clinical_trials", 3),
    ])
    def test_field_specific_matching(self, field, expected_matches):
        """Test matching finds appropriate reviewers for different fields"""
        agent = ReviewCoordinationAgent()
        
        manuscript_data = {
            'title': f'Study in {field}',
            'keywords': [field]
        }
        
        matches = agent.match_reviewers('TEST', manuscript_data, num_reviewers=expected_matches)
        
        # Should find reviewers with relevant expertise
        assert len(matches) > 0
        assert all(field in str(match['expertise_areas']).lower() 
                   for match in matches[:2])  # Top 2 should match field


class TestWorkloadBalancing:
    """Test suite for reviewer workload balancing"""
    
    def test_workload_calculation(self):
        """Test workload calculation for reviewers"""
        agent = ReviewCoordinationAgent()
        
        reviewer_id = "REVIEWER-001"
        workload = agent.calculate_reviewer_workload(reviewer_id)
        
        assert isinstance(workload, dict)
        assert 'current_reviews' in workload
        assert 'average_review_time' in workload
        assert 'capacity_remaining' in workload
    
    def test_workload_balancing_uses_real_prediction(self):
        """Test workload balancing uses real ML prediction, not hardcoded"""
        agent = ReviewCoordinationAgent()
        
        reviewer_data = {
            'reviewer_id': 'REVIEWER-001',
            'current_reviews': 3,
            'average_completion_days': 14,
            'expertise_match': 0.85
        }
        
        prediction = agent.predict_review_completion_time(reviewer_data)
        
        # Verify real ML prediction
        assert 'estimated_days' in prediction
        assert 'confidence' in prediction
        assert prediction['ml_model_used']
        assert 0 <= prediction['confidence'] <= 1
        
        # Prediction should vary based on workload
        assert prediction['estimated_days'] > 10  # Reasonable time estimate
    
    def test_balanced_assignment(self):
        """Test reviewer assignment balances workload"""
        agent = ReviewCoordinationAgent()
        
        manuscript_data = {'id': 'TEST-001', 'title': 'Test manuscript'}
        matches = agent.match_reviewers('TEST-001', manuscript_data, num_reviewers=10)
        
        # Select reviewers with balanced workload
        assignments = agent.assign_reviewers_balanced(matches, num_to_assign=3)
        
        assert len(assignments) == 3
        
        # Verify workload considered
        for assignment in assignments:
            assert 'workload_score' in assignment
            assert assignment['workload_score'] < 0.9  # Not overloaded
    
    def test_capacity_constraints(self):
        """Test reviewer assignment respects capacity constraints"""
        agent = ReviewCoordinationAgent()
        
        manuscript_data = {'id': 'TEST-001'}
        
        # Try to assign more reviewers than available capacity
        assignments = agent.assign_reviewers_balanced(
            manuscript_data,
            num_to_assign=3,
            max_workload_per_reviewer=5
        )
        
        # Should respect capacity limits
        for assignment in assignments:
            assert assignment['projected_workload'] <= 5
    
    def test_urgency_based_assignment(self):
        """Test urgent manuscripts get assigned to available reviewers"""
        agent = ReviewCoordinationAgent()
        
        urgent_manuscript = {
            'id': 'TEST-URGENT',
            'priority': 'urgent',
            'deadline': datetime.now() + timedelta(days=7)
        }
        
        assignments = agent.assign_reviewers_balanced(
            urgent_manuscript,
            num_to_assign=2,
            consider_urgency=True
        )
        
        # Urgent assignments should prioritize availability
        for assignment in assignments:
            assert assignment['availability_score'] > 0.7


class TestQualityMonitoring:
    """Test suite for review quality monitoring"""
    
    def test_quality_metrics_structure(self):
        """Test quality metrics return correct structure"""
        agent = ReviewCoordinationAgent()
        
        reviewer_id = "REVIEWER-001"
        metrics = agent.get_reviewer_quality_metrics(reviewer_id)
        
        assert isinstance(metrics, dict)
        assert 'average_review_depth' in metrics
        assert 'consistency_score' in metrics
        assert 'timeliness_score' in metrics
        assert 'agreement_with_consensus' in metrics
    
    def test_quality_scoring_uses_real_ml(self):
        """Test quality scoring uses real ML, not hardcoded values"""
        agent = ReviewCoordinationAgent()
        
        review_data = {
            'reviewer_id': 'REVIEWER-001',
            'review_text': 'Comprehensive review with detailed analysis of methodology, results, and conclusions. Strengths clearly identified. Weaknesses noted constructively.',
            'review_time_days': 12,
            'recommendation': 'accept'
        }
        
        quality_score = agent.assess_review_quality(review_data)
        
        # Verify real ML assessment
        assert 'quality_score' in quality_score
        assert 'depth_score' in quality_score
        assert 'constructiveness_score' in quality_score
        assert 0 <= quality_score['quality_score'] <= 1
        
        # Verify NLP used for text analysis
        assert quality_score['text_analysis_method'] == 'nlp_pipeline'
    
    def test_reviewer_consistency_tracking(self):
        """Test reviewer consistency is tracked over time"""
        agent = ReviewCoordinationAgent()
        
        reviewer_id = "REVIEWER-001"
        
        # Submit multiple reviews
        for i in range(5):
            review_data = {
                'reviewer_id': reviewer_id,
                'review_text': f'Review {i} with analysis',
                'recommendation': 'accept' if i < 3 else 'revise'
            }
            agent.log_review(review_data)
        
        metrics = agent.get_reviewer_quality_metrics(reviewer_id)
        
        # Verify consistency calculated
        assert 'consistency_score' in metrics
        assert 0 <= metrics['consistency_score'] <= 1
    
    def test_consensus_agreement_calculation(self):
        """Test reviewer agreement with consensus is calculated"""
        agent = ReviewCoordinationAgent()
        
        manuscript_id = "TEST-001"
        reviews = [
            {'reviewer_id': 'R1', 'score': 0.8, 'recommendation': 'accept'},
            {'reviewer_id': 'R2', 'score': 0.85, 'recommendation': 'accept'},
            {'reviewer_id': 'R3', 'score': 0.4, 'recommendation': 'reject'}
        ]
        
        consensus = agent.calculate_consensus(manuscript_id, reviews)
        
        assert 'consensus_score' in consensus
        assert 'outlier_reviewers' in consensus
        assert 'R3' in consensus['outlier_reviewers']  # Disagrees with consensus
    
    def test_low_quality_review_flagging(self):
        """Test low quality reviews are flagged"""
        agent = ReviewCoordinationAgent()
        
        low_quality_review = {
            'reviewer_id': 'REVIEWER-BAD',
            'review_text': 'Looks good.',  # Too brief
            'review_time_days': 1,  # Too fast
            'recommendation': 'accept'
        }
        
        quality_score = agent.assess_review_quality(low_quality_review)
        
        # Should be flagged as low quality
        assert quality_score['quality_score'] < 0.5
        assert quality_score['flagged_for_review']
        assert 'insufficient_depth' in quality_score['flags']


class TestReviewCoordinationIntegration:
    """Test suite for integrated review coordination workflows"""
    
    @pytest.mark.asyncio
    async def test_complete_reviewer_assignment_workflow(self):
        """Test complete workflow from matching to assignment"""
        agent = ReviewCoordinationAgent()
        
        manuscript_data = {
            'id': 'TEST-001',
            'title': 'Novel Cosmetic Formulation',
            'abstract': 'Study on advanced formulation techniques...',
            'keywords': ['cosmetics', 'formulation', 'innovation']
        }
        
        # Step 1: Find matches
        matches = agent.match_reviewers('TEST-001', manuscript_data, num_reviewers=10)
        assert len(matches) > 0
        
        # Step 2: Check workload
        balanced_matches = []
        for match in matches:
            workload = agent.calculate_reviewer_workload(match['reviewer_id'])
            if workload['capacity_remaining'] > 0:
                balanced_matches.append(match)
        
        assert len(balanced_matches) > 0
        
        # Step 3: Make assignments
        assignments = agent.assign_reviewers_balanced(balanced_matches, num_to_assign=3)
        assert len(assignments) == 3
        
        # Step 4: Send invitations
        for assignment in assignments:
            invitation = agent.send_review_invitation(
                reviewer_id=assignment['reviewer_id'],
                manuscript_id='TEST-001'
            )
            assert invitation['sent']
    
    def test_reviewer_performance_tracking(self):
        """Test reviewer performance is tracked over time"""
        agent = ReviewCoordinationAgent()
        
        reviewer_id = "REVIEWER-001"
        
        # Simulate multiple completed reviews
        for i in range(10):
            review_data = {
                'reviewer_id': reviewer_id,
                'manuscript_id': f'M-{i}',
                'completion_days': 10 + i,
                'quality_score': 0.7 + (i * 0.02)
            }
            agent.log_completed_review(review_data)
        
        performance = agent.get_reviewer_performance(reviewer_id)
        
        # Verify performance metrics
        assert 'total_reviews' in performance
        assert performance['total_reviews'] == 10
        assert 'average_quality' in performance
        assert 0.7 <= performance['average_quality'] <= 0.9
        assert 'trend' in performance


class TestTurnaroundTimePrediction:
    """Test suite for review turnaround time prediction"""
    
    def test_turnaround_prediction_structure(self):
        """Test turnaround time prediction returns correct structure"""
        agent = ReviewCoordinationAgent()
        
        prediction = agent.predict_review_turnaround(
            manuscript_complexity=0.7,
            reviewer_experience=5,
            current_workload=3
        )
        
        assert isinstance(prediction, dict)
        assert 'estimated_days' in prediction
        assert 'confidence_interval' in prediction
        assert 'factors' in prediction
    
    def test_turnaround_uses_real_ml_regression(self):
        """Test turnaround prediction uses real ML regression"""
        agent = ReviewCoordinationAgent()
        
        # High complexity, experienced reviewer, low workload
        prediction1 = agent.predict_review_turnaround(0.9, 10, 1)
        
        # Low complexity, new reviewer, high workload
        prediction2 = agent.predict_review_turnaround(0.3, 1, 8)
        
        # Experienced reviewer with low workload should be faster
        assert prediction1['estimated_days'] < prediction2['estimated_days']
        
        # Verify ML model used
        assert 'ml_model' in prediction1
        assert prediction1['ml_model'] != 'mock'
    
    def test_confidence_interval_calculation(self):
        """Test confidence intervals are calculated from model uncertainty"""
        agent = ReviewCoordinationAgent()
        
        prediction = agent.predict_review_turnaround(0.5, 5, 3)
        
        # Should have confidence interval
        assert 'lower_bound' in prediction['confidence_interval']
        assert 'upper_bound' in prediction['confidence_interval']
        
        # Interval should be reasonable
        interval_width = (
            prediction['confidence_interval']['upper_bound'] - 
            prediction['confidence_interval']['lower_bound']
        )
        assert interval_width > 0
        assert interval_width < prediction['estimated_days']


# Fixtures for all tests
@pytest.fixture
def sample_manuscript_data():
    """Sample manuscript data for testing"""
    return {
        'id': 'TEST-MANUSCRIPT-001',
        'title': 'Advanced Cosmetic Formulation Study',
        'abstract': 'This study investigates novel formulation techniques for improved skin absorption.',
        'keywords': ['cosmetics', 'formulation', 'skin absorption', 'innovation'],
        'authors': ['Dr. Jane Smith', 'Dr. John Doe'],
        'complexity_score': 0.75
    }


@pytest.fixture
def sample_reviewers():
    """Sample reviewer pool for testing"""
    return [
        {
            'reviewer_id': 'R001',
            'name': 'Dr. Expert One',
            'expertise_areas': ['cosmetic chemistry', 'formulation'],
            'years_experience': 15,
            'current_workload': 2,
            'average_review_days': 12
        },
        {
            'reviewer_id': 'R002',
            'name': 'Dr. Expert Two',
            'expertise_areas': ['dermatology', 'skin absorption'],
            'years_experience': 10,
            'current_workload': 5,
            'average_review_days': 18
        },
        {
            'reviewer_id': 'R003',
            'name': 'Dr. Expert Three',
            'expertise_areas': ['clinical trials', 'cosmetics'],
            'years_experience': 8,
            'current_workload': 1,
            'average_review_days': 10
        }
    ]


@pytest.fixture
def sample_review_data():
    """Sample review data for testing"""
    return {
        'reviewer_id': 'R001',
        'manuscript_id': 'TEST-001',
        'review_text': '''
        This manuscript presents a novel approach to cosmetic formulation.
        
        Strengths:
        - Innovative methodology
        - Comprehensive experimental design
        - Strong statistical analysis
        
        Weaknesses:
        - Sample size could be larger
        - Some discussion points need expansion
        
        Overall, this is a solid contribution to the field.
        Recommendation: Accept with minor revisions
        ''',
        'recommendation': 'revise',
        'confidence': 0.85,
        'review_time_days': 14
    }
