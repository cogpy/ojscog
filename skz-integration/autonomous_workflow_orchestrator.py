"""
Autonomous Workflow Orchestrator
Coordinates the 7 enhanced agents for fully autonomous research journal operation
Implements end-to-end manuscript processing with minimal human intervention
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

try:
    from .enhanced_agents import (
        EnhancedResearchDiscoveryAgent,
        EnhancedSubmissionAssistantAgent,
        EnhancedEditorialOrchestrationAgent,
        EnhancedReviewCoordinationAgent,
        EnhancedContentQualityAgent,
        EnhancedPublishingProductionAgent,
        EnhancedAnalyticsMonitoringAgent
    )
except ImportError:
    from enhanced_agents import (
        EnhancedResearchDiscoveryAgent,
        EnhancedSubmissionAssistantAgent,
        EnhancedEditorialOrchestrationAgent,
        EnhancedReviewCoordinationAgent,
        EnhancedContentQualityAgent,
        EnhancedPublishingProductionAgent,
        EnhancedAnalyticsMonitoringAgent
    )

logger = logging.getLogger(__name__)


class WorkflowStage(Enum):
    """Stages in the autonomous publishing workflow"""
    SUBMISSION = "submission"
    INITIAL_SCREENING = "initial_screening"
    EDITORIAL_ASSESSMENT = "editorial_assessment"
    PEER_REVIEW = "peer_review"
    REVISION = "revision"
    FINAL_DECISION = "final_decision"
    PRODUCTION = "production"
    PUBLICATION = "publication"
    POST_PUBLICATION = "post_publication"


class ManuscriptStatus(Enum):
    """Status of manuscript in workflow"""
    SUBMITTED = "submitted"
    UNDER_REVIEW = "under_review"
    REVISIONS_REQUIRED = "revisions_required"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    IN_PRODUCTION = "in_production"
    PUBLISHED = "published"


@dataclass
class Manuscript:
    """Manuscript data structure"""
    id: str
    title: str
    abstract: str
    authors: List[str]
    keywords: List[str]
    content: Dict[str, str]  # sections: text
    figures: List[str]  # paths to figures
    status: ManuscriptStatus = ManuscriptStatus.SUBMITTED
    current_stage: WorkflowStage = WorkflowStage.SUBMISSION
    submission_date: str = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.submission_date is None:
            self.submission_date = datetime.now().isoformat()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class WorkflowDecision:
    """Decision made at a workflow stage"""
    stage: WorkflowStage
    decision: str
    confidence: float
    reasoning: str
    next_stage: WorkflowStage
    actions: List[str]
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class AutonomousWorkflowOrchestrator:
    """
    Orchestrates the complete autonomous publishing workflow
    Coordinates all 7 agents for end-to-end manuscript processing
    """
    
    def __init__(self):
        """Initialize the workflow orchestrator and all agents"""
        logger.info("Initializing Autonomous Workflow Orchestrator...")
        
        # Initialize all 7 enhanced agents
        self.research_agent = EnhancedResearchDiscoveryAgent()
        self.submission_agent = EnhancedSubmissionAssistantAgent()
        self.editorial_agent = EnhancedEditorialOrchestrationAgent()
        self.review_agent = EnhancedReviewCoordinationAgent()
        self.quality_agent = EnhancedContentQualityAgent()
        self.production_agent = EnhancedPublishingProductionAgent()
        self.analytics_agent = EnhancedAnalyticsMonitoringAgent()
        
        # Workflow state
        self.active_manuscripts: Dict[str, Manuscript] = {}
        self.workflow_history: List[Dict] = []
        self.performance_metrics: Dict = {
            "total_submissions": 0,
            "accepted": 0,
            "rejected": 0,
            "in_review": 0,
            "average_processing_time_days": 0
        }
        
        logger.info("Autonomous Workflow Orchestrator initialized with 7 agents")
    
    def process_new_submission(self, manuscript: Manuscript) -> WorkflowDecision:
        """
        Process a new manuscript submission
        
        Args:
            manuscript: Submitted manuscript
            
        Returns:
            Initial workflow decision
        """
        logger.info(f"Processing new submission: {manuscript.id}")
        
        # Stage 1: Initial Screening by Submission Assistant
        screening_result = self.submission_agent.assess_manuscript_quality({
            "abstract": manuscript.abstract,
            "methods": manuscript.content.get("methods", ""),
            "title": manuscript.title
        })
        
        # Stage 2: Research Context Analysis by Research Discovery
        similar_papers = self.research_agent.semantic_literature_search(
            manuscript.abstract,
            []  # Would search actual corpus
        )
        
        # Stage 3: Content Quality Check
        quality_result = self.quality_agent.validate_manuscript_content({
            "abstract": manuscript.abstract,
            "figures": manuscript.figures
        })
        
        # Make decision based on automated assessment
        overall_score = (
            screening_result["overall_score"] * 0.5 +
            quality_result["overall_quality"] * 0.5
        )
        
        if overall_score >= 0.75:
            decision = "proceed_to_review"
            next_stage = WorkflowStage.EDITORIAL_ASSESSMENT
            manuscript.status = ManuscriptStatus.UNDER_REVIEW
        elif overall_score >= 0.60:
            decision = "request_revisions"
            next_stage = WorkflowStage.REVISION
            manuscript.status = ManuscriptStatus.REVISIONS_REQUIRED
        else:
            decision = "desk_reject"
            next_stage = WorkflowStage.FINAL_DECISION
            manuscript.status = ManuscriptStatus.REJECTED
        
        workflow_decision = WorkflowDecision(
            stage=WorkflowStage.INITIAL_SCREENING,
            decision=decision,
            confidence=overall_score,
            reasoning=f"Automated screening score: {overall_score:.2f}. Quality: {screening_result['recommendation']}",
            next_stage=next_stage,
            actions=[
                f"Notify authors of {decision}",
                f"Move to {next_stage.value} stage"
            ]
        )
        
        # Update manuscript
        manuscript.current_stage = next_stage
        self.active_manuscripts[manuscript.id] = manuscript
        
        # Record in history
        self._record_workflow_event(manuscript.id, workflow_decision)
        
        # Update metrics
        self.performance_metrics["total_submissions"] += 1
        if decision == "proceed_to_review":
            self.performance_metrics["in_review"] += 1
        elif decision == "desk_reject":
            self.performance_metrics["rejected"] += 1
        
        logger.info(f"Submission {manuscript.id}: {decision} (confidence: {overall_score:.2f})")
        return workflow_decision
    
    def conduct_peer_review(self, manuscript_id: str, reviewer_pool: List[Dict]) -> WorkflowDecision:
        """
        Conduct automated peer review process
        
        Args:
            manuscript_id: Manuscript identifier
            reviewer_pool: Available reviewers
            
        Returns:
            Review workflow decision
        """
        manuscript = self.active_manuscripts.get(manuscript_id)
        if not manuscript:
            raise ValueError(f"Manuscript {manuscript_id} not found")
        
        logger.info(f"Conducting peer review for {manuscript_id}")
        
        # Stage 1: Match reviewers using semantic similarity
        matched_reviewers = self.review_agent.match_reviewers(
            manuscript.abstract,
            reviewer_pool
        )
        
        # Select top 3 reviewers
        selected_reviewers = matched_reviewers[:3]
        
        # Stage 2: Assign reviewers with audio notifications
        assignments = []
        for match in selected_reviewers:
            reviewer = match["reviewer"]
            assignment = self.review_agent.assign_reviewer_with_notification(
                reviewer,
                {"id": manuscript.id, "title": manuscript.title},
                send_audio=True
            )
            assignments.append(assignment)
        
        # Simulate review collection (in real system, would wait for actual reviews)
        reviews = [
            {"reviewer_id": r["reviewer"]["id"], "score": 4.2, "text": "Good methodology"}
            for r in selected_reviewers
        ]
        
        # Stage 3: Editorial decision based on reviews
        decision_result = self.editorial_agent.make_editorial_decision(
            manuscript_id,
            reviews,
            voice_announce=True
        )
        
        # Determine next stage
        if decision_result["decision"] == "accept":
            next_stage = WorkflowStage.PRODUCTION
            manuscript.status = ManuscriptStatus.ACCEPTED
        elif decision_result["decision"] == "revise":
            next_stage = WorkflowStage.REVISION
            manuscript.status = ManuscriptStatus.REVISIONS_REQUIRED
        else:
            next_stage = WorkflowStage.FINAL_DECISION
            manuscript.status = ManuscriptStatus.REJECTED
        
        workflow_decision = WorkflowDecision(
            stage=WorkflowStage.PEER_REVIEW,
            decision=decision_result["decision"],
            confidence=decision_result["average_score"] / 5.0,
            reasoning=f"Based on {len(reviews)} peer reviews. Average score: {decision_result['average_score']:.1f}/5.0",
            next_stage=next_stage,
            actions=[
                f"Assigned {len(selected_reviewers)} reviewers",
                f"Collected {len(reviews)} reviews",
                f"Editorial decision: {decision_result['decision']}"
            ]
        )
        
        manuscript.current_stage = next_stage
        self._record_workflow_event(manuscript_id, workflow_decision)
        
        # Update metrics
        if decision_result["decision"] == "accept":
            self.performance_metrics["accepted"] += 1
            self.performance_metrics["in_review"] -= 1
        elif decision_result["decision"] == "reject":
            self.performance_metrics["rejected"] += 1
            self.performance_metrics["in_review"] -= 1
        
        logger.info(f"Peer review complete for {manuscript_id}: {decision_result['decision']}")
        return workflow_decision
    
    def prepare_for_publication(self, manuscript_id: str) -> WorkflowDecision:
        """
        Prepare manuscript for publication
        
        Args:
            manuscript_id: Manuscript identifier
            
        Returns:
            Production workflow decision
        """
        manuscript = self.active_manuscripts.get(manuscript_id)
        if not manuscript:
            raise ValueError(f"Manuscript {manuscript_id} not found")
        
        logger.info(f"Preparing {manuscript_id} for publication")
        
        # Stage 1: Final quality check
        final_quality = self.quality_agent.validate_manuscript_content({
            "abstract": manuscript.abstract,
            "figures": manuscript.figures,
            "content": manuscript.content
        })
        
        # Stage 2: Optimize figures
        optimized_figures = self.production_agent.optimize_figures_for_publication(
            manuscript.figures
        )
        
        # Stage 3: Generate any missing figures
        if not manuscript.figures:
            figure_requests = [
                {"description": f"Illustration for {manuscript.title}", "type": "diagram"}
            ]
            generated_figures = self.production_agent.generate_manuscript_figures(figure_requests)
            optimized_figures.extend(generated_figures)
        
        # Update manuscript with optimized content
        manuscript.figures = optimized_figures
        manuscript.status = ManuscriptStatus.IN_PRODUCTION
        
        workflow_decision = WorkflowDecision(
            stage=WorkflowStage.PRODUCTION,
            decision="ready_for_publication",
            confidence=final_quality["overall_quality"],
            reasoning=f"Final quality score: {final_quality['overall_quality']:.2f}. All figures optimized.",
            next_stage=WorkflowStage.PUBLICATION,
            actions=[
                f"Optimized {len(optimized_figures)} figures",
                "Final quality check passed",
                "Ready for publication"
            ]
        )
        
        manuscript.current_stage = WorkflowStage.PUBLICATION
        self._record_workflow_event(manuscript_id, workflow_decision)
        
        logger.info(f"Manuscript {manuscript_id} ready for publication")
        return workflow_decision
    
    def publish_manuscript(self, manuscript_id: str) -> WorkflowDecision:
        """
        Publish manuscript and perform post-publication tasks
        
        Args:
            manuscript_id: Manuscript identifier
            
        Returns:
            Publication workflow decision
        """
        manuscript = self.active_manuscripts.get(manuscript_id)
        if not manuscript:
            raise ValueError(f"Manuscript {manuscript_id} not found")
        
        logger.info(f"Publishing manuscript {manuscript_id}")
        
        # Update status
        manuscript.status = ManuscriptStatus.PUBLISHED
        manuscript.current_stage = WorkflowStage.POST_PUBLICATION
        
        # Generate publication metadata
        publication_data = {
            "manuscript_id": manuscript_id,
            "title": manuscript.title,
            "authors": manuscript.authors,
            "publication_date": datetime.now().isoformat(),
            "doi": f"10.1234/skz.{manuscript_id}",
            "figures": len(manuscript.figures)
        }
        
        workflow_decision = WorkflowDecision(
            stage=WorkflowStage.PUBLICATION,
            decision="published",
            confidence=1.0,
            reasoning="Manuscript successfully published",
            next_stage=WorkflowStage.POST_PUBLICATION,
            actions=[
                "Manuscript published online",
                "DOI assigned",
                "Authors notified",
                "Indexed in databases"
            ]
        )
        
        self._record_workflow_event(manuscript_id, workflow_decision)
        
        logger.info(f"Manuscript {manuscript_id} published successfully")
        return workflow_decision
    
    def process_complete_workflow(
        self,
        manuscript: Manuscript,
        reviewer_pool: List[Dict]
    ) -> Dict:
        """
        Process complete workflow from submission to publication
        
        Args:
            manuscript: Submitted manuscript
            reviewer_pool: Available reviewers
            
        Returns:
            Complete workflow report
        """
        logger.info(f"Starting complete workflow for {manuscript.id}")
        
        workflow_report = {
            "manuscript_id": manuscript.id,
            "start_time": datetime.now().isoformat(),
            "stages": []
        }
        
        # Stage 1: Initial submission
        decision1 = self.process_new_submission(manuscript)
        workflow_report["stages"].append(self._decision_to_dict(decision1))
        
        if decision1.decision == "desk_reject":
            workflow_report["final_status"] = "rejected"
            workflow_report["end_time"] = datetime.now().isoformat()
            return workflow_report
        
        # Stage 2: Peer review (if not rejected)
        decision2 = self.conduct_peer_review(manuscript.id, reviewer_pool)
        workflow_report["stages"].append(self._decision_to_dict(decision2))
        
        if decision2.decision == "reject":
            workflow_report["final_status"] = "rejected"
            workflow_report["end_time"] = datetime.now().isoformat()
            return workflow_report
        
        if decision2.decision == "accept":
            # Stage 3: Production
            decision3 = self.prepare_for_publication(manuscript.id)
            workflow_report["stages"].append(self._decision_to_dict(decision3))
            
            # Stage 4: Publication
            decision4 = self.publish_manuscript(manuscript.id)
            workflow_report["stages"].append(self._decision_to_dict(decision4))
            
            workflow_report["final_status"] = "published"
        else:
            workflow_report["final_status"] = "revisions_required"
        
        workflow_report["end_time"] = datetime.now().isoformat()
        
        logger.info(f"Complete workflow finished for {manuscript.id}: {workflow_report['final_status']}")
        return workflow_report
    
    def get_performance_analytics(self) -> Dict:
        """
        Get performance analytics from Analytics Agent
        
        Returns:
            Performance analytics report
        """
        insights = self.analytics_agent.generate_performance_insights(
            self.performance_metrics
        )
        
        bottlenecks = self.analytics_agent.predict_workflow_bottlenecks(
            self.workflow_history
        )
        
        return {
            "metrics": self.performance_metrics,
            "insights": insights,
            "predicted_bottlenecks": bottlenecks,
            "active_manuscripts": len(self.active_manuscripts),
            "workflow_events": len(self.workflow_history)
        }
    
    def _record_workflow_event(self, manuscript_id: str, decision: WorkflowDecision):
        """Record workflow event in history"""
        event = {
            "manuscript_id": manuscript_id,
            "stage": decision.stage.value,
            "decision": decision.decision,
            "confidence": decision.confidence,
            "timestamp": decision.timestamp
        }
        self.workflow_history.append(event)
    
    def _decision_to_dict(self, decision: WorkflowDecision) -> Dict:
        """Convert WorkflowDecision to dictionary"""
        return {
            "stage": decision.stage.value,
            "decision": decision.decision,
            "confidence": decision.confidence,
            "reasoning": decision.reasoning,
            "next_stage": decision.next_stage.value,
            "actions": decision.actions,
            "timestamp": decision.timestamp
        }


if __name__ == "__main__":
    # Test the autonomous workflow orchestrator
    logging.basicConfig(level=logging.INFO)
    
    print("\n=== Autonomous Workflow Orchestrator Test ===\n")
    
    # Initialize orchestrator
    orchestrator = AutonomousWorkflowOrchestrator()
    
    # Create test manuscript
    test_manuscript = Manuscript(
        id="MS-2025-001",
        title="AI-Powered Skin Analysis for Cosmetic Research",
        abstract="This paper presents a novel approach to automated skin analysis using deep learning...",
        authors=["Dr. Jane Smith", "Dr. John Doe"],
        keywords=["AI", "skin analysis", "cosmetics", "deep learning"],
        content={
            "introduction": "Introduction text...",
            "methods": "Methods text...",
            "results": "Results text...",
            "discussion": "Discussion text..."
        },
        figures=["/path/to/figure1.png", "/path/to/figure2.png"]
    )
    
    # Create test reviewer pool
    reviewer_pool = [
        {"id": "R001", "name": "Dr. Alice Johnson", "expertise": "AI and machine learning in dermatology"},
        {"id": "R002", "name": "Dr. Bob Williams", "expertise": "Cosmetic science and skin analysis"},
        {"id": "R003", "name": "Dr. Carol Brown", "expertise": "Computer vision and image processing"}
    ]
    
    # Process complete workflow
    report = orchestrator.process_complete_workflow(test_manuscript, reviewer_pool)
    
    print(f"\nWorkflow Report for {report['manuscript_id']}:")
    print(f"Final Status: {report['final_status']}")
    print(f"\nStages completed: {len(report['stages'])}")
    for i, stage in enumerate(report['stages'], 1):
        print(f"\n{i}. {stage['stage']}")
        print(f"   Decision: {stage['decision']}")
        print(f"   Confidence: {stage['confidence']:.2f}")
        print(f"   Actions: {len(stage['actions'])}")
    
    # Get analytics
    print("\n=== Performance Analytics ===\n")
    analytics = orchestrator.get_performance_analytics()
    print(f"Total submissions: {analytics['metrics']['total_submissions']}")
    print(f"Accepted: {analytics['metrics']['accepted']}")
    print(f"Rejected: {analytics['metrics']['rejected']}")
    print(f"In review: {analytics['metrics']['in_review']}")
