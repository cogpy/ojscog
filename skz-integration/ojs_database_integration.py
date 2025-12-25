"""
OJS Database Integration Layer
Provides seamless integration between autonomous agents and OJS database
Handles manuscript data, user management, and workflow state synchronization
"""

import logging
import mysql.connector
from mysql.connector import Error
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import json
import os

logger = logging.getLogger(__name__)


class SubmissionStatus(Enum):
    """OJS submission status codes"""
    QUEUED = 1
    INCOMPLETE = 2
    DECLINED = 3
    PUBLISHED = 4
    SCHEDULED = 5


class ReviewRound(Enum):
    """Review round stages"""
    SUBMISSION = 1
    EXTERNAL_REVIEW = 2
    REVISION = 3
    COPYEDITING = 4
    PRODUCTION = 5


@dataclass
class OJSManuscript:
    """OJS manuscript data structure"""
    submission_id: int
    context_id: int
    title: str
    abstract: str
    authors: List[Dict[str, str]]
    keywords: List[str]
    status: SubmissionStatus
    stage_id: int
    date_submitted: datetime
    date_last_activity: datetime
    locale: str = "en_US"
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class OJSReviewer:
    """OJS reviewer data structure"""
    user_id: int
    username: str
    email: str
    first_name: str
    last_name: str
    affiliation: str
    country: str
    expertise: List[str]
    review_count: int = 0
    avg_completion_days: float = 0.0


@dataclass
class OJSReview:
    """OJS review data structure"""
    review_id: int
    submission_id: int
    reviewer_id: int
    round: int
    recommendation: Optional[str]
    comments: str
    date_assigned: datetime
    date_due: datetime
    date_completed: Optional[datetime]
    quality: int = 0  # 1-5 rating


class OJSDatabaseConnection:
    """
    Manages connection to OJS MySQL database
    Provides context manager for safe connection handling
    """
    
    def __init__(self, config: Dict[str, str] = None):
        """
        Initialize database connection
        
        Args:
            config: Database configuration dictionary
        """
        if config is None:
            config = self._load_config_from_env()
        
        self.config = config
        self.connection = None
        self.cursor = None
    
    def _load_config_from_env(self) -> Dict[str, str]:
        """Load database configuration from environment variables"""
        return {
            'host': os.getenv('OJS_DB_HOST', 'localhost'),
            'port': int(os.getenv('OJS_DB_PORT', '3306')),
            'database': os.getenv('OJS_DB_NAME', 'ojs'),
            'user': os.getenv('OJS_DB_USER', 'ojs_user'),
            'password': os.getenv('OJS_DB_PASSWORD', ''),
            'charset': 'utf8mb4',
            'use_unicode': True
        }
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
    
    def connect(self):
        """Establish database connection"""
        try:
            self.connection = mysql.connector.connect(**self.config)
            self.cursor = self.connection.cursor(dictionary=True)
            logger.info(f"Connected to OJS database: {self.config['database']}")
        except Error as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def close(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.connection and self.connection.is_connected():
            self.connection.close()
            logger.info("Database connection closed")
    
    def execute(self, query: str, params: Tuple = None) -> List[Dict]:
        """
        Execute SQL query
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            List of result dictionaries
        """
        try:
            self.cursor.execute(query, params or ())
            return self.cursor.fetchall()
        except Error as e:
            logger.error(f"Query execution failed: {e}")
            logger.error(f"Query: {query}")
            raise
    
    def execute_update(self, query: str, params: Tuple = None) -> int:
        """
        Execute UPDATE/INSERT/DELETE query
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            Number of affected rows
        """
        try:
            self.cursor.execute(query, params or ())
            self.connection.commit()
            return self.cursor.rowcount
        except Error as e:
            self.connection.rollback()
            logger.error(f"Update execution failed: {e}")
            raise


class OJSManuscriptRepository:
    """
    Repository for manuscript data access
    Provides high-level interface for manuscript operations
    """
    
    def __init__(self, db_connection: OJSDatabaseConnection):
        """Initialize repository with database connection"""
        self.db = db_connection
    
    def get_manuscript(self, submission_id: int) -> Optional[OJSManuscript]:
        """
        Retrieve manuscript by submission ID
        
        Args:
            submission_id: Submission ID
            
        Returns:
            OJSManuscript object or None
        """
        query = """
            SELECT 
                s.submission_id,
                s.context_id,
                s.stage_id,
                s.status,
                s.date_submitted,
                s.date_last_activity,
                s.locale,
                ps.setting_value as title,
                ps2.setting_value as abstract
            FROM submissions s
            LEFT JOIN publication_settings ps ON s.current_publication_id = ps.publication_id 
                AND ps.setting_name = 'title'
            LEFT JOIN publication_settings ps2 ON s.current_publication_id = ps2.publication_id 
                AND ps2.setting_name = 'abstract'
            WHERE s.submission_id = %s
        """
        
        results = self.db.execute(query, (submission_id,))
        
        if not results:
            return None
        
        row = results[0]
        
        # Get authors
        authors = self._get_manuscript_authors(submission_id)
        
        # Get keywords
        keywords = self._get_manuscript_keywords(submission_id)
        
        return OJSManuscript(
            submission_id=row['submission_id'],
            context_id=row['context_id'],
            title=row.get('title', ''),
            abstract=row.get('abstract', ''),
            authors=authors,
            keywords=keywords,
            status=SubmissionStatus(row['status']),
            stage_id=row['stage_id'],
            date_submitted=row['date_submitted'],
            date_last_activity=row['date_last_activity'],
            locale=row['locale']
        )
    
    def _get_manuscript_authors(self, submission_id: int) -> List[Dict[str, str]]:
        """Get authors for a manuscript"""
        query = """
            SELECT 
                a.author_id,
                a.email,
                a.first_name,
                a.middle_name,
                a.last_name,
                a.affiliation,
                a.country,
                a.seq
            FROM authors a
            JOIN publications p ON a.publication_id = p.publication_id
            JOIN submissions s ON p.publication_id = s.current_publication_id
            WHERE s.submission_id = %s
            ORDER BY a.seq
        """
        
        results = self.db.execute(query, (submission_id,))
        
        return [
            {
                'author_id': row['author_id'],
                'email': row['email'],
                'first_name': row['first_name'],
                'middle_name': row.get('middle_name', ''),
                'last_name': row['last_name'],
                'affiliation': row.get('affiliation', ''),
                'country': row.get('country', '')
            }
            for row in results
        ]
    
    def _get_manuscript_keywords(self, submission_id: int) -> List[str]:
        """Get keywords for a manuscript"""
        query = """
            SELECT ps.setting_value as keywords
            FROM publication_settings ps
            JOIN submissions s ON ps.publication_id = s.current_publication_id
            WHERE s.submission_id = %s AND ps.setting_name = 'keywords'
        """
        
        results = self.db.execute(query, (submission_id,))
        
        if results and results[0].get('keywords'):
            # Keywords are stored as JSON array
            try:
                keywords_data = json.loads(results[0]['keywords'])
                if isinstance(keywords_data, dict):
                    # Extract keywords from locale dict
                    return keywords_data.get('en_US', [])
                return keywords_data
            except json.JSONDecodeError:
                return []
        
        return []
    
    def get_pending_submissions(self, context_id: int = None) -> List[OJSManuscript]:
        """
        Get all pending submissions
        
        Args:
            context_id: Optional journal context ID
            
        Returns:
            List of pending manuscripts
        """
        query = """
            SELECT submission_id
            FROM submissions
            WHERE status = %s
        """
        params = [SubmissionStatus.QUEUED.value]
        
        if context_id:
            query += " AND context_id = %s"
            params.append(context_id)
        
        results = self.db.execute(query, tuple(params))
        
        manuscripts = []
        for row in results:
            manuscript = self.get_manuscript(row['submission_id'])
            if manuscript:
                manuscripts.append(manuscript)
        
        return manuscripts
    
    def update_submission_status(self, submission_id: int, status: SubmissionStatus) -> bool:
        """
        Update submission status
        
        Args:
            submission_id: Submission ID
            status: New status
            
        Returns:
            True if successful
        """
        query = """
            UPDATE submissions
            SET status = %s, date_last_activity = NOW()
            WHERE submission_id = %s
        """
        
        rows_affected = self.db.execute_update(query, (status.value, submission_id))
        
        if rows_affected > 0:
            logger.info(f"Updated submission {submission_id} status to {status.name}")
            return True
        
        return False
    
    def update_submission_stage(self, submission_id: int, stage_id: int) -> bool:
        """
        Update submission workflow stage
        
        Args:
            submission_id: Submission ID
            stage_id: New stage ID
            
        Returns:
            True if successful
        """
        query = """
            UPDATE submissions
            SET stage_id = %s, date_last_activity = NOW()
            WHERE submission_id = %s
        """
        
        rows_affected = self.db.execute_update(query, (stage_id, submission_id))
        
        if rows_affected > 0:
            logger.info(f"Updated submission {submission_id} stage to {stage_id}")
            return True
        
        return False


class OJSReviewerRepository:
    """
    Repository for reviewer data access
    Provides interface for reviewer operations
    """
    
    def __init__(self, db_connection: OJSDatabaseConnection):
        """Initialize repository with database connection"""
        self.db = db_connection
    
    def get_reviewer(self, user_id: int) -> Optional[OJSReviewer]:
        """
        Get reviewer by user ID
        
        Args:
            user_id: User ID
            
        Returns:
            OJSReviewer object or None
        """
        query = """
            SELECT 
                u.user_id,
                u.username,
                u.email,
                u.first_name,
                u.last_name,
                u.affiliation,
                u.country
            FROM users u
            JOIN user_user_groups uug ON u.user_id = uug.user_id
            JOIN user_groups ug ON uug.user_group_id = ug.user_group_id
            WHERE u.user_id = %s AND ug.role_id = 4096
        """
        
        results = self.db.execute(query, (user_id,))
        
        if not results:
            return None
        
        row = results[0]
        
        # Get reviewer expertise
        expertise = self._get_reviewer_expertise(user_id)
        
        # Get review statistics
        review_count, avg_days = self._get_reviewer_stats(user_id)
        
        return OJSReviewer(
            user_id=row['user_id'],
            username=row['username'],
            email=row['email'],
            first_name=row['first_name'],
            last_name=row['last_name'],
            affiliation=row.get('affiliation', ''),
            country=row.get('country', ''),
            expertise=expertise,
            review_count=review_count,
            avg_completion_days=avg_days
        )
    
    def _get_reviewer_expertise(self, user_id: int) -> List[str]:
        """Get reviewer expertise keywords"""
        query = """
            SELECT us.setting_value as expertise
            FROM user_settings us
            WHERE us.user_id = %s AND us.setting_name = 'reviewerInterests'
        """
        
        results = self.db.execute(query, (user_id,))
        
        if results and results[0].get('expertise'):
            try:
                expertise_data = json.loads(results[0]['expertise'])
                if isinstance(expertise_data, list):
                    return expertise_data
                elif isinstance(expertise_data, dict):
                    return expertise_data.get('en_US', [])
            except json.JSONDecodeError:
                return []
        
        return []
    
    def _get_reviewer_stats(self, user_id: int) -> Tuple[int, float]:
        """Get reviewer statistics"""
        query = """
            SELECT 
                COUNT(*) as review_count,
                AVG(DATEDIFF(date_completed, date_assigned)) as avg_days
            FROM review_assignments
            WHERE reviewer_id = %s AND date_completed IS NOT NULL
        """
        
        results = self.db.execute(query, (user_id,))
        
        if results:
            return (
                results[0].get('review_count', 0),
                results[0].get('avg_days', 0.0) or 0.0
            )
        
        return 0, 0.0
    
    def get_available_reviewers(self, context_id: int) -> List[OJSReviewer]:
        """
        Get all available reviewers for a journal
        
        Args:
            context_id: Journal context ID
            
        Returns:
            List of available reviewers
        """
        query = """
            SELECT DISTINCT u.user_id
            FROM users u
            JOIN user_user_groups uug ON u.user_id = uug.user_id
            JOIN user_groups ug ON uug.user_group_id = ug.user_group_id
            WHERE ug.context_id = %s AND ug.role_id = 4096
        """
        
        results = self.db.execute(query, (context_id,))
        
        reviewers = []
        for row in results:
            reviewer = self.get_reviewer(row['user_id'])
            if reviewer:
                reviewers.append(reviewer)
        
        return reviewers
    
    def assign_reviewer(
        self,
        submission_id: int,
        reviewer_id: int,
        round_id: int,
        due_date: datetime
    ) -> int:
        """
        Assign reviewer to submission
        
        Args:
            submission_id: Submission ID
            reviewer_id: Reviewer user ID
            round_id: Review round ID
            due_date: Review due date
            
        Returns:
            Review assignment ID
        """
        query = """
            INSERT INTO review_assignments (
                submission_id,
                reviewer_id,
                round,
                date_assigned,
                date_due,
                declined,
                cancelled
            ) VALUES (%s, %s, %s, NOW(), %s, 0, 0)
        """
        
        self.db.execute_update(
            query,
            (submission_id, reviewer_id, round_id, due_date)
        )
        
        # Get the inserted ID
        review_id = self.db.cursor.lastrowid
        
        logger.info(f"Assigned reviewer {reviewer_id} to submission {submission_id}")
        
        return review_id


class OJSIntegrationManager:
    """
    High-level manager for OJS database integration
    Provides unified interface for autonomous agents
    """
    
    def __init__(self, db_config: Dict[str, str] = None):
        """
        Initialize integration manager
        
        Args:
            db_config: Database configuration
        """
        self.db_config = db_config
        self.connection = None
        self.manuscript_repo = None
        self.reviewer_repo = None
    
    def __enter__(self):
        """Context manager entry"""
        self.connection = OJSDatabaseConnection(self.db_config)
        self.connection.connect()
        self.manuscript_repo = OJSManuscriptRepository(self.connection)
        self.reviewer_repo = OJSReviewerRepository(self.connection)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self.connection:
            self.connection.close()
    
    def sync_manuscript_to_agent_format(self, submission_id: int) -> Dict:
        """
        Convert OJS manuscript to agent-compatible format
        
        Args:
            submission_id: Submission ID
            
        Returns:
            Manuscript dictionary for agents
        """
        manuscript = self.manuscript_repo.get_manuscript(submission_id)
        
        if not manuscript:
            return None
        
        return {
            "id": f"MS-{manuscript.submission_id}",
            "title": manuscript.title,
            "abstract": manuscript.abstract,
            "authors": [
                f"{a['first_name']} {a['last_name']}" 
                for a in manuscript.authors
            ],
            "keywords": manuscript.keywords,
            "content": {
                "abstract": manuscript.abstract
            },
            "figures": [],  # Would need to extract from files
            "metadata": {
                "ojs_submission_id": manuscript.submission_id,
                "context_id": manuscript.context_id,
                "status": manuscript.status.name,
                "stage_id": manuscript.stage_id,
                "date_submitted": manuscript.date_submitted.isoformat()
            }
        }
    
    def sync_agent_decision_to_ojs(
        self,
        submission_id: int,
        decision: str,
        stage_id: int = None
    ) -> bool:
        """
        Sync agent decision back to OJS
        
        Args:
            submission_id: Submission ID
            decision: Decision (accept, reject, revise)
            stage_id: Optional new stage ID
            
        Returns:
            True if successful
        """
        # Map decision to OJS status
        status_map = {
            "accept": SubmissionStatus.PUBLISHED,
            "reject": SubmissionStatus.DECLINED,
            "revise": SubmissionStatus.QUEUED
        }
        
        status = status_map.get(decision.lower())
        
        if not status:
            logger.error(f"Unknown decision: {decision}")
            return False
        
        # Update status
        success = self.manuscript_repo.update_submission_status(submission_id, status)
        
        # Update stage if provided
        if success and stage_id:
            success = self.manuscript_repo.update_submission_stage(submission_id, stage_id)
        
        return success


if __name__ == "__main__":
    # Test the integration
    logging.basicConfig(level=logging.INFO)
    
    print("\n=== OJS Database Integration Test ===\n")
    
    # Test connection
    try:
        with OJSIntegrationManager() as manager:
            print("✓ Database connection successful")
            
            # Test manuscript retrieval
            manuscripts = manager.manuscript_repo.get_pending_submissions()
            print(f"✓ Found {len(manuscripts)} pending submissions")
            
            # Test reviewer retrieval
            reviewers = manager.reviewer_repo.get_available_reviewers(1)
            print(f"✓ Found {len(reviewers)} available reviewers")
            
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
