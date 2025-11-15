#!/usr/bin/env python3
"""
OJSCog Database Initialization Script

Initializes Supabase and Neon databases with the required schemas
for the autonomous research journal system.

Author: OJSCog Team
Date: 2025-11-15
Version: 1.0
"""

import os
import sys
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def init_supabase_schema():
    """Initialize Supabase database schema."""
    try:
        from supabase import create_client
        
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_KEY')
        
        if not supabase_url or not supabase_key:
            logger.warning("Supabase credentials not found in environment variables")
            logger.info("Skipping Supabase initialization")
            return False
        
        logger.info("Initializing Supabase database...")
        
        # Create Supabase client
        supabase = create_client(supabase_url, supabase_key)
        
        # Read schema file
        schema_path = Path(__file__).parent / "supabase_schema.sql"
        with open(schema_path, 'r') as f:
            schema_sql = f.read()
        
        # Note: Supabase Python client doesn't support direct SQL execution
        # Schema should be applied via Supabase dashboard or CLI
        logger.info("Supabase schema file ready at: %s", schema_path)
        logger.info("Please apply the schema using Supabase dashboard or CLI:")
        logger.info("  supabase db push --db-url %s", supabase_url)
        
        # Verify connection
        try:
            result = supabase.table('agents_state').select('*').limit(1).execute()
            logger.info("✓ Supabase connection verified")
            logger.info("✓ agents_state table accessible")
            return True
        except Exception as e:
            logger.warning("Supabase tables not yet created: %s", e)
            logger.info("Please apply the schema file first")
            return False
    
    except ImportError:
        logger.error("Supabase client not installed. Install with: pip install supabase")
        return False
    except Exception as e:
        logger.error("Failed to initialize Supabase: %s", e)
        return False


def init_neon_schema():
    """Initialize Neon database schema."""
    try:
        import psycopg2
        from psycopg2 import sql
        
        # Get Neon connection string from environment
        neon_dsn = os.getenv('NEON_DATABASE_URL') or os.getenv('DATABASE_URL')
        
        if not neon_dsn:
            logger.warning("Neon database URL not found in environment variables")
            logger.info("Skipping Neon initialization")
            return False
        
        logger.info("Initializing Neon database...")
        
        # Connect to Neon
        conn = psycopg2.connect(neon_dsn)
        conn.autocommit = True
        cursor = conn.cursor()
        
        # Read schema file
        schema_path = Path(__file__).parent / "neon_hypergraph_schema.sql"
        with open(schema_path, 'r') as f:
            schema_sql = f.read()
        
        # Execute schema
        logger.info("Applying Neon hypergraph schema...")
        cursor.execute(schema_sql)
        
        logger.info("✓ Neon schema applied successfully")
        
        # Verify tables
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name LIKE 'hypergraph%'
        """)
        tables = cursor.fetchall()
        logger.info("✓ Created %d hypergraph tables", len(tables))
        
        cursor.close()
        conn.close()
        
        return True
    
    except ImportError:
        logger.error("psycopg2 not installed. Install with: pip install psycopg2-binary")
        return False
    except Exception as e:
        logger.error("Failed to initialize Neon: %s", e)
        return False


def verify_redis_connection():
    """Verify Redis connection."""
    try:
        import redis
        
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        
        logger.info("Verifying Redis connection...")
        
        r = redis.from_url(redis_url)
        r.ping()
        
        logger.info("✓ Redis connection verified")
        
        # Set a test key
        r.set('ojscog:init:test', 'success')
        value = r.get('ojscog:init:test')
        
        if value == b'success':
            logger.info("✓ Redis read/write verified")
            r.delete('ojscog:init:test')
            return True
        
        return False
    
    except ImportError:
        logger.error("Redis client not installed. Install with: pip install redis")
        return False
    except Exception as e:
        logger.error("Failed to connect to Redis: %s", e)
        return False


def create_initial_data():
    """Create initial data in databases."""
    try:
        from supabase import create_client
        
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_KEY')
        
        if not supabase_url or not supabase_key:
            logger.info("Skipping initial data creation (no Supabase credentials)")
            return False
        
        logger.info("Creating initial data...")
        
        supabase = create_client(supabase_url, supabase_key)
        
        # Check if agents already exist
        result = supabase.table('agents_state').select('agent_id').execute()
        
        if len(result.data) > 0:
            logger.info("Agents already exist, skipping initial data creation")
            return True
        
        # Create the 7 autonomous agents
        agents = [
            {
                'agent_id': 'agent_001',
                'agent_name': 'Research Discovery Agent',
                'agent_type': 'research_discovery',
                'current_phase': 'idle',
                'status': 'idle',
                'context': {},
                'memory': {},
                'performance_metrics': {}
            },
            {
                'agent_id': 'agent_002',
                'agent_name': 'Submission Assistant Agent',
                'agent_type': 'submission_assistant',
                'current_phase': 'idle',
                'status': 'idle',
                'context': {},
                'memory': {},
                'performance_metrics': {}
            },
            {
                'agent_id': 'agent_003',
                'agent_name': 'Editorial Orchestration Agent',
                'agent_type': 'editorial_orchestration',
                'current_phase': 'idle',
                'status': 'idle',
                'context': {},
                'memory': {},
                'performance_metrics': {}
            },
            {
                'agent_id': 'agent_004',
                'agent_name': 'Review Coordination Agent',
                'agent_type': 'review_coordination',
                'current_phase': 'idle',
                'status': 'idle',
                'context': {},
                'memory': {},
                'performance_metrics': {}
            },
            {
                'agent_id': 'agent_005',
                'agent_name': 'Content Quality Agent',
                'agent_type': 'content_quality',
                'current_phase': 'idle',
                'status': 'idle',
                'context': {},
                'memory': {},
                'performance_metrics': {}
            },
            {
                'agent_id': 'agent_006',
                'agent_name': 'Publishing Production Agent',
                'agent_type': 'publishing_production',
                'current_phase': 'idle',
                'status': 'idle',
                'context': {},
                'memory': {},
                'performance_metrics': {}
            },
            {
                'agent_id': 'agent_007',
                'agent_name': 'Analytics & Monitoring Agent',
                'agent_type': 'analytics_monitoring',
                'current_phase': 'idle',
                'status': 'idle',
                'context': {},
                'memory': {},
                'performance_metrics': {}
            }
        ]
        
        # Insert agents
        supabase.table('agents_state').insert(agents).execute()
        
        logger.info("✓ Created 7 autonomous agents")
        
        return True
    
    except Exception as e:
        logger.error("Failed to create initial data: %s", e)
        return False


def main():
    """Main initialization function."""
    logger.info("=" * 60)
    logger.info("OJSCog Database Initialization")
    logger.info("=" * 60)
    
    # Check for required environment variables
    logger.info("\nChecking environment configuration...")
    
    env_vars = {
        'SUPABASE_URL': os.getenv('SUPABASE_URL'),
        'SUPABASE_KEY': os.getenv('SUPABASE_KEY'),
        'DATABASE_URL': os.getenv('DATABASE_URL'),
        'REDIS_URL': os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    }
    
    for var, value in env_vars.items():
        if value:
            logger.info("✓ %s configured", var)
        else:
            logger.warning("✗ %s not configured", var)
    
    # Initialize databases
    results = {
        'supabase': init_supabase_schema(),
        'neon': init_neon_schema(),
        'redis': verify_redis_connection()
    }
    
    # Create initial data
    if results['supabase']:
        results['initial_data'] = create_initial_data()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Initialization Summary")
    logger.info("=" * 60)
    
    for component, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        logger.info("%s: %s", component.upper(), status)
    
    logger.info("=" * 60)
    
    if all(results.values()):
        logger.info("\n✓ All databases initialized successfully!")
        return 0
    else:
        logger.warning("\n⚠ Some databases failed to initialize")
        logger.info("Please check the logs above for details")
        return 1


if __name__ == "__main__":
    sys.exit(main())
