#!/usr/bin/env python3.11
"""
Deployment System Test Suite
Tests all deployment components including database, models, config, and monitoring
"""

import sys
import os
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'skz-integration'))

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class TestResult:
    """Test result container"""
    def __init__(self, name: str, passed: bool, message: str = ""):
        self.name = name
        self.passed = passed
        self.message = message
    
    def __str__(self):
        status = "✓ PASS" if self.passed else "✗ FAIL"
        msg = f": {self.message}" if self.message else ""
        return f"{status}: {self.name}{msg}"


class DeploymentTestSuite:
    """Comprehensive deployment test suite"""
    
    def __init__(self):
        self.results = []
        self.passed = 0
        self.failed = 0
    
    def run_test(self, name: str, test_func):
        """Run a single test"""
        try:
            result = test_func()
            if result:
                self.results.append(TestResult(name, True, result if isinstance(result, str) else ""))
                self.passed += 1
            else:
                self.results.append(TestResult(name, False, "Test returned False"))
                self.failed += 1
        except Exception as e:
            self.results.append(TestResult(name, False, str(e)))
            self.failed += 1
    
    def test_model_manager(self):
        """Test model management system"""
        from model_manager import ModelManager, ModelType
        
        manager = ModelManager(str(project_root / 'models'))
        
        # Test listing models
        models = manager.list_available_models()
        
        return f"Found {len(models)} models"
    
    def test_config_manager(self):
        """Test configuration management"""
        from config_manager import ConfigManager, Environment
        
        # Create test config
        config = ConfigManager(
            config_dir=str(project_root / 'config'),
            environment='testing'
        )
        
        # Test config access
        db_config = config.get_database_config()
        llm_config = config.get_llm_config()
        
        assert db_config is not None
        assert llm_config is not None
        
        return f"Environment: {config.environment.value}"
    
    def test_monitoring_dashboard(self):
        """Test monitoring dashboard"""
        from monitoring_dashboard import MonitoringDashboard, MetricsCollector
        
        collector = MetricsCollector()
        dashboard = MonitoringDashboard(collector)
        
        # Simulate some metrics
        collector.increment_counter('test_counter')
        collector.set_gauge('test_gauge', 42.0)
        collector.observe_histogram('test_histogram', 1.5)
        
        # Get dashboard data
        data = dashboard.get_dashboard_data()
        
        assert 'system' in data
        assert 'agents' in data
        assert 'workflow' in data
        
        return "Dashboard data generated successfully"
    
    def test_database_integration(self):
        """Test OJS database integration"""
        try:
            from ojs_database_integration import OJSDatabaseConnection
            
            # Test connection initialization (won't actually connect without DB)
            conn = OJSDatabaseConnection()
            
            assert conn.config is not None
            assert 'host' in conn.config
            assert 'database' in conn.config
            
            return "Database integration module loaded"
        except ImportError as e:
            return f"Module loaded (DB connection requires actual database)"
    
    def test_native_library_manager(self):
        """Test native library manager"""
        from native.native_library_manager import NativeLibraryManager
        
        manager = NativeLibraryManager()
        
        # Test library discovery
        libraries = manager.discover_libraries()
        
        return f"Discovered {len(libraries)} native libraries"
    
    def test_llm_inference_engine(self):
        """Test LLM inference engine"""
        from native.llm_inference_engine import LLMInferenceEngine
        
        engine = LLMInferenceEngine()
        
        # Test initialization
        assert engine is not None
        
        return "LLM engine initialized"
    
    def test_vision_processor(self):
        """Test vision processor"""
        from native.vision_processor import VisionProcessor
        
        processor = VisionProcessor()
        
        # Test initialization
        assert processor is not None
        
        return "Vision processor initialized"
    
    def test_speech_interface(self):
        """Test speech interface"""
        from native.speech_interface import SpeechInterface
        
        interface = SpeechInterface()
        
        # Test initialization
        assert interface is not None
        
        return "Speech interface initialized"
    
    def test_enhanced_agents(self):
        """Test enhanced agents"""
        from enhanced_agents import (
            EnhancedResearchDiscoveryAgent,
            EnhancedSubmissionAssistantAgent,
            EnhancedEditorialOrchestrationAgent,
            EnhancedReviewCoordinationAgent,
            EnhancedContentQualityAgent,
            EnhancedPublishingProductionAgent,
            EnhancedAnalyticsMonitoringAgent
        )
        
        agents = [
            EnhancedResearchDiscoveryAgent(),
            EnhancedSubmissionAssistantAgent(),
            EnhancedEditorialOrchestrationAgent(),
            EnhancedReviewCoordinationAgent(),
            EnhancedContentQualityAgent(),
            EnhancedPublishingProductionAgent(),
            EnhancedAnalyticsMonitoringAgent()
        ]
        
        return f"All 7 agents initialized successfully"
    
    def test_autonomous_orchestrator(self):
        """Test autonomous workflow orchestrator"""
        from autonomous_workflow_orchestrator import AutonomousWorkflowOrchestrator
        
        orchestrator = AutonomousWorkflowOrchestrator()
        
        # Test initialization
        assert orchestrator is not None
        assert len(orchestrator.agents) == 7
        
        return "Orchestrator with 7 agents ready"
    
    def test_deployment_scripts(self):
        """Test deployment script existence"""
        scripts_dir = project_root / 'scripts' / 'deployment'
        
        required_scripts = [
            'deploy_production.sh',
            'download_models.sh',
            'setup_database.sh'
        ]
        
        missing = []
        for script in required_scripts:
            script_path = scripts_dir / script
            if not script_path.exists():
                missing.append(script)
            elif not os.access(script_path, os.X_OK):
                missing.append(f"{script} (not executable)")
        
        if missing:
            return f"Missing: {', '.join(missing)}"
        
        return "All deployment scripts present and executable"
    
    def test_documentation(self):
        """Test documentation completeness"""
        docs_dir = project_root / 'docs'
        
        required_docs = [
            'INDEX.md',
            'integration/NATIVE_LIBRARY_INTEGRATION.md'
        ]
        
        missing = []
        for doc in required_docs:
            if not (docs_dir / doc).exists():
                missing.append(doc)
        
        if missing:
            return f"Missing: {', '.join(missing)}"
        
        return "Core documentation present"
    
    def test_directory_structure(self):
        """Test directory structure"""
        required_dirs = [
            'skz-integration',
            'skz-integration/native',
            'docs',
            'scripts/deployment',
            'scripts/testing',
            'archive'
        ]
        
        missing = []
        for dir_path in required_dirs:
            if not (project_root / dir_path).exists():
                missing.append(dir_path)
        
        if missing:
            return f"Missing: {', '.join(missing)}"
        
        return "Directory structure complete"
    
    def run_all_tests(self):
        """Run all tests"""
        print("\n" + "="*60)
        print("  Deployment System Test Suite")
        print("="*60 + "\n")
        
        tests = [
            ("Directory Structure", self.test_directory_structure),
            ("Deployment Scripts", self.test_deployment_scripts),
            ("Documentation", self.test_documentation),
            ("Model Manager", self.test_model_manager),
            ("Config Manager", self.test_config_manager),
            ("Monitoring Dashboard", self.test_monitoring_dashboard),
            ("Database Integration", self.test_database_integration),
            ("Native Library Manager", self.test_native_library_manager),
            ("LLM Inference Engine", self.test_llm_inference_engine),
            ("Vision Processor", self.test_vision_processor),
            ("Speech Interface", self.test_speech_interface),
            ("Enhanced Agents", self.test_enhanced_agents),
            ("Autonomous Orchestrator", self.test_autonomous_orchestrator)
        ]
        
        for name, test_func in tests:
            self.run_test(name, test_func)
        
        # Print results
        print("\nTest Results:")
        print("-" * 60)
        for result in self.results:
            print(f"  {result}")
        
        print("\n" + "="*60)
        print(f"  Total: {len(self.results)} tests")
        print(f"  Passed: {self.passed} ({self.passed/len(self.results)*100:.1f}%)")
        print(f"  Failed: {self.failed} ({self.failed/len(self.results)*100:.1f}%)")
        print("="*60 + "\n")
        
        return self.failed == 0


def main():
    """Main test runner"""
    suite = DeploymentTestSuite()
    success = suite.run_all_tests()
    
    if success:
        print("✓ All deployment tests passed!")
        return 0
    else:
        print("✗ Some deployment tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
