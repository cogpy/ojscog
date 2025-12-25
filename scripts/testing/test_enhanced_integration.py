#!/usr/bin/env python3.11
"""
Comprehensive Test Suite for Enhanced Agent Integration
Tests native library integration, enhanced agents, and autonomous workflow
"""

import sys
import os
import logging
from pathlib import Path

# Add skz-integration to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "skz-integration"))

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_native_library_manager():
    """Test native library manager functionality"""
    logger.info("Testing Native Library Manager...")
    
    try:
        from native.native_library_manager import get_library_manager, LibraryType
        
        manager = get_library_manager()
        
        # Test statistics
        stats = manager.get_statistics()
        assert stats["total_libraries"] > 0, "No libraries found"
        assert stats["total_size_mb"] > 0, "Total size should be positive"
        
        # Test library listing
        llm_libs = manager.list_available_libraries(LibraryType.LLM)
        assert len(llm_libs) > 0, "No LLM libraries found"
        
        vision_libs = manager.list_available_libraries(LibraryType.VISION)
        assert len(vision_libs) > 0, "No vision libraries found"
        
        speech_libs = manager.list_available_libraries(LibraryType.SPEECH)
        assert len(speech_libs) > 0, "No speech libraries found"
        
        # Test library info
        for lib_name in llm_libs[:3]:
            info = manager.get_library_info(lib_name)
            assert info is not None, f"Failed to get info for {lib_name}"
            assert "capabilities" in info, "Missing capabilities"
        
        logger.info(f"✓ Native Library Manager: {stats['total_libraries']} libraries, {stats['total_size_mb']:.2f} MB")
        return True
        
    except Exception as e:
        logger.error(f"✗ Native Library Manager test failed: {e}")
        return False


def test_llm_inference_engine():
    """Test LLM inference engine"""
    logger.info("Testing LLM Inference Engine...")
    
    try:
        from native.llm_inference_engine import (
            LLMInferenceEngine,
            AgentLLMInterface,
            InferenceConfig,
            ModelType,
            InferenceBackend
        )
        
        # Test configuration
        config = InferenceConfig(
            model_type=ModelType.LLAMA,
            model_path="/models/test.gguf",
            backend=InferenceBackend.GGML_CPU,
            context_length=1024
        )
        
        # Test engine initialization
        engine = LLMInferenceEngine(config)
        assert engine is not None, "Engine initialization failed"
        
        # Test agent interface
        agent_llm = AgentLLMInterface("Test Agent", config)
        assert agent_llm is not None, "Agent LLM interface initialization failed"
        
        # Test model info
        info = agent_llm.engine.get_model_info()
        assert "loaded" in info, "Missing loaded status"
        
        logger.info("✓ LLM Inference Engine: Initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"✗ LLM Inference Engine test failed: {e}")
        return False


def test_vision_processor():
    """Test vision processor"""
    logger.info("Testing Vision Processor...")
    
    try:
        from native.vision_processor import (
            VisionProcessor,
            PublishingVisionInterface,
            VisionConfig,
            VisionTask
        )
        
        # Test processor initialization
        processor = VisionProcessor()
        assert processor is not None, "Processor initialization failed"
        
        # Test publishing interface
        pub_vision = PublishingVisionInterface()
        assert pub_vision is not None, "Publishing interface initialization failed"
        
        logger.info("✓ Vision Processor: Initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"✗ Vision Processor test failed: {e}")
        return False


def test_speech_interface():
    """Test speech interface"""
    logger.info("Testing Speech Interface...")
    
    try:
        from native.speech_interface import (
            SpeechEngine,
            EditorialSpeechInterface,
            ReviewCoordinationSpeechInterface,
            TTSConfig
        )
        
        # Test engine initialization
        engine = SpeechEngine()
        assert engine is not None, "Engine initialization failed"
        
        # Test editorial interface
        editorial = EditorialSpeechInterface()
        assert editorial is not None, "Editorial interface initialization failed"
        
        # Test review coordination interface
        review = ReviewCoordinationSpeechInterface()
        assert review is not None, "Review interface initialization failed"
        
        logger.info("✓ Speech Interface: Initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"✗ Speech Interface test failed: {e}")
        return False


def test_enhanced_agents():
    """Test enhanced agents"""
    logger.info("Testing Enhanced Agents...")
    
    try:
        from enhanced_agents import (
            EnhancedResearchDiscoveryAgent,
            EnhancedSubmissionAssistantAgent,
            EnhancedEditorialOrchestrationAgent,
            EnhancedReviewCoordinationAgent,
            EnhancedContentQualityAgent,
            EnhancedPublishingProductionAgent,
            EnhancedAnalyticsMonitoringAgent,
            initialize_enhanced_agents
        )
        
        # Test individual agent initialization
        agents = [
            EnhancedResearchDiscoveryAgent(),
            EnhancedSubmissionAssistantAgent(),
            EnhancedEditorialOrchestrationAgent(),
            EnhancedReviewCoordinationAgent(),
            EnhancedContentQualityAgent(),
            EnhancedPublishingProductionAgent(),
            EnhancedAnalyticsMonitoringAgent()
        ]
        
        for agent in agents:
            assert agent is not None, f"Agent {agent.name} initialization failed"
            assert hasattr(agent, 'name'), "Agent missing name attribute"
        
        # Test batch initialization
        all_agents = initialize_enhanced_agents()
        assert len(all_agents) == 7, "Not all agents initialized"
        
        logger.info(f"✓ Enhanced Agents: All 7 agents initialized")
        return True
        
    except Exception as e:
        logger.error(f"✗ Enhanced Agents test failed: {e}")
        return False


def test_autonomous_workflow():
    """Test autonomous workflow orchestrator"""
    logger.info("Testing Autonomous Workflow Orchestrator...")
    
    try:
        from autonomous_workflow_orchestrator import (
            AutonomousWorkflowOrchestrator,
            Manuscript,
            ManuscriptStatus,
            WorkflowStage
        )
        
        # Test orchestrator initialization
        orchestrator = AutonomousWorkflowOrchestrator()
        assert orchestrator is not None, "Orchestrator initialization failed"
        
        # Test manuscript creation
        manuscript = Manuscript(
            id="TEST-001",
            title="Test Manuscript",
            abstract="Test abstract",
            authors=["Test Author"],
            keywords=["test"],
            content={"introduction": "Test intro"},
            figures=[]
        )
        
        assert manuscript.id == "TEST-001", "Manuscript ID mismatch"
        assert manuscript.status == ManuscriptStatus.SUBMITTED, "Wrong initial status"
        
        # Test performance analytics
        analytics = orchestrator.get_performance_analytics()
        assert "metrics" in analytics, "Missing metrics"
        assert "insights" in analytics, "Missing insights"
        
        logger.info("✓ Autonomous Workflow Orchestrator: Initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"✗ Autonomous Workflow test failed: {e}")
        return False


def test_integration_completeness():
    """Test integration completeness"""
    logger.info("Testing Integration Completeness...")
    
    try:
        # Check all required files exist
        base_path = Path(__file__).parent.parent.parent / "skz-integration"
        
        required_files = [
            "native/__init__.py",
            "native/native_library_manager.py",
            "native/llm_inference_engine.py",
            "native/vision_processor.py",
            "native/speech_interface.py",
            "enhanced_agents.py",
            "autonomous_workflow_orchestrator.py"
        ]
        
        missing_files = []
        for file_path in required_files:
            full_path = base_path / file_path
            if not full_path.exists():
                missing_files.append(file_path)
        
        assert len(missing_files) == 0, f"Missing files: {missing_files}"
        
        # Check native libraries exist
        native_lib_path = base_path / "native" / "arm64-v8a"
        assert native_lib_path.exists(), "Native library directory not found"
        
        lib_count = len(list(native_lib_path.glob("*.so")))
        assert lib_count > 0, "No native libraries found"
        
        logger.info(f"✓ Integration Completeness: All required files present, {lib_count} native libraries")
        return True
        
    except Exception as e:
        logger.error(f"✗ Integration Completeness test failed: {e}")
        return False


def run_all_tests():
    """Run all tests and report results"""
    logger.info("\n" + "="*60)
    logger.info("Enhanced Agent Integration Test Suite")
    logger.info("="*60 + "\n")
    
    tests = [
        ("Native Library Manager", test_native_library_manager),
        ("LLM Inference Engine", test_llm_inference_engine),
        ("Vision Processor", test_vision_processor),
        ("Speech Interface", test_speech_interface),
        ("Enhanced Agents", test_enhanced_agents),
        ("Autonomous Workflow", test_autonomous_workflow),
        ("Integration Completeness", test_integration_completeness)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
        print()  # Blank line between tests
    
    # Summary
    logger.info("="*60)
    logger.info("Test Summary")
    logger.info("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {test_name}")
    
    logger.info("-"*60)
    logger.info(f"Total: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    logger.info("="*60)
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
