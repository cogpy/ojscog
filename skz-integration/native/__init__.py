"""
Native Library Integration Module
Provides access to ARM64-v8a native libraries for AI/ML capabilities
"""

from .native_library_manager import (
    NativeLibraryManager,
    LibraryType,
    LibraryInfo,
    get_library_manager
)

from .llm_inference_engine import (
    LLMInferenceEngine,
    AgentLLMInterface,
    InferenceConfig,
    InferenceResult,
    ModelType,
    InferenceBackend
)

from .vision_processor import (
    VisionProcessor,
    PublishingVisionInterface,
    VisionConfig,
    VisionTask,
    ImageQuality,
    ImageGenerationRequest,
    ImageAnalysisResult
)

from .speech_interface import (
    SpeechEngine,
    EditorialSpeechInterface,
    ReviewCoordinationSpeechInterface,
    AccessibilitySpeechInterface,
    TTSConfig,
    STTConfig,
    SpeechTask,
    Voice,
    Language,
    SpeechResult
)

__all__ = [
    # Library Manager
    "NativeLibraryManager",
    "LibraryType",
    "LibraryInfo",
    "get_library_manager",
    
    # LLM Inference
    "LLMInferenceEngine",
    "AgentLLMInterface",
    "InferenceConfig",
    "InferenceResult",
    "ModelType",
    "InferenceBackend",
    
    # Vision Processing
    "VisionProcessor",
    "PublishingVisionInterface",
    "VisionConfig",
    "VisionTask",
    "ImageQuality",
    "ImageGenerationRequest",
    "ImageAnalysisResult",
    
    # Speech Interface
    "SpeechEngine",
    "EditorialSpeechInterface",
    "ReviewCoordinationSpeechInterface",
    "AccessibilitySpeechInterface",
    "TTSConfig",
    "STTConfig",
    "SpeechTask",
    "Voice",
    "Language",
    "SpeechResult",
]

__version__ = "1.0.0"
__author__ = "SKZ Integration Team"
