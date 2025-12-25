"""
Native Library Manager for ARM64-v8a Libraries
Provides unified interface for loading and managing native AI/ML libraries
"""

import os
import ctypes
import logging
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class LibraryType(Enum):
    """Types of native libraries available"""
    LLM = "llm"
    VISION = "vision"
    SPEECH = "speech"
    NLP = "nlp"
    MATH = "math"
    RUNTIME = "runtime"


@dataclass
class LibraryInfo:
    """Information about a native library"""
    name: str
    path: str
    type: LibraryType
    size_mb: float
    loaded: bool = False
    handle: Optional[Any] = None
    capabilities: list = None
    
    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = []


class NativeLibraryManager:
    """
    Manages loading and access to ARM64-v8a native libraries
    Provides unified interface for AI/ML capabilities
    """
    
    def __init__(self, library_dir: str = None):
        """
        Initialize the native library manager
        
        Args:
            library_dir: Path to directory containing native libraries
        """
        if library_dir is None:
            # Default to arm64-v8a directory in skz-integration
            base_dir = Path(__file__).parent
            library_dir = base_dir / "arm64-v8a"
        
        self.library_dir = Path(library_dir)
        self.loaded_libraries: Dict[str, LibraryInfo] = {}
        self.library_catalog = self._build_catalog()
        
        logger.info(f"Initialized NativeLibraryManager with {len(self.library_catalog)} libraries")
    
    def _build_catalog(self) -> Dict[str, LibraryInfo]:
        """Build catalog of available libraries"""
        catalog = {}
        
        # LLM Libraries
        llm_libs = {
            "llama": ("libllama.so", ["text-generation", "embeddings", "chat"]),
            "llama-jni": ("libllama-jni.so", ["java-binding", "android-support"]),
            "executorch": ("libexecutorch_llama_jni.so", ["optimized-inference", "mobile"]),
            "ggml": ("libggml.so", ["tensor-ops", "quantization"]),
            "ggml-base": ("libggml-base.so", ["core-operations"]),
            "ggml-cpu": ("libggml-cpu.so", ["cpu-inference"]),
            "ggml-opencl": ("libggml-opencl.so", ["gpu-acceleration", "opencl"]),
            "ggml-vulkan": ("libggml-vulkan.so", ["gpu-acceleration", "vulkan"]),
            "ggml-blas": ("libggml-blas.so", ["optimized-math"]),
            "ctranslate2": ("libctranslate2.so", ["transformer-inference", "optimization"]),
            "ctranslate2-jni": ("libctranslate2-jni.so", ["java-binding"]),
            "hermes": ("libhermes.so", ["javascript-engine", "ai-integration"]),
        }
        
        for name, (filename, caps) in llm_libs.items():
            path = self.library_dir / filename
            if path.exists():
                catalog[name] = LibraryInfo(
                    name=name,
                    path=str(path),
                    type=LibraryType.LLM,
                    size_mb=path.stat().st_size / (1024 * 1024),
                    capabilities=caps
                )
        
        # Vision Libraries
        vision_libs = {
            "image-generator": ("libimagegenerator_gpu.so", ["image-generation", "stable-diffusion"]),
            "mediapipe": ("libmediapipe_tasks_vision_image_generator_jni.so", ["vision-tasks", "image-gen"]),
            "ncnn": ("libncnn.so", ["neural-network", "mobile-inference"]),
            "sd-jni": ("libsd-jni.so", ["stable-diffusion", "java-binding"]),
            "rive": ("librive-android.so", ["animations", "graphics"]),
        }
        
        for name, (filename, caps) in vision_libs.items():
            path = self.library_dir / filename
            if path.exists():
                catalog[name] = LibraryInfo(
                    name=name,
                    path=str(path),
                    type=LibraryType.VISION,
                    size_mb=path.stat().st_size / (1024 * 1024),
                    capabilities=caps
                )
        
        # Speech Libraries
        speech_libs = {
            "espeak-ng": ("libespeak-ng.so", ["text-to-speech", "multilingual"]),
            "piper": ("libpiper_phonemize.so", ["phoneme-conversion", "tts"]),
            "kaldi-decoder": ("libkaldi-decoder-core.so", ["speech-recognition", "decoding"]),
            "kaldi-fbank": ("libkaldi-native-fbank-core.so", ["feature-extraction", "audio"]),
            "sherpa-onnx": ("libsherpa-onnx-jni.so", ["speech-processing", "onnx"]),
        }
        
        for name, (filename, caps) in speech_libs.items():
            path = self.library_dir / filename
            if path.exists():
                catalog[name] = LibraryInfo(
                    name=name,
                    path=str(path),
                    type=LibraryType.SPEECH,
                    size_mb=path.stat().st_size / (1024 * 1024),
                    capabilities=caps
                )
        
        # NLP Libraries
        nlp_libs = {
            "sentencepiece": ("libsentencepiece.so", ["tokenization", "subword"]),
            "sentencepiece-train": ("libsentencepiece_train.so", ["model-training"]),
            "sentencepiece-core": ("libssentencepiece_core.so", ["core-tokenization"]),
            "tokenizers": ("libtokenizers-jni.so", ["fast-tokenization", "transformers"]),
        }
        
        for name, (filename, caps) in nlp_libs.items():
            path = self.library_dir / filename
            if path.exists():
                catalog[name] = LibraryInfo(
                    name=name,
                    path=str(path),
                    type=LibraryType.NLP,
                    size_mb=path.stat().st_size / (1024 * 1024),
                    capabilities=caps
                )
        
        # Math Libraries
        math_libs = {
            "openblas": ("libopenblas.so", ["linear-algebra", "optimized"]),
            "omp": ("libomp.so", ["parallel-processing", "openmp"]),
        }
        
        for name, (filename, caps) in math_libs.items():
            path = self.library_dir / filename
            if path.exists():
                catalog[name] = LibraryInfo(
                    name=name,
                    path=str(path),
                    type=LibraryType.MATH,
                    size_mb=path.stat().st_size / (1024 * 1024),
                    capabilities=caps
                )
        
        # Runtime Libraries
        runtime_libs = {
            "onnxruntime": ("libonnxruntime.so", ["onnx-inference", "cross-platform"]),
            "onnxruntime4j": ("libonnxruntime4j_jni.so", ["java-binding"]),
            "onnxruntimejsi": ("libonnxruntimejsihelper.so", ["javascript-binding"]),
            "tvm": ("libtvm4j_runtime_packed.so", ["optimized-inference", "compilation"]),
            "qnn": ("liblaylaQNN.so", ["qualcomm-acceleration", "npu"]),
        }
        
        for name, (filename, caps) in runtime_libs.items():
            path = self.library_dir / filename
            if path.exists():
                catalog[name] = LibraryInfo(
                    name=name,
                    path=str(path),
                    type=LibraryType.RUNTIME,
                    size_mb=path.stat().st_size / (1024 * 1024),
                    capabilities=caps
                )
        
        return catalog
    
    def load_library(self, name: str) -> bool:
        """
        Load a native library by name
        
        Args:
            name: Library name from catalog
            
        Returns:
            True if loaded successfully, False otherwise
        """
        if name not in self.library_catalog:
            logger.error(f"Library '{name}' not found in catalog")
            return False
        
        if name in self.loaded_libraries:
            logger.info(f"Library '{name}' already loaded")
            return True
        
        lib_info = self.library_catalog[name]
        
        try:
            # Load the library using ctypes
            handle = ctypes.CDLL(lib_info.path)
            lib_info.handle = handle
            lib_info.loaded = True
            self.loaded_libraries[name] = lib_info
            
            logger.info(f"Successfully loaded library '{name}' ({lib_info.size_mb:.2f} MB)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load library '{name}': {e}")
            return False
    
    def load_libraries_by_type(self, lib_type: LibraryType) -> int:
        """
        Load all libraries of a specific type
        
        Args:
            lib_type: Type of libraries to load
            
        Returns:
            Number of libraries successfully loaded
        """
        count = 0
        for name, lib_info in self.library_catalog.items():
            if lib_info.type == lib_type:
                if self.load_library(name):
                    count += 1
        
        logger.info(f"Loaded {count} libraries of type {lib_type.value}")
        return count
    
    def get_library(self, name: str) -> Optional[LibraryInfo]:
        """Get information about a loaded library"""
        return self.loaded_libraries.get(name)
    
    def list_available_libraries(self, lib_type: LibraryType = None) -> list:
        """
        List available libraries, optionally filtered by type
        
        Args:
            lib_type: Optional library type filter
            
        Returns:
            List of library names
        """
        if lib_type is None:
            return list(self.library_catalog.keys())
        
        return [
            name for name, lib_info in self.library_catalog.items()
            if lib_info.type == lib_type
        ]
    
    def get_capabilities(self, name: str) -> list:
        """Get capabilities of a library"""
        lib_info = self.library_catalog.get(name)
        if lib_info:
            return lib_info.capabilities
        return []
    
    def get_library_info(self, name: str) -> Optional[Dict]:
        """Get detailed information about a library"""
        lib_info = self.library_catalog.get(name)
        if lib_info:
            return {
                "name": lib_info.name,
                "path": lib_info.path,
                "type": lib_info.type.value,
                "size_mb": lib_info.size_mb,
                "loaded": lib_info.loaded,
                "capabilities": lib_info.capabilities
            }
        return None
    
    def get_statistics(self) -> Dict:
        """Get statistics about available and loaded libraries"""
        stats = {
            "total_libraries": len(self.library_catalog),
            "loaded_libraries": len(self.loaded_libraries),
            "total_size_mb": sum(lib.size_mb for lib in self.library_catalog.values()),
            "loaded_size_mb": sum(lib.size_mb for lib in self.loaded_libraries.values()),
            "by_type": {}
        }
        
        for lib_type in LibraryType:
            type_libs = [lib for lib in self.library_catalog.values() if lib.type == lib_type]
            loaded_type_libs = [lib for lib in self.loaded_libraries.values() if lib.type == lib_type]
            
            stats["by_type"][lib_type.value] = {
                "available": len(type_libs),
                "loaded": len(loaded_type_libs),
                "size_mb": sum(lib.size_mb for lib in type_libs)
            }
        
        return stats
    
    def unload_library(self, name: str) -> bool:
        """Unload a library from memory"""
        if name not in self.loaded_libraries:
            logger.warning(f"Library '{name}' is not loaded")
            return False
        
        try:
            lib_info = self.loaded_libraries[name]
            # Note: ctypes doesn't provide a direct way to unload libraries
            # This marks it as unloaded in our tracking
            lib_info.loaded = False
            lib_info.handle = None
            del self.loaded_libraries[name]
            
            logger.info(f"Unloaded library '{name}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unload library '{name}': {e}")
            return False
    
    def unload_all(self):
        """Unload all loaded libraries"""
        for name in list(self.loaded_libraries.keys()):
            self.unload_library(name)
        
        logger.info("Unloaded all libraries")


# Singleton instance
_manager_instance: Optional[NativeLibraryManager] = None


def get_library_manager() -> NativeLibraryManager:
    """Get singleton instance of NativeLibraryManager"""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = NativeLibraryManager()
    return _manager_instance


if __name__ == "__main__":
    # Test the library manager
    logging.basicConfig(level=logging.INFO)
    
    manager = get_library_manager()
    
    print("\n=== Native Library Manager ===\n")
    
    # Show statistics
    stats = manager.get_statistics()
    print(f"Total libraries: {stats['total_libraries']}")
    print(f"Total size: {stats['total_size_mb']:.2f} MB")
    print("\nLibraries by type:")
    for lib_type, type_stats in stats['by_type'].items():
        print(f"  {lib_type}: {type_stats['available']} libraries ({type_stats['size_mb']:.2f} MB)")
    
    # List LLM libraries
    print("\n=== LLM Libraries ===")
    llm_libs = manager.list_available_libraries(LibraryType.LLM)
    for lib_name in llm_libs:
        info = manager.get_library_info(lib_name)
        print(f"  {lib_name}: {info['size_mb']:.2f} MB - {', '.join(info['capabilities'])}")
    
    # List Vision libraries
    print("\n=== Vision Libraries ===")
    vision_libs = manager.list_available_libraries(LibraryType.VISION)
    for lib_name in vision_libs:
        info = manager.get_library_info(lib_name)
        print(f"  {lib_name}: {info['size_mb']:.2f} MB - {', '.join(info['capabilities'])}")
    
    # List Speech libraries
    print("\n=== Speech Libraries ===")
    speech_libs = manager.list_available_libraries(LibraryType.SPEECH)
    for lib_name in speech_libs:
        info = manager.get_library_info(lib_name)
        print(f"  {lib_name}: {info['size_mb']:.2f} MB - {', '.join(info['capabilities'])}")
