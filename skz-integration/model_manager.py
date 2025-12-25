"""
Model Management System
Handles model lifecycle, versioning, and automatic updates
Provides centralized model access for all autonomous agents
"""

import logging
import json
import os
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import requests
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Types of AI models"""
    LLM = "llm"
    EMBEDDING = "embedding"
    VISION = "vision"
    SPEECH_TTS = "speech_tts"
    SPEECH_STT = "speech_stt"
    TOKENIZER = "tokenizer"


class ModelFormat(Enum):
    """Model file formats"""
    GGUF = "gguf"
    ONNX = "onnx"
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    NCNN = "ncnn"
    SAFETENSORS = "safetensors"


@dataclass
class ModelMetadata:
    """Metadata for a model"""
    model_id: str
    name: str
    model_type: ModelType
    format: ModelFormat
    version: str
    size_bytes: int
    path: str
    checksum: str
    source_url: Optional[str] = None
    description: Optional[str] = None
    parameters: Dict[str, Any] = None
    capabilities: List[str] = None
    requirements: Dict[str, str] = None
    date_added: str = None
    last_used: str = None
    
    def __post_init__(self):
        if self.date_added is None:
            self.date_added = datetime.now().isoformat()
        if self.parameters is None:
            self.parameters = {}
        if self.capabilities is None:
            self.capabilities = []
        if self.requirements is None:
            self.requirements = {}


class ModelRegistry:
    """
    Central registry for all AI models
    Tracks model metadata, versions, and usage
    """
    
    def __init__(self, registry_path: str = None):
        """
        Initialize model registry
        
        Args:
            registry_path: Path to registry JSON file
        """
        if registry_path is None:
            models_dir = os.getenv('MODELS_DIR', '/models')
            registry_path = os.path.join(models_dir, 'model_registry.json')
        
        self.registry_path = Path(registry_path)
        self.models: Dict[str, ModelMetadata] = {}
        
        self._load_registry()
    
    def _load_registry(self):
        """Load registry from file"""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r') as f:
                    data = json.load(f)
                
                # Convert to ModelMetadata objects
                for model_id, model_data in data.get('models', {}).items():
                    # Handle nested structure
                    if isinstance(model_data, dict) and 'primary' in model_data:
                        model_data = model_data['primary']
                    
                    if isinstance(model_data, dict):
                        self.models[model_id] = ModelMetadata(
                            model_id=model_id,
                            name=model_data.get('name', model_id),
                            model_type=ModelType(model_data.get('type', 'llm')),
                            format=ModelFormat(model_data.get('format', 'gguf')),
                            version=model_data.get('version', '1.0.0'),
                            size_bytes=model_data.get('size_bytes', 0),
                            path=model_data.get('path', ''),
                            checksum=model_data.get('checksum', ''),
                            source_url=model_data.get('source_url'),
                            description=model_data.get('description'),
                            parameters=model_data.get('parameters', {}),
                            capabilities=model_data.get('capabilities', []),
                            requirements=model_data.get('requirements', {}),
                            date_added=model_data.get('date_added'),
                            last_used=model_data.get('last_used')
                        )
                
                logger.info(f"Loaded {len(self.models)} models from registry")
            except Exception as e:
                logger.error(f"Failed to load registry: {e}")
        else:
            logger.info("Registry file not found, starting with empty registry")
    
    def _save_registry(self):
        """Save registry to file"""
        try:
            # Ensure directory exists
            self.registry_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert to JSON-serializable format
            data = {
                'version': '1.0.0',
                'updated': datetime.now().isoformat(),
                'models': {}
            }
            
            for model_id, model in self.models.items():
                model_dict = asdict(model)
                # Convert enums to strings
                model_dict['model_type'] = model.model_type.value
                model_dict['format'] = model.format.value
                data['models'][model_id] = model_dict
            
            with open(self.registry_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved registry with {len(self.models)} models")
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
    
    def register_model(self, metadata: ModelMetadata) -> bool:
        """
        Register a new model
        
        Args:
            metadata: Model metadata
            
        Returns:
            True if successful
        """
        try:
            self.models[metadata.model_id] = metadata
            self._save_registry()
            logger.info(f"Registered model: {metadata.model_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            return False
    
    def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """
        Get model metadata by ID
        
        Args:
            model_id: Model identifier
            
        Returns:
            ModelMetadata or None
        """
        return self.models.get(model_id)
    
    def list_models(
        self,
        model_type: ModelType = None,
        format: ModelFormat = None
    ) -> List[ModelMetadata]:
        """
        List models with optional filtering
        
        Args:
            model_type: Filter by model type
            format: Filter by format
            
        Returns:
            List of matching models
        """
        models = list(self.models.values())
        
        if model_type:
            models = [m for m in models if m.model_type == model_type]
        
        if format:
            models = [m for m in models if m.format == format]
        
        return models
    
    def update_last_used(self, model_id: str):
        """Update last used timestamp for a model"""
        if model_id in self.models:
            self.models[model_id].last_used = datetime.now().isoformat()
            self._save_registry()
    
    def remove_model(self, model_id: str) -> bool:
        """
        Remove model from registry
        
        Args:
            model_id: Model identifier
            
        Returns:
            True if successful
        """
        if model_id in self.models:
            del self.models[model_id]
            self._save_registry()
            logger.info(f"Removed model: {model_id}")
            return True
        return False


class ModelDownloader:
    """
    Handles model downloading from various sources
    Supports HuggingFace, direct URLs, and local files
    """
    
    def __init__(self, cache_dir: str = None):
        """
        Initialize model downloader
        
        Args:
            cache_dir: Directory for downloaded models
        """
        if cache_dir is None:
            cache_dir = os.getenv('MODELS_DIR', '/models')
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def download_from_url(
        self,
        url: str,
        output_path: str = None,
        show_progress: bool = True
    ) -> Optional[str]:
        """
        Download model from URL
        
        Args:
            url: Source URL
            output_path: Optional output path
            show_progress: Show download progress
            
        Returns:
            Path to downloaded file or None
        """
        try:
            if output_path is None:
                # Generate output path from URL
                filename = Path(urlparse(url).path).name
                output_path = self.cache_dir / filename
            else:
                output_path = Path(output_path)
            
            # Check if already downloaded
            if output_path.exists():
                logger.info(f"Model already exists: {output_path}")
                return str(output_path)
            
            logger.info(f"Downloading model from {url}")
            
            # Download with streaming
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(output_path, 'wb') as f:
                if show_progress and total_size > 0:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        downloaded += len(chunk)
                        progress = (downloaded / total_size) * 100
                        print(f"\rProgress: {progress:.1f}%", end='', flush=True)
                    print()  # New line after progress
                else:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            
            logger.info(f"Downloaded model to {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            return None
    
    def download_from_huggingface(
        self,
        repo_id: str,
        filename: str,
        output_path: str = None
    ) -> Optional[str]:
        """
        Download model from HuggingFace
        
        Args:
            repo_id: HuggingFace repository ID
            filename: Model filename
            output_path: Optional output path
            
        Returns:
            Path to downloaded file or None
        """
        url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
        return self.download_from_url(url, output_path)
    
    def calculate_checksum(self, file_path: str) -> str:
        """
        Calculate SHA256 checksum of file
        
        Args:
            file_path: Path to file
            
        Returns:
            Hex digest of checksum
        """
        sha256 = hashlib.sha256()
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        
        return sha256.hexdigest()
    
    def verify_checksum(self, file_path: str, expected_checksum: str) -> bool:
        """
        Verify file checksum
        
        Args:
            file_path: Path to file
            expected_checksum: Expected checksum
            
        Returns:
            True if checksum matches
        """
        actual_checksum = self.calculate_checksum(file_path)
        return actual_checksum == expected_checksum


class ModelManager:
    """
    High-level model management system
    Combines registry, downloader, and lifecycle management
    """
    
    def __init__(self, models_dir: str = None):
        """
        Initialize model manager
        
        Args:
            models_dir: Base directory for models
        """
        if models_dir is None:
            models_dir = os.getenv('MODELS_DIR', '/models')
        
        self.models_dir = Path(models_dir)
        self.registry = ModelRegistry(str(self.models_dir / 'model_registry.json'))
        self.downloader = ModelDownloader(str(self.models_dir))
    
    def add_model_from_url(
        self,
        model_id: str,
        name: str,
        model_type: ModelType,
        format: ModelFormat,
        url: str,
        version: str = "1.0.0",
        description: str = None,
        parameters: Dict = None
    ) -> bool:
        """
        Download and register model from URL
        
        Args:
            model_id: Unique model identifier
            name: Human-readable name
            model_type: Type of model
            format: Model format
            url: Source URL
            version: Model version
            description: Optional description
            parameters: Optional parameters
            
        Returns:
            True if successful
        """
        # Download model
        output_path = self.models_dir / model_type.value / f"{model_id}.{format.value}"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        downloaded_path = self.downloader.download_from_url(url, str(output_path))
        
        if not downloaded_path:
            return False
        
        # Calculate checksum
        checksum = self.downloader.calculate_checksum(downloaded_path)
        
        # Get file size
        size_bytes = Path(downloaded_path).stat().st_size
        
        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            name=name,
            model_type=model_type,
            format=format,
            version=version,
            size_bytes=size_bytes,
            path=downloaded_path,
            checksum=checksum,
            source_url=url,
            description=description,
            parameters=parameters or {}
        )
        
        # Register model
        return self.registry.register_model(metadata)
    
    def get_model_path(self, model_id: str) -> Optional[str]:
        """
        Get path to model file
        
        Args:
            model_id: Model identifier
            
        Returns:
            Path to model file or None
        """
        metadata = self.registry.get_model(model_id)
        
        if metadata:
            # Update last used
            self.registry.update_last_used(model_id)
            return metadata.path
        
        return None
    
    def list_available_models(self, model_type: ModelType = None) -> List[Dict]:
        """
        List all available models
        
        Args:
            model_type: Optional filter by type
            
        Returns:
            List of model information dictionaries
        """
        models = self.registry.list_models(model_type=model_type)
        
        return [
            {
                'id': m.model_id,
                'name': m.name,
                'type': m.model_type.value,
                'format': m.format.value,
                'version': m.version,
                'size_mb': m.size_bytes / (1024 * 1024),
                'path': m.path,
                'last_used': m.last_used
            }
            for m in models
        ]
    
    def delete_model(self, model_id: str, delete_file: bool = False) -> bool:
        """
        Delete model from registry and optionally from disk
        
        Args:
            model_id: Model identifier
            delete_file: Whether to delete the model file
            
        Returns:
            True if successful
        """
        metadata = self.registry.get_model(model_id)
        
        if not metadata:
            return False
        
        # Delete file if requested
        if delete_file:
            try:
                Path(metadata.path).unlink()
                logger.info(f"Deleted model file: {metadata.path}")
            except Exception as e:
                logger.error(f"Failed to delete model file: {e}")
        
        # Remove from registry
        return self.registry.remove_model(model_id)
    
    def get_model_info(self, model_id: str) -> Optional[Dict]:
        """
        Get detailed model information
        
        Args:
            model_id: Model identifier
            
        Returns:
            Model information dictionary or None
        """
        metadata = self.registry.get_model(model_id)
        
        if not metadata:
            return None
        
        return {
            'id': metadata.model_id,
            'name': metadata.name,
            'type': metadata.model_type.value,
            'format': metadata.format.value,
            'version': metadata.version,
            'size_bytes': metadata.size_bytes,
            'size_mb': metadata.size_bytes / (1024 * 1024),
            'path': metadata.path,
            'checksum': metadata.checksum,
            'source_url': metadata.source_url,
            'description': metadata.description,
            'parameters': metadata.parameters,
            'capabilities': metadata.capabilities,
            'requirements': metadata.requirements,
            'date_added': metadata.date_added,
            'last_used': metadata.last_used
        }


if __name__ == "__main__":
    # Test the model manager
    logging.basicConfig(level=logging.INFO)
    
    print("\n=== Model Manager Test ===\n")
    
    manager = ModelManager()
    
    # List available models
    models = manager.list_available_models()
    print(f"Available models: {len(models)}")
    
    for model in models:
        print(f"  - {model['name']} ({model['type']}) - {model['size_mb']:.2f} MB")
    
    # Test model path retrieval
    if models:
        model_id = models[0]['id']
        path = manager.get_model_path(model_id)
        print(f"\nModel path for {model_id}: {path}")
