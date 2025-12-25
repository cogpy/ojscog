"""
Environment Configuration Management System
Handles configuration loading, validation, and environment-specific settings
Provides centralized configuration access for all components
"""

import logging
import os
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict, field
from enum import Enum

logger = logging.getLogger(__name__)


class Environment(Enum):
    """Deployment environments"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


@dataclass
class DatabaseConfig:
    """Database configuration"""
    host: str = "localhost"
    port: int = 3306
    database: str = "ojs"
    user: str = "ojs_user"
    password: str = ""
    charset: str = "utf8mb4"
    pool_size: int = 10
    max_overflow: int = 20


@dataclass
class LLMConfig:
    """LLM configuration"""
    model_path: str = "/models/llama-7b-q4.gguf"
    backend: str = "ggml-cpu"
    context_length: int = 2048
    max_tokens: int = 512
    num_threads: int = 4
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    enable_cache: bool = True


@dataclass
class VisionConfig:
    """Vision processing configuration"""
    backend: str = "ncnn"
    use_gpu: bool = False
    image_size: List[int] = field(default_factory=lambda: [512, 512])
    batch_size: int = 1
    quality_threshold: float = 0.7


@dataclass
class SpeechConfig:
    """Speech processing configuration"""
    tts_voice: str = "neutral"
    tts_language: str = "en"
    tts_speed: float = 1.0
    stt_language: str = "en"
    stt_model_size: str = "base"
    sample_rate: int = 22050


@dataclass
class CacheConfig:
    """Cache configuration"""
    enabled: bool = True
    directory: str = "/tmp/cache"
    size_mb: int = 1024
    ttl_seconds: int = 3600


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    directory: str = "/var/log/ojscog"
    rotation: str = "daily"
    retention_days: int = 30
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


@dataclass
class MonitoringConfig:
    """Monitoring configuration"""
    enabled: bool = True
    metrics_port: int = 9090
    tracing_enabled: bool = False
    health_check_interval: int = 60


@dataclass
class SecurityConfig:
    """Security configuration"""
    enable_auth: bool = True
    jwt_secret_key: str = ""
    jwt_expiry_hours: int = 24
    api_rate_limit: int = 100
    cors_origins: List[str] = field(default_factory=list)


@dataclass
class FeatureFlags:
    """Feature flags"""
    enable_voice_commands: bool = True
    enable_audio_notifications: bool = True
    enable_figure_generation: bool = True
    enable_semantic_search: bool = True
    enable_auto_reviewer_matching: bool = True
    enable_predictive_analytics: bool = True


@dataclass
class ApplicationConfig:
    """Complete application configuration"""
    environment: Environment = Environment.PRODUCTION
    project_root: str = "/opt/ojscog"
    models_dir: str = "/models"
    data_dir: str = "/var/lib/ojscog"
    
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    speech: SpeechConfig = field(default_factory=SpeechConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    features: FeatureFlags = field(default_factory=FeatureFlags)


class ConfigLoader:
    """
    Configuration loader with multiple source support
    Loads from files, environment variables, and defaults
    """
    
    def __init__(self, config_dir: str = None):
        """
        Initialize config loader
        
        Args:
            config_dir: Directory containing config files
        """
        if config_dir is None:
            config_dir = os.getenv('CONFIG_DIR', '/etc/ojscog')
        
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
    
    def load_from_yaml(self, filename: str) -> Dict:
        """
        Load configuration from YAML file
        
        Args:
            filename: YAML filename
            
        Returns:
            Configuration dictionary
        """
        filepath = self.config_dir / filename
        
        if not filepath.exists():
            logger.warning(f"Config file not found: {filepath}")
            return {}
        
        try:
            with open(filepath, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded config from {filepath}")
            return config or {}
        except Exception as e:
            logger.error(f"Failed to load config from {filepath}: {e}")
            return {}
    
    def load_from_json(self, filename: str) -> Dict:
        """
        Load configuration from JSON file
        
        Args:
            filename: JSON filename
            
        Returns:
            Configuration dictionary
        """
        filepath = self.config_dir / filename
        
        if not filepath.exists():
            logger.warning(f"Config file not found: {filepath}")
            return {}
        
        try:
            with open(filepath, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded config from {filepath}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config from {filepath}: {e}")
            return {}
    
    def load_from_env(self, prefix: str = "OJSCOG_") -> Dict:
        """
        Load configuration from environment variables
        
        Args:
            prefix: Environment variable prefix
            
        Returns:
            Configuration dictionary
        """
        config = {}
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Remove prefix and convert to lowercase
                config_key = key[len(prefix):].lower()
                
                # Convert nested keys (e.g., DB_HOST -> db.host)
                parts = config_key.split('_')
                
                # Try to parse as JSON for complex values
                try:
                    parsed_value = json.loads(value)
                    value = parsed_value
                except (json.JSONDecodeError, TypeError):
                    # Keep as string, try to convert to appropriate type
                    if value.lower() in ('true', 'false'):
                        value = value.lower() == 'true'
                    elif value.isdigit():
                        value = int(value)
                    elif value.replace('.', '', 1).isdigit():
                        value = float(value)
                
                # Build nested dictionary
                current = config
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = value
        
        if config:
            logger.info(f"Loaded {len(config)} config values from environment")
        
        return config
    
    def save_to_yaml(self, config: Dict, filename: str):
        """
        Save configuration to YAML file
        
        Args:
            config: Configuration dictionary
            filename: Output filename
        """
        filepath = self.config_dir / filename
        
        try:
            with open(filepath, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            logger.info(f"Saved config to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save config to {filepath}: {e}")


class ConfigManager:
    """
    Central configuration management system
    Provides unified access to all configuration
    """
    
    def __init__(
        self,
        config_dir: str = None,
        environment: str = None,
        config_file: str = None
    ):
        """
        Initialize configuration manager
        
        Args:
            config_dir: Configuration directory
            environment: Deployment environment
            config_file: Specific config file to load
        """
        self.loader = ConfigLoader(config_dir)
        
        # Determine environment
        if environment is None:
            environment = os.getenv('DEPLOYMENT_ENV', 'production')
        self.environment = Environment(environment)
        
        # Load configuration
        self.config = self._load_configuration(config_file)
        
        # Validate configuration
        self._validate_configuration()
        
        logger.info(f"Configuration initialized for {self.environment.value} environment")
    
    def _load_configuration(self, config_file: str = None) -> ApplicationConfig:
        """
        Load configuration from multiple sources
        Priority: env vars > config file > defaults
        """
        # Start with defaults
        config_dict = asdict(ApplicationConfig())
        
        # Load from environment-specific file
        if config_file is None:
            config_file = f"config.{self.environment.value}.yaml"
        
        file_config = self.loader.load_from_yaml(config_file)
        config_dict = self._deep_merge(config_dict, file_config)
        
        # Load from environment variables (highest priority)
        env_config = self.loader.load_from_env()
        config_dict = self._deep_merge(config_dict, env_config)
        
        # Convert to ApplicationConfig
        return self._dict_to_config(config_dict)
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _dict_to_config(self, config_dict: Dict) -> ApplicationConfig:
        """Convert dictionary to ApplicationConfig"""
        # Convert nested dictionaries to dataclass instances
        if 'database' in config_dict and isinstance(config_dict['database'], dict):
            config_dict['database'] = DatabaseConfig(**config_dict['database'])
        
        if 'llm' in config_dict and isinstance(config_dict['llm'], dict):
            config_dict['llm'] = LLMConfig(**config_dict['llm'])
        
        if 'vision' in config_dict and isinstance(config_dict['vision'], dict):
            config_dict['vision'] = VisionConfig(**config_dict['vision'])
        
        if 'speech' in config_dict and isinstance(config_dict['speech'], dict):
            config_dict['speech'] = SpeechConfig(**config_dict['speech'])
        
        if 'cache' in config_dict and isinstance(config_dict['cache'], dict):
            config_dict['cache'] = CacheConfig(**config_dict['cache'])
        
        if 'logging' in config_dict and isinstance(config_dict['logging'], dict):
            config_dict['logging'] = LoggingConfig(**config_dict['logging'])
        
        if 'monitoring' in config_dict and isinstance(config_dict['monitoring'], dict):
            config_dict['monitoring'] = MonitoringConfig(**config_dict['monitoring'])
        
        if 'security' in config_dict and isinstance(config_dict['security'], dict):
            config_dict['security'] = SecurityConfig(**config_dict['security'])
        
        if 'features' in config_dict and isinstance(config_dict['features'], dict):
            config_dict['features'] = FeatureFlags(**config_dict['features'])
        
        # Convert environment string to enum
        if 'environment' in config_dict and isinstance(config_dict['environment'], str):
            config_dict['environment'] = Environment(config_dict['environment'])
        
        return ApplicationConfig(**config_dict)
    
    def _validate_configuration(self):
        """Validate configuration values"""
        errors = []
        
        # Validate database
        if not self.config.database.password and self.environment == Environment.PRODUCTION:
            errors.append("Database password is required in production")
        
        # Validate security
        if not self.config.security.jwt_secret_key and self.config.security.enable_auth:
            errors.append("JWT secret key is required when authentication is enabled")
        
        # Validate paths
        required_dirs = [
            self.config.project_root,
            self.config.models_dir,
            self.config.data_dir
        ]
        
        for dir_path in required_dirs:
            if not Path(dir_path).exists():
                logger.warning(f"Directory does not exist: {dir_path}")
        
        if errors:
            error_msg = "\n".join(errors)
            logger.error(f"Configuration validation failed:\n{error_msg}")
            raise ValueError(f"Invalid configuration:\n{error_msg}")
        
        logger.info("Configuration validation passed")
    
    def get_database_config(self) -> DatabaseConfig:
        """Get database configuration"""
        return self.config.database
    
    def get_llm_config(self) -> LLMConfig:
        """Get LLM configuration"""
        return self.config.llm
    
    def get_vision_config(self) -> VisionConfig:
        """Get vision configuration"""
        return self.config.vision
    
    def get_speech_config(self) -> SpeechConfig:
        """Get speech configuration"""
        return self.config.speech
    
    def is_feature_enabled(self, feature: str) -> bool:
        """
        Check if a feature is enabled
        
        Args:
            feature: Feature name
            
        Returns:
            True if enabled
        """
        return getattr(self.config.features, f"enable_{feature}", False)
    
    def export_config(self, format: str = "yaml") -> str:
        """
        Export configuration to string
        
        Args:
            format: Output format (yaml or json)
            
        Returns:
            Configuration as string
        """
        config_dict = asdict(self.config)
        
        # Convert enums to strings
        config_dict['environment'] = self.config.environment.value
        
        if format == "yaml":
            return yaml.dump(config_dict, default_flow_style=False)
        elif format == "json":
            return json.dumps(config_dict, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def save_config(self, filename: str = None):
        """
        Save current configuration to file
        
        Args:
            filename: Output filename
        """
        if filename is None:
            filename = f"config.{self.environment.value}.yaml"
        
        config_dict = asdict(self.config)
        config_dict['environment'] = self.config.environment.value
        
        self.loader.save_to_yaml(config_dict, filename)


# Global configuration instance
_config_manager: Optional[ConfigManager] = None


def get_config() -> ConfigManager:
    """
    Get global configuration manager instance
    
    Returns:
        ConfigManager instance
    """
    global _config_manager
    
    if _config_manager is None:
        _config_manager = ConfigManager()
    
    return _config_manager


def initialize_config(
    config_dir: str = None,
    environment: str = None,
    config_file: str = None
) -> ConfigManager:
    """
    Initialize global configuration
    
    Args:
        config_dir: Configuration directory
        environment: Deployment environment
        config_file: Specific config file
        
    Returns:
        ConfigManager instance
    """
    global _config_manager
    _config_manager = ConfigManager(config_dir, environment, config_file)
    return _config_manager


if __name__ == "__main__":
    # Test configuration manager
    logging.basicConfig(level=logging.INFO)
    
    print("\n=== Configuration Manager Test ===\n")
    
    # Initialize config
    config = get_config()
    
    print(f"Environment: {config.environment.value}")
    print(f"Project Root: {config.config.project_root}")
    print(f"Models Dir: {config.config.models_dir}")
    print()
    
    # Test database config
    db_config = config.get_database_config()
    print(f"Database: {db_config.database}@{db_config.host}:{db_config.port}")
    print()
    
    # Test LLM config
    llm_config = config.get_llm_config()
    print(f"LLM Model: {llm_config.model_path}")
    print(f"LLM Backend: {llm_config.backend}")
    print(f"Context Length: {llm_config.context_length}")
    print()
    
    # Test feature flags
    print("Feature Flags:")
    print(f"  Voice Commands: {config.is_feature_enabled('voice_commands')}")
    print(f"  Semantic Search: {config.is_feature_enabled('semantic_search')}")
    print(f"  Figure Generation: {config.is_feature_enabled('figure_generation')}")
    print()
    
    # Export config
    print("Exported Configuration (YAML):")
    print(config.export_config("yaml")[:500] + "...")
