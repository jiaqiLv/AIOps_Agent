"""Model factory for creating LLM instances"""

from enum import Enum
from typing import Dict, Any, Optional

from app.models.base import BaseLLM, LLMConfig
from app.models.deepseek import DeepSeekModel
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ModelType(str, Enum):
    """Supported model types"""

    DEEPSEEK = "deepseek"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


# Default model configurations for each provider
DEFAULT_MODEL_CONFIGS: Dict[ModelType, Dict[str, Any]] = {
    ModelType.DEEPSEEK: {
        "model_name": "deepseek-chat",
        "base_url": "https://api.deepseek.com",
    },
}


def create_model(
    model_type: ModelType = ModelType.DEEPSEEK,
    api_key: str = "",
    config: Optional[LLMConfig] = None,
    **kwargs
) -> BaseLLM:
    """
    Create an LLM model instance.

    Args:
        model_type: Type of model to create
        api_key: API key for the model provider
        config: Optional LLMConfig, if not provided uses defaults
        **kwargs: Additional configuration overrides

    Returns:
        Configured LLM model instance
    """
    logger.info(f"Creating model: {model_type.value if hasattr(model_type, 'value') else model_type}")

    # Get default config for the model type
    default_config = DEFAULT_MODEL_CONFIGS.get(model_type, {})

    # Merge configs: kwargs overrides default_config
    merged_config = {**default_config, **kwargs}

    # Create or update config
    if config is None:
        config = LLMConfig(
            api_key=api_key,
            **merged_config
        )
    else:
        # Update config with any provided overrides
        if api_key:
            config.api_key = api_key
        for key, value in merged_config.items():
            if hasattr(config, key):
                setattr(config, key, value)

    # Create the appropriate model instance
    if model_type == ModelType.DEEPSEEK:
        return DeepSeekModel(config)

    raise ValueError(f"Unsupported model type: {model_type}")


def create_deepseek_model(
    api_key: str,
    model_name: str = "deepseek-chat",
    **kwargs
) -> DeepSeekModel:
    """
    Convenience function to create a DeepSeek model.

    Args:
        api_key: DeepSeek API key
        model_name: Model name (default: deepseek-chat)
        **kwargs: Additional configuration

    Returns:
        Configured DeepSeek model
    """
    return DeepSeekModel(LLMConfig(
        api_key=api_key,
        model_name=model_name,
        base_url="https://api.deepseek.com",
        **kwargs
    ))
