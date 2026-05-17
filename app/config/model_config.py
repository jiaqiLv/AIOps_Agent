"""Model configuration for LLM calls - using DeepSeek by default"""

from typing import Optional, Dict, Any
from pydantic import BaseModel

from app.config.settings import settings
from app.models import create_model, ModelType, BaseLLM
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ModelConfig(BaseModel):
    """LLM model configuration (legacy, kept for compatibility)"""

    model_name: str = "deepseek-chat"
    temperature: float = 0.7
    max_tokens: int = 2000
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

    # For structured output
    response_format: Optional[dict] = None


def get_llm(
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> BaseLLM:
    """
    Get an LLM instance based on current configuration.

    Args:
        provider: Model provider (deepseek, openai, anthropic). Defaults to settings.LLM_PROVIDER
        api_key: Optional API key override
        **kwargs: Additional configuration overrides

    Returns:
        Configured LLM instance
    """
    provider = provider or settings.LLM_PROVIDER

    # Determine API key based on provider
    if provider == "deepseek":
        api_key = api_key or settings.DEEPSEEK_API_KEY
    elif provider == "openai":
        api_key = api_key or settings.OPENAI_API_KEY
    elif provider == "anthropic":
        api_key = api_key or settings.ANTHROPIC_API_KEY
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    # Extract known kwargs to avoid duplicates
    temperature = kwargs.pop("temperature", settings.LLM_TEMPERATURE)
    max_tokens = kwargs.pop("max_tokens", settings.LLM_MAX_TOKENS)

    # Build kwargs with defaults from settings
    config_kwargs = {
        "temperature": temperature,
        "max_tokens": max_tokens,
        **kwargs
    }

    try:
        model_type = ModelType(provider)
        return create_model(
            model_type=model_type,
            api_key=api_key,
            **config_kwargs
        )
    except ValueError:
        raise ValueError(f"Unsupported provider: {provider}")


def get_deepseek_llm(**kwargs) -> BaseLLM:
    """
    Convenience function to get a DeepSeek LLM instance.

    Args:
        **kwargs: Configuration overrides

    Returns:
        Configured DeepSeek LLM instance
    """
    # Extract known kwargs to avoid duplicates
    temperature = kwargs.pop("temperature", settings.LLM_TEMPERATURE)
    max_tokens = kwargs.pop("max_tokens", settings.LLM_MAX_TOKENS)

    return create_model(
        model_type=ModelType.DEEPSEEK,
        api_key=settings.DEEPSEEK_API_KEY,
        model_name=settings.DEEPSEEK_MODEL,
        base_url=settings.DEEPSEEK_BASE_URL,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs
    )


# Default model configs (legacy, kept for compatibility)
DEFAULT_MODEL_CONFIG = ModelConfig(
    model_name="deepseek-chat",
    temperature=0.7,
    max_tokens=2000
)
FAST_MODEL_CONFIG = ModelConfig(
    model_name="deepseek-chat",
    temperature=0.3,
    max_tokens=1000
)
PRECISE_MODEL_CONFIG = ModelConfig(
    model_name="deepseek-chat",
    temperature=0.1,
    max_tokens=3000
)

# Default LLM instance (lazy loaded)
_default_llm: Optional[BaseLLM] = None


def get_default_llm() -> BaseLLM:
    """
    Get the default LLM instance (singleton pattern).

    Returns:
        Default LLM instance
    """
    global _default_llm
    if _default_llm is None:
        logger.info(f"Initializing default LLM: {settings.LLM_PROVIDER}")
        _default_llm = get_llm()
    return _default_llm
