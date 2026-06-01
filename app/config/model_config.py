"""Model configuration for LLM calls - using DeepSeek by default"""

from typing import Optional

from app.config.settings import settings
from app.models import create_model, ModelType, BaseLLM
from app.utils.logger import get_logger

logger = get_logger(__name__)


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
    # Extract known kwargs to avoid duplicates with explicit params
    api_key = kwargs.pop("api_key", settings.DEEPSEEK_API_KEY)
    model_name = kwargs.pop("model_name", settings.DEEPSEEK_MODEL)
    base_url = kwargs.pop("base_url", settings.DEEPSEEK_BASE_URL)
    temperature = kwargs.pop("temperature", settings.LLM_TEMPERATURE)
    max_tokens = kwargs.pop("max_tokens", settings.LLM_MAX_TOKENS)

    return create_model(
        model_type=ModelType.DEEPSEEK,
        api_key=api_key,
        model_name=model_name,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs
    )
