"""Base interface for LLM models"""

from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod
from pydantic import BaseModel


class LLMConfig(BaseModel):
    """Configuration for LLM models"""

    # Model settings
    model_name: str = "deepseek-chat"
    temperature: float = 0.7
    max_tokens: int = 2000
    top_p: float = 0.9

    # API settings
    api_key: str = ""
    base_url: Optional[str] = None

    # Timeout settings
    timeout: int = 60

    # Additional kwargs
    extra_kwargs: Dict[str, Any] = {}


class BaseLLM(ABC):
    """
    Base class for LLM models.

    Provides a unified interface for different LLM providers.
    """

    def __init__(self, config: LLMConfig):
        """
        Initialize the LLM model.

        Args:
            config: LLM configuration
        """
        self.config = config
        self._client = None
        self._initialize()

    @abstractmethod
    def _initialize(self) -> None:
        """Initialize the underlying LLM client"""
        pass

    @abstractmethod
    def invoke(self, prompt: str, **kwargs) -> str:
        """
        Invoke the LLM with a prompt.

        Args:
            prompt: The prompt to send to the LLM
            **kwargs: Additional arguments

        Returns:
            The LLM response text
        """
        pass

    @abstractmethod
    async def ainvoke(self, prompt: str, **kwargs) -> str:
        """
        Async invoke the LLM with a prompt.

        Args:
            prompt: The prompt to send to the LLM
            **kwargs: Additional arguments

        Returns:
            The LLM response text
        """
        pass

    def get_model_name(self) -> str:
        """Get the model name"""
        return self.config.model_name

    def get_config(self) -> LLMConfig:
        """Get the configuration"""
        return self.config
