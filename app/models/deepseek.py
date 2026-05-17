"""DeepSeek model implementation"""

from typing import Optional, Dict, Any, List, Union
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage

from app.models.base import BaseLLM, LLMConfig
from app.utils.logger import get_logger

logger = get_logger(__name__)


class DeepSeekModel(BaseLLM):
    """
    DeepSeek model implementation.

    DeepSeek API is compatible with OpenAI API, so we use ChatOpenAI
    with a custom base_url.
    """

    # Default DeepSeek API endpoint
    DEFAULT_BASE_URL = "https://api.deepseek.com"

    def _initialize(self) -> None:
        """Initialize the DeepSeek client"""
        base_url = self.config.base_url or self.DEFAULT_BASE_URL

        logger.info(f"Initializing DeepSeek model: {self.config.model_name}")
        logger.debug(f"Base URL: {base_url}")

        self._client = ChatOpenAI(
            model=self.config.model_name,
            api_key=self.config.api_key,
            base_url=base_url,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
            timeout=self.config.timeout,
            **self.config.extra_kwargs
        )

    def invoke(
        self,
        prompt: Union[str, List[BaseMessage]] = None,
        system_prompt: Optional[str] = None,
        messages: Optional[List[BaseMessage]] = None,
        **kwargs
    ) -> str:
        """
        Invoke the DeepSeek model.

        Args:
            prompt: The user prompt (string) or messages list (for LangGraph compatibility)
            system_prompt: Optional system prompt
            messages: Alternative way to pass messages (for LangGraph compatibility)
            **kwargs: Additional arguments

        Returns:
            The model response
        """
        try:
            # Handle different input formats for compatibility
            if messages is not None and isinstance(messages, list):
                # LangGraph format: list of BaseMessage objects
                message_list = messages
            elif isinstance(prompt, list):
                # LangGraph format: prompt is a list of BaseMessage objects
                message_list = prompt
            else:
                # Legacy format: prompt is a string
                message_list = []
                if system_prompt:
                    message_list.append(SystemMessage(content=system_prompt))
                if prompt:
                    if isinstance(prompt, str):
                        message_list.append(HumanMessage(content=prompt))
                    else:
                        message_list.append(prompt)

            if not message_list:
                raise ValueError("No messages provided to invoke")

            logger.debug(f"Sending {len(message_list)} messages to DeepSeek")

            response = self._client.invoke(message_list)

            result = response.content
            logger.debug(f"Received response from DeepSeek: {str(result)[:100]}...")

            return result

        except Exception as e:
            logger.error(f"Error invoking DeepSeek model: {e}")
            raise

    async def ainvoke(
        self,
        prompt: Union[str, List[BaseMessage]] = None,
        system_prompt: Optional[str] = None,
        messages: Optional[List[BaseMessage]] = None,
        **kwargs
    ) -> str:
        """
        Async invoke the DeepSeek model.

        Args:
            prompt: The user prompt (string) or messages list (for LangGraph compatibility)
            system_prompt: Optional system prompt
            messages: Alternative way to pass messages (for LangGraph compatibility)
            **kwargs: Additional arguments

        Returns:
            The model response
        """
        try:
            # Handle different input formats for compatibility
            if messages is not None and isinstance(messages, list):
                # LangGraph format: list of BaseMessage objects
                message_list = messages
            elif isinstance(prompt, list):
                # LangGraph format: prompt is a list of BaseMessage objects
                message_list = prompt
            else:
                # Legacy format: prompt is a string
                message_list = []
                if system_prompt:
                    message_list.append(SystemMessage(content=system_prompt))
                if prompt:
                    if isinstance(prompt, str):
                        message_list.append(HumanMessage(content=prompt))
                    else:
                        message_list.append(prompt)

            if not message_list:
                raise ValueError("No messages provided to ainvoke")

            logger.debug(f"Sending async {len(message_list)} messages to DeepSeek")

            response = await self._client.ainvoke(message_list)

            result = response.content
            logger.debug(f"Received async response from DeepSeek: {str(result)[:100]}...")

            return result

        except Exception as e:
            logger.error(f"Error async invoking DeepSeek model: {e}")
            raise

    def bind_tools(self, tools: list) -> "DeepSeekModel":
        """
        Bind tools to the model for function calling.

        Args:
            tools: List of tools to bind

        Returns:
            Self with tools bound
        """
        self._client = self._client.bind_tools(tools)
        return self

    def get_client(self) -> ChatOpenAI:
        """
        Get the underlying LangChain client.

        This is useful when you need to use the client directly
        with LangGraph's tool calling functionality.

        Returns:
            The ChatOpenAI client instance
        """
        return self._client

    def with_structured_output(self, schema: Dict[str, Any]) -> "DeepSeekModel":
        """
        Configure the model to return structured output.

        Args:
            schema: The output schema

        Returns:
            Self with structured output configured
        """
        self._client = self._client.with_structured_output(schema)
        return self
