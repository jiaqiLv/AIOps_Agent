"""Application configuration settings"""

import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    """Application configuration"""

    # Project basic info
    APP_NAME: str = "aiops-diagnose-agent"
    VERSION: str = "0.1.0"

    # Model provider selection (deepseek, openai, anthropic)
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "deepseek")

    # DeepSeek API
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    # OpenAI API (optional, for backward compatibility)
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "")

    # Anthropic API (optional)
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

    # LangSmith (optional)
    LANGCHAIN_TRACING_V2: bool = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    LANGCHAIN_API_KEY: str = os.getenv("LANGCHAIN_API_KEY", "")
    LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "aiops-diagnose-agent")

    # LangGraph
    LANGGRAPH_DEBUG: bool = os.getenv("LANGGRAPH_DEBUG", "true").lower() == "true"

    # MCP Tools (reserved for future implementation)
    RCD_MCP_SERVER_URL: str = os.getenv("RCD_MCP_SERVER_URL", "http://localhost:8001")
    PC_MCP_SERVER_URL: str = os.getenv("PC_MCP_SERVER_URL", "http://localhost:8002")

    # LLM default settings
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "2000"))

    class Config:
        env_file = ".env"


settings = Settings()
