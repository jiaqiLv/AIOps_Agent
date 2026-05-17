"""Tests for the models module"""

import pytest
from app.models import BaseLLM, LLMConfig, DeepSeekModel, create_model, ModelType
from app.models.deepseek import DeepSeekModel as DirectDeepSeekModel
from app.config.model_config import get_llm, get_deepseek_llm


def test_llm_config():
    """Test LLMConfig creation"""
    config = LLMConfig(
        model_name="deepseek-chat",
        temperature=0.7,
        max_tokens=2000,
        api_key="test_key"
    )

    assert config.model_name == "deepseek-chat"
    assert config.temperature == 0.7
    assert config.max_tokens == 2000
    assert config.api_key == "test_key"


def test_deepseek_model_creation():
    """Test DeepSeek model creation"""
    config = LLMConfig(
        model_name="deepseek-chat",
        api_key="test_key"
    )

    model = DeepSeekModel(config)

    assert model is not None
    assert model.get_model_name() == "deepseek-chat"
    assert model.get_config() == config


def test_model_factory():
    """Test model factory"""
    # Test DeepSeek model creation
    model = create_model(
        model_type=ModelType.DEEPSEEK,
        api_key="test_key"
    )

    assert model is not None
    assert isinstance(model, BaseLLM)
    assert model.get_model_name() == "deepseek-chat"


def test_get_llm():
    """Test get_llm function"""
    # Test without API key (should still create model)
    model = get_llm(provider="deepseek", api_key="test_key")

    assert model is not None
    assert isinstance(model, BaseLLM)


def test_get_deepseek_llm():
    """Test get_deepseek_llm convenience function"""
    model = get_deepseek_llm(api_key="test_key")

    assert model is not None
    assert isinstance(model, BaseLLM)
    assert model.get_model_name() == "deepseek-chat"


def test_model_type_enum():
    """Test ModelType enum"""
    assert ModelType.DEEPSEEK == "deepseek"
    assert ModelType.OPENAI == "openai"
    assert ModelType.ANTHROPIC == "anthropic"


def test_unsupported_model_type():
    """Test that unsupported model types raise error"""
    with pytest.raises(ValueError):
        create_model(
            model_type="unsupported",
            api_key="test_key"
        )


def test_deepseek_base_url():
    """Test DeepSeek base URL configuration"""
    config = LLMConfig(
        model_name="deepseek-chat",
        api_key="test_key",
        base_url="https://custom.api.url"
    )

    model = DeepSeekModel(config)

    assert model is not None
    assert model.config.base_url == "https://custom.api.url"
