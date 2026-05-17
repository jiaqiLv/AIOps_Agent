"""Prompt loader utility for loading markdown prompts from files"""

import os
from typing import Optional
from app.utils.logger import get_logger

logger = get_logger(__name__)


def load_prompt(prompt_path: str) -> str:
    """
    Load a prompt from a markdown file.

    Args:
        prompt_path: Path to the prompt markdown file (relative to project root or absolute)

    Returns:
        The prompt content as a string

    Raises:
        FileNotFoundError: If the prompt file doesn't exist
        IOError: If there's an error reading the file
    """
    # Handle relative paths
    if not os.path.isabs(prompt_path):
        # Try to resolve relative to project root
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        prompt_path = os.path.join(project_root, prompt_path)

    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            content = f.read()

        logger.debug(f"Loaded prompt from {prompt_path}")
        return content

    except Exception as e:
        logger.error(f"Error loading prompt from {prompt_path}: {e}")
        raise IOError(f"Failed to load prompt from {prompt_path}: {e}")


def get_system_prompt(agent_name: str) -> str:
    """
    Get the system prompt for a specific agent.

    Args:
        agent_name: Name of the agent (e.g., 'supervisor', 'diagnose')

    Returns:
        The system prompt content

    Raises:
        FileNotFoundError: If the prompt file doesn't exist
    """
    prompt_file = f"app/prompts/{agent_name}_system.md"
    return load_prompt(prompt_file)


def get_refine_prompt(agent_name: str) -> str:
    """
    Get the refine prompt for a specific agent.

    Args:
        agent_name: Name of the agent (e.g., 'diagnose')

    Returns:
        The refine prompt content

    Raises:
        FileNotFoundError: If the prompt file doesn't exist
    """
    prompt_file = f"app/prompts/{agent_name}_refine.md"
    return load_prompt(prompt_file)
