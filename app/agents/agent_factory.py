"""Agent Factory for building agents from configuration"""

import os
import importlib
from typing import Any, Dict, Optional, List
from langgraph.graph import StateGraph, END

from app.utils.logger import get_logger
from app.utils.prompt_loader import load_prompt
from app.tools.tool_registry import ToolRegistry


class AgentFactory:
    """
    Factory for building agents from configuration files.

    This factory handles:
    - Loading agent configurations from YAML
    - Loading system prompts from markdown files
    - Building agent graphs with LangGraph
    - Binding tools to agents
    """

    def __init__(
        self,
        agents_config_path: str = "app/config/agents.yaml",
        tool_registry: Optional[ToolRegistry] = None
    ):
        """
        Initialize the agent factory.

        Args:
            agents_config_path: Path to the agents configuration YAML file
            tool_registry: Tool registry instance (creates default if None)
        """
        self.agents_config_path = agents_config_path
        self.tool_registry = tool_registry or ToolRegistry()
        self.agents_config: Dict[str, Any] = {}

        # Load configurations
        self._load_config()

    def _load_config(self) -> None:
        """Load agent configuration from YAML file"""
        try:
            import yaml

            with open(self.agents_config_path, 'r', encoding='utf-8') as f:
                self.agents_config = yaml.safe_load(f)

            logger.info(f"Loaded agent config from {self.agents_config_path}")

        except FileNotFoundError:
            logger.error(f"Agents config file not found: {self.agents_config_path}")
        except ImportError:
            logger.warning("PyYAML not installed, using fallback agent config")
            self._load_fallback_config()
        except Exception as e:
            logger.error(f"Error loading agent config: {e}")

    def _load_fallback_config(self) -> None:
        """Load fallback agent configuration"""
        self.agents_config = {
            "agents": {
                "supervisor": {
                    "name": "supervisor",
                    "type": "supervisor_agent",
                    "system_prompt": "app/prompts/supervisor_system.md",
                    "tools": ["diagnose_subagent", "ask_user"],
                    "max_iterations": 8
                },
                "diagnose": {
                    "name": "diagnose",
                    "type": "diagnose_agent",
                    "system_prompt": "app/prompts/diagnose_system.md",
                    "refine_prompt": "app/prompts/diagnose_refine.md",
                    "tools": ["csv_reader_tool", "rcd_tool", "pc_tool"],
                    "max_iterations": 10
                }
            }
        }

    def build_agent(self, agent_name: str) -> Optional[StateGraph]:
        """
        Build an agent graph from configuration.

        Args:
            agent_name: Name of the agent to build

        Returns:
            Compiled StateGraph or None if agent not found
        """
        if "agents" not in self.agents_config or agent_name not in self.agents_config["agents"]:
            logger.error(f"Agent not found in config: {agent_name}")
            return None

        agent_config = self.agents_config["agents"][agent_name]

        try:
            # Load state schema
            state_schema_class = self._load_state_schema(
                agent_config.get("state_schema", f"app.models.{agent_name}_state.{agent_name.capitalize()}State")
            )

            # Build graph
            if agent_config.get("type") == "supervisor_agent":
                graph = self._build_supervisor_agent(agent_config, state_schema_class)
            elif agent_config.get("type") == "diagnose_agent":
                graph = self._build_diagnose_agent(agent_config, state_schema_class)
            else:
                logger.error(f"Unknown agent type: {agent_config.get('type')}")
                return None

            logger.info(f"Built agent: {agent_name}")
            return graph

        except Exception as e:
            logger.error(f"Error building agent {agent_name}: {e}")
            return None

    def _load_state_schema(self, schema_path: str) -> type:
        """Load state schema class from module path"""
        parts = schema_path.split(".")
        module_path = ".".join(parts[:-1])
        class_name = parts[-1]

        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    def _load_system_prompt(self, prompt_path: str) -> str:
        """Load system prompt from markdown file"""
        try:
            return load_prompt(prompt_path)
        except FileNotFoundError:
            logger.warning(f"Prompt file not found: {prompt_path}")
            return ""

    def _build_supervisor_agent(self, config: Dict[str, Any], state_schema: type) -> StateGraph:
        """Build supervisor agent graph"""
        from app.agents.supervisor_agent import supervisor_llm_node

        builder = StateGraph(state_schema)
        builder.add_node("supervisor_llm", supervisor_llm_node)
        builder.set_entry_point("supervisor_llm")
        builder.add_edge("supervisor_llm", END)

        return builder.compile()

    def _build_diagnose_agent(self, config: Dict[str, Any], state_schema: type) -> StateGraph:
        """Build diagnose agent graph (sequential workflow)"""
        from app.agents.diagnose_agent import (
            parse_params_node,
            load_csv_node,
            run_rcd_node,
            run_pc_node,
            refine_node,
        )

        builder = StateGraph(state_schema)

        builder.add_node("parse_params", parse_params_node)
        builder.add_node("load_csv", load_csv_node)
        builder.add_node("run_rcd", run_rcd_node)
        builder.add_node("run_pc", run_pc_node)
        builder.add_node("refine", refine_node)

        builder.set_entry_point("parse_params")
        builder.add_edge("parse_params", "load_csv")
        builder.add_edge("load_csv", "run_rcd")
        builder.add_edge("run_rcd", "run_pc")
        builder.add_edge("run_pc", "refine")
        builder.add_edge("refine", END)

        return builder.compile()

    def get_agent_config(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific agent"""
        if "agents" in self.agents_config and agent_name in self.agents_config["agents"]:
            return self.agents_config["agents"][agent_name]
        return None

    def list_agents(self) -> List[str]:
        """List all available agent names"""
        if "agents" in self.agents_config:
            return list(self.agents_config["agents"].keys())
        return []


# Global agent factory instance
_agent_factory: Optional[AgentFactory] = None


def get_agent_factory(
    agents_config_path: str = "app/config/agents.yaml",
    tool_registry: Optional[ToolRegistry] = None
) -> AgentFactory:
    """
    Get the global agent factory instance.

    Args:
        agents_config_path: Path to agents config file
        tool_registry: Tool registry instance

    Returns:
        AgentFactory instance
    """
    global _agent_factory

    if _agent_factory is None:
        _agent_factory = AgentFactory(agents_config_path, tool_registry)

    return _agent_factory
