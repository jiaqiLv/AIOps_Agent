"""Agent Factory for building agents from configuration"""

import os
import importlib
import json
from typing import Any, Dict, Optional, List
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage

from app.utils.logger import get_logger
from app.utils.prompt_loader import load_prompt
from app.utils.path_resolver import resolve_config_path
from app.tools.tool_registry import ToolRegistry
from app.tools.langchain_tool_adapters import create_diagnose_tools
from app.utils.tool_prompt_generator import inject_tools_into_prompt, generate_diagnose_agent_tools_prompt


logger = get_logger(__name__)


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
        # Resolve config path relative to project root
        self.agents_config_path = resolve_config_path(agents_config_path)
        self.tool_registry = tool_registry or ToolRegistry()

        # Ensure tools are loaded
        if not self.tool_registry._loaded:
            self.tool_registry.load_tools()

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
            logger.warning(f"Agents config file not found: {self.agents_config_path}, using fallback")
            self._load_fallback_config()
        except ImportError:
            logger.warning("PyYAML not installed, using fallback agent config")
            self._load_fallback_config()
        except Exception as e:
            logger.error(f"Error loading agent config: {e}, using fallback config")
            self._load_fallback_config()

    def _load_fallback_config(self) -> None:
        """Load fallback agent configuration"""
        self.agents_config = {
            "agents": {
                "supervisor": {
                    "name": "supervisor",
                    "type": "supervisor_agent",
                    "state_schema": "app.models.supervisor_state.SupervisorState",
                    "system_prompt": "app/prompts/supervisor_system.md",
                    "tools": ["diagnose_subagent", "ask_user"],
                    "max_iterations": 8
                },
                "diagnose": {
                    "name": "diagnose",
                    "type": "react_agent",
                    "state_schema": "app.models.react_agent_state.ReactAgentState",
                    "system_prompt": "app/prompts/diagnose_system.md",
                    "refine_prompt": "app/prompts/diagnose_refine.md",
                    "tools": ["csv_reader_tool", "three_sigma_tool", "rcd_tool", "pc_tool", "graph_visualization_tool", "ask_user"],
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
            elif agent_config.get("type") in ("diagnose_agent", "react_agent"):
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
        """Build diagnose agent graph using prompt-based ReAct loop workflow."""
        from app.agents.nodes.prompt_based_react import (
            create_prompt_based_model_node,
            create_prompt_based_route_function,
        )
        from app.agents.nodes.react_nodes import extract_results_node, create_final_response_node
        from app.config.model_config import get_llm

        # Load configuration
        system_prompt_path = config.get("system_prompt", "app/prompts/diagnose_system.md")
        refine_prompt_path = config.get("refine_prompt", "app/prompts/diagnose_refine.md")
        model_config = config.get("model", {})
        max_iterations = config.get("max_iterations", 10)

        # Load base system prompt (without tools - tools will be injected)
        base_system_prompt = self._load_system_prompt(system_prompt_path)

        # Create LLM (without binding tools)
        llm = get_llm(
            provider=model_config.get("provider", "deepseek"),
            model_name=model_config.get("name", "deepseek-chat"),
            temperature=model_config.get("temperature", 0)
        )

        # Get LangChain tools
        tools = create_diagnose_tools(self.tool_registry)

        if not tools:
            logger.error("No tools were created! Check tool_registry configuration.")
            # Fallback: create minimal tools
            from langchain_core.tools import StructuredTool
            from pydantic import Field

            def dummy_csv_reader(data_path: str = Field(..., description="CSV file path")) -> str:
                return json.dumps({"error": "Tool not properly configured"}, ensure_ascii=False)

            tools = [StructuredTool.from_function(
                func=dummy_csv_reader,
                name="csv_reader_tool",
                description="Read CSV data (fallback)",
            )]

        logger.info(f"Created {len(tools)} tools for prompt-based ReAct: {[t.name for t in tools]}")

        # Get the underlying LangChain client for direct invocation
        if hasattr(llm, 'get_client'):
            llm_client = llm.get_client()
            logger.info("Using underlying LangChain client for prompt-based ReAct")
        else:
            llm_client = llm
            logger.info("Using LLM wrapper directly for prompt-based ReAct")

        # Build graph
        builder = StateGraph(state_schema)

        # Create prompt-based model node (tools description in prompt)
        model_node = create_prompt_based_model_node(
            llm=llm_client,
            tools=tools,
            system_prompt=base_system_prompt
        )

        # Add nodes
        builder.add_node("model_node", model_node)
        builder.add_node("extract_results_node", extract_results_node)
        builder.add_node("final_response_node", create_final_response_node(refine_prompt_path, llm))

        # Set entry point
        builder.set_entry_point("model_node")

        # Create routing function
        route_function = create_prompt_based_route_function()

        # Add conditional edges
        builder.add_conditional_edges(
            "model_node",
            route_function,
            {
                "tools": "extract_results_node",  # After tool call, extract results
                "model": "model_node",  # Try again (first iteration no tools)
                "final": "final_response_node"  # Done
            }
        )

        builder.add_conditional_edges(
            "extract_results_node",
            lambda state: "model" if not state.get("final_response") else "final",
            {
                "model": "model_node",
                "final": "final_response_node"
            }
        )

        builder.add_edge("final_response_node", END)

        graph = builder.compile()
        logger.info("Diagnose agent built with prompt-based ReAct loop workflow")
        return graph

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
