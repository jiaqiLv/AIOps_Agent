"""Agents module - Multi-agent system with nested subgraphs

Exports:
- main_graph: The main entry point graph with nested subgraphs
- supervisor_agent: Supervisor subgraph (can be used independently)
- AgentFactory: Factory for building agents from configuration
- get_agent_factory: Get the global agent factory instance

Note: The diagnose agent is now built dynamically using AgentFactory.
Use get_agent_factory().build_agent("diagnose") to get the diagnose subgraph.
"""

from app.agents.main_graph import main_graph
from app.agents.supervisor_agent import supervisor_agent
from app.agents.agent_factory import AgentFactory, get_agent_factory

__all__ = [
    "main_graph",
    "supervisor_agent",
    "AgentFactory",
    "get_agent_factory"
]
