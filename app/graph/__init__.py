"""LangGraph workflow modules - Single main graph with nested subgraphs"""

# Export main workflow graph
from app.agents.main_graph import main_graph

# Default export
graph = main_graph

__all__ = ["graph", "main_graph"]
