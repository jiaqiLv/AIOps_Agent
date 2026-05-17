"""Main workflow builder - single entry point with nested subgraphs

The main graph contains:
- supervisor_agent: Nested subgraph (expandable in Studio)
- diagnose_agent: Nested subgraph (expandable in Studio)
"""

from app.agents.main_graph import main_graph
from app.utils.logger import get_logger

logger = get_logger(__name__)


# Export the main graph
def get_main_graph():
    """
    Get the main workflow graph.

    Returns:
        Compiled StateGraph for the main workflow
    """
    return main_graph
