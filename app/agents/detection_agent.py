"""Detection Agent - config-driven ReAct loop for anomaly detection.

All wiring (tools, prompt, LLM, graph structure) is driven by
``app/config/agents.yaml`` via the agent registry. This module only
exposes the public builder and LazyGraph proxy.
"""

from app.agents.agent_registry import load_agent_config, build_react_agent
from app.utils.lazy_graph import LazyGraph
from app.utils.logger import get_logger

logger = get_logger(__name__)


def build_detection_agent():
    """Build the detection agent via the config-driven registry."""
    logger.info("Building detection agent via registry")
    config = load_agent_config("detection")
    return build_react_agent(config)


graph = LazyGraph(build_detection_agent)
detection_agent = graph
