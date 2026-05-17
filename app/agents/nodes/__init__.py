"""ReAct Loop Nodes

This package provides generic node implementations for configuration-driven
ReAct loops.
"""

from app.agents.nodes.react_nodes import (
    create_model_node,
    extract_results_node,
    create_final_response_node,
    route_after_model,
    route_after_extract,
    compress_messages,
)

from app.agents.nodes.prompt_based_react import (
    create_prompt_based_model_node,
    create_prompt_based_route_function,
    parse_tool_call,
    execute_tool,
)

__all__ = [
    "create_model_node",
    "extract_results_node",
    "create_final_response_node",
    "route_after_model",
    "route_after_extract",
    "compress_messages",
    "create_prompt_based_model_node",
    "create_prompt_based_route_function",
    "parse_tool_call",
    "execute_tool",
]
