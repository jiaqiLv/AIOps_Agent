"""Tool Registry for dynamic tool loading and validation"""

import os
import importlib
from typing import Any, Dict, List, Optional, Callable
from app.utils.logger import get_logger
from app.utils.json_utils import from_json_file
from app.utils.path_resolver import resolve_config_path

logger = get_logger(__name__)


class ToolRegistry:
    """
    Registry for managing tools dynamically loaded from configuration.

    This registry handles:
    - Loading tools from YAML configuration
    - Dynamic module and function imports
    - Parameter validation
    - Agent-specific tool allowlists
    """

    def __init__(self, tools_config_path: str = "app/config/tools.yaml"):
        """
        Initialize the tool registry.

        Args:
            tools_config_path: Path to the tools configuration YAML file
        """
        # Resolve config path relative to project root
        self.tools_config_path = resolve_config_path(tools_config_path)
        self.tools: Dict[str, Dict[str, Any]] = {}
        self._loaded = False

    def load_tools(self) -> None:
        """Load tools from the configuration file"""
        if self._loaded:
            return

        try:
            # Try to load from YAML
            import yaml

            with open(self.tools_config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            if 'tools' not in config:
                logger.error("Invalid tools config: missing 'tools' key")
                return

            for tool_name, tool_config in config['tools'].items():
                self._register_tool(tool_name, tool_config)

            self._loaded = True
            logger.info(f"Loaded {len(self.tools)} tools from {self.tools_config_path}")

        except FileNotFoundError:
            logger.error(f"Tools config file not found: {self.tools_config_path}")
        except ImportError:
            logger.warning("PyYAML not installed, using fallback tool loading")
            self._load_fallback_tools()
        except Exception as e:
            logger.error(f"Error loading tools: {e}")

    def _load_fallback_tools(self) -> None:
        """Load fallback tools when YAML is not available"""
        fallback_tools = {
            "csv_reader_tool": {
                "name": "csv_reader_tool",
                "description": "Read CSV data files",
                "module": "app.tools.csv_reader_tool",
                "function": "read_csv_metrics",
                "required_fields": ["data_path"]
            },
            "three_sigma_tool": {
                "name": "three_sigma_tool",
                "description": "Run 3-sigma anomaly detection on time-series metrics",
                "module": "app.tools.three_sigma",
                "function": "run_three_sigma",
                "required_fields": ["inject_time"]
            },
            "rcd_tool": {
                "name": "rcd_tool",
                "description": "Run IAF-RCL algorithm for root cause analysis",
                "module": "app.tools.rcd_wrapper",
                "function": "run_rcd_analysis",
                "required_fields": ["data", "inject_time"]
            },
            "pc_tool": {
                "name": "pc_tool",
                "description": "Run KE-FPC algorithm for causal discovery",
                "module": "app.tools.pc_wrapper",
                "function": "run_pc_analysis",
                "required_fields": ["data"]
            }
        }

        for tool_name, tool_config in fallback_tools.items():
            self._register_tool(tool_name, tool_config)

        self._loaded = True
        logger.info(f"Loaded {len(self.tools)} fallback tools")

    def _register_tool(self, name: str, config: Dict[str, Any]) -> None:
        """
        Register a tool configuration.

        Args:
            name: Tool name
            config: Tool configuration dictionary
        """
        self.tools[name] = {
            "name": name,
            "description": config.get("description", ""),
            "module": config.get("module", ""),
            "function": config.get("function", ""),
            "required_fields": config.get("required_fields", []),
            "config": config
        }

    def get_tool_function(self, tool_name: str) -> Optional[Callable]:
        """
        Get the actual function for a tool.

        Args:
            tool_name: Name of the tool

        Returns:
            The tool function or None if not found
        """
        if tool_name not in self.tools:
            logger.error(f"Tool not found: {tool_name}")
            return None

        tool_config = self.tools[tool_name]
        module_name = tool_config.get("module")
        function_name = tool_config.get("function")

        if not module_name or not function_name:
            logger.error(f"Invalid tool config for {tool_name}: missing module or function")
            return None

        try:
            module = importlib.import_module(module_name)
            func = getattr(module, function_name)
            return func
        except ImportError as e:
            logger.error(f"Failed to import module {module_name}: {e}")
            return None
        except AttributeError as e:
            logger.error(f"Function {function_name} not found in module {module_name}: {e}")
            return None

    def get_tools_for_agent(self, tool_names: List[str]) -> List[Dict[str, Any]]:
        """
        Get tool configurations for an agent's allowlist.

        Args:
            tool_names: List of tool names that the agent can use

        Returns:
            List of tool configurations
        """
        available_tools = []

        for tool_name in tool_names:
            if tool_name in self.tools:
                available_tools.append(self.tools[tool_name])
            else:
                logger.warning(f"Tool {tool_name} not available for agent")

        return available_tools

    def validate_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate a tool call.

        Args:
            tool_name: Name of the tool being called
            arguments: Arguments provided for the tool call

        Returns:
            Validation result with keys:
            - valid: bool
            - missing_fields: list of str
            - error: str or None
        """
        if tool_name not in self.tools:
            return {
                "valid": False,
                "missing_fields": [],
                "error": f"Tool {tool_name} is not registered"
            }

        tool_config = self.tools[tool_name]
        required_fields = tool_config.get("required_fields", [])

        # Check for one_of requirement
        if "one_of" in required_fields and isinstance(required_fields[0], list):
            # Check if at least one of the option sets is satisfied
            one_valid = False
            for option_set in required_fields:
                if all(field in arguments and arguments[field] is not None for field in option_set):
                    one_valid = True
                    break

            if not one_valid:
                options_str = " or ".join([str(opt) for opt in required_fields])
                return {
                    "valid": False,
                    "missing_fields": required_fields,
                    "error": f"Tool {tool_name} requires one of: {options_str}"
                }

        else:
            # Standard required fields check
            missing = []
            for field in required_fields:
                if field not in arguments or arguments[field] is None:
                    missing.append(field)

            if missing:
                return {
                    "valid": False,
                    "missing_fields": missing,
                    "error": f"Tool {tool_name} missing required fields: {missing}"
                }

        return {
            "valid": True,
            "missing_fields": [],
            "error": None
        }

    def list_tools(self) -> List[str]:
        """Get list of all registered tool names"""
        return list(self.tools.keys())

    def get_tool_description(self, tool_name: str) -> str:
        """Get the description of a tool"""
        if tool_name in self.tools:
            return self.tools[tool_name].get("description", "")
        return ""

    def get_langchain_tools(self, tool_names: List[str]) -> List:
        """
        Get LangChain StructuredTool instances for a list of tool names.

        This method creates LangChain-compatible tools that can be used with
        LangChain's ToolNode and bind_tools().

        Args:
            tool_names: List of tool names to convert

        Returns:
            List of LangChain StructuredTool instances
        """
        from app.tools.langchain_tool_adapters import create_langchain_tools
        return create_langchain_tools(tool_names, self)


# Global tool registry instance
_tool_registry: Optional[ToolRegistry] = None


def get_tool_registry(tools_config_path: str = "app/config/tools.yaml") -> ToolRegistry:
    """
    Get the global tool registry instance.

    Args:
        tools_config_path: Path to tools config file

    Returns:
        ToolRegistry instance
    """
    global _tool_registry

    if _tool_registry is None:
        _tool_registry = ToolRegistry(tools_config_path)
        _tool_registry.load_tools()

    return _tool_registry


# Legacy compatibility
def get_tool(name: str) -> Callable:
    """Legacy function for backward compatibility"""
    registry = get_tool_registry()
    func = registry.get_tool_function(name)
    if func is None:
        raise KeyError(f"Tool not found: {name}")
    return func


# Create global instance for backward compatibility
tool_registry = get_tool_registry()
