"""Tool configuration settings"""

from typing import Dict, Any


# CSV reader configuration
CSV_READER_CONFIG: Dict[str, Any] = {
    "preview_rows": 5,
    "max_preview_rows": 10,
    "encoding": "utf-8",
}

# IAF-RCL tool configuration (rcd_tool)
RCD_TOOL_CONFIG: Dict[str, Any] = {
    "enabled": False,
    "endpoint": None,
    "timeout": 30,
}

# KE-FPC tool configuration (pc_tool)
PC_TOOL_CONFIG: Dict[str, Any] = {
    "enabled": False,
    "endpoint": None,
    "timeout": 30,
}
