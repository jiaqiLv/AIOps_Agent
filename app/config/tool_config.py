"""Tool configuration settings"""

from typing import Dict, Any


# CSV reader configuration
CSV_READER_CONFIG: Dict[str, Any] = {
    "preview_rows": 5,
    "max_preview_rows": 10,
    "encoding": "utf-8",
}

# RCD tool configuration (reserved for future implementation)
RCD_TOOL_CONFIG: Dict[str, Any] = {
    "enabled": False,
    "endpoint": None,
    "timeout": 30,
}

# PC tool configuration (reserved for future implementation)
PC_TOOL_CONFIG: Dict[str, Any] = {
    "enabled": False,
    "endpoint": None,
    "timeout": 30,
}
