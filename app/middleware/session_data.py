"""M7: Session Data Isolation Middleware

Creates isolated output directories per analysis session so that
reports, graph visualizations and trace files don't collide.

Integration point:
  - app.main.run_conversation / run_single_request  →  at session start
"""

import os
import uuid
from datetime import datetime
from typing import Optional

from app.utils.logger import get_logger

logger = get_logger(__name__)


class SessionDataManager:
    """Manages per-session output directories."""

    def __init__(self, base_dir: str = "outputs"):
        self.base_dir = base_dir
        self.session_id: Optional[str] = None
        self.session_dir: Optional[str] = None

    def create_session(self, session_id: Optional[str] = None) -> str:
        """Create a new isolated session directory and return its path."""
        if session_id is None:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{uuid.uuid4().hex[:6]}"
        self.session_id = session_id
        self.session_dir = os.path.join(self.base_dir, "sessions", session_id)
        os.makedirs(self.session_dir, exist_ok=True)
        logger.info(f"SESSION: Created session directory: {self.session_dir}")
        return self.session_dir

    def get_output_path(self, filename: str) -> str:
        """Return the full path for *filename* within the current session."""
        if self.session_dir is None:
            self.create_session()
        return os.path.join(self.session_dir, filename)  # type: ignore

    @property
    def output_dir(self) -> str:
        if self.session_dir is None:
            self.create_session()
        return self.session_dir  # type: ignore

    def reset(self):
        """Clear session state (does NOT delete files)."""
        self.session_id = None
        self.session_dir = None


# ── Singleton ───────────────────────────────────────────────────────

_manager: Optional[SessionDataManager] = None


def get_session_manager() -> SessionDataManager:
    global _manager
    if _manager is None:
        _manager = SessionDataManager()
    return _manager
