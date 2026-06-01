"""Tests for the main entry point"""

import pytest
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_main_imports():
    """Test that main module can be imported"""
    import app.main
    assert app.main is not None


def test_main_graph_import():
    """Test that main graph can be imported"""
    from app.agents.main_graph import main_graph
    assert main_graph is not None


def test_config_import():
    """Test that config can be imported"""
    from app.config.settings import settings
    assert settings is not None


def test_agents_import():
    """Test that all agents can be imported"""
    from app.agents import main_graph, supervisor_agent, detection_agent, diagnose_agent
    assert main_graph is not None
    assert supervisor_agent is not None
    assert detection_agent is not None
    assert diagnose_agent is not None


def test_plan_execute_state_import():
    """Test that PlanExecuteState can be imported"""
    from app.models.plan_execute_state import PlanExecuteState, PlanStep
    assert PlanExecuteState is not None
    assert PlanStep is not None
