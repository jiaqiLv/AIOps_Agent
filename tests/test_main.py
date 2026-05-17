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


def test_graph_import():
    """Test that graph can be imported"""
    from app.graph.builder import graph
    assert graph is not None


def test_config_import():
    """Test that config can be imported"""
    from app.config.settings import settings
    assert settings is not None


def test_tools_import():
    """Test that tools can be imported"""
    from app.tools import read_csv_headers, rcd_analyze, pc_analyze
    assert read_csv_headers is not None
    assert rcd_analyze is not None
    assert pc_analyze is not None
