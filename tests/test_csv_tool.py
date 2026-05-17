"""Tests for the CSV reader tool"""

import pytest
from app.tools.csv_reader_tool import read_csv_headers, read_csv_data


def test_read_csv_headers_success():
    """Test reading CSV headers successfully"""
    result = read_csv_headers("./data/sample_metrics.csv")

    assert result.get("success") is True
    assert "headers" in result
    assert len(result.get("headers", [])) > 0
    assert "timestamp" in result.get("headers", [])
    assert "service_name" in result.get("headers", [])


def test_read_csv_headers_file_not_found():
    """Test reading non-existent file"""
    result = read_csv_headers("./data/nonexistent.csv")

    assert result.get("success") is False
    assert "error" in result


def test_read_csv_headers_shape():
    """Test that shape information is correct"""
    result = read_csv_headers("./data/sample_metrics.csv")

    assert result.get("success") is True
    shape = result.get("shape", {})
    assert "rows" in shape
    assert "columns" in shape
    assert shape["rows"] > 0
    assert shape["columns"] > 0


def test_read_csv_data_with_filters():
    """Test reading CSV data with filters"""
    result = read_csv_data(
        "./data/sample_metrics.csv",
        filters={"service_name": "checkoutservice"},
        limit=10
    )

    assert result.get("success") is True
    assert "data" in result
    assert len(result.get("data", [])) <= 10
