"""KE-FPC algorithm wrapper tool (causal discovery via causal-learn PC)."""

import sys
import os
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import networkx as nx

from app.config.algorithm_names import ALGORITHM_KE_FPC
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Try to import causal-learn
try:
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.cit import fisherz
    CAUSAL_LEARN_AVAILABLE = True
except ImportError:
    CAUSAL_LEARN_AVAILABLE = False
    logger.warning("causal-learn not available, will use correlation-based fallback")


def run_pc_analysis(
    data: pd.DataFrame,
    alpha: float = 0.05,
    independent_test_method: str = "fisherz",
    max_path_length: int = -1,
    enable_time_lag: bool = True,
    enable_igci: bool = False,
    verbose: bool = False,
    abnormal_kpi: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run KE-FPC algorithm for causal discovery and root cause analysis.

    Args:
        data: DataFrame with time series metrics (time column is 10-digit timestamp)
        alpha: Significance level for independence tests (default: 0.05)
        independent_test_method: CI test method ("fisherz", "chisq", etc.)
        max_path_length: Maximum path length for causal discovery (-1 for unlimited)
        enable_time_lag: Enable time-lag based orientation
        enable_igci: Enable IGCI-based orientation
        verbose: Verbose mode
        abnormal_kpi: Name of the abnormal KPI metric (used to find root causes that affect it)

    Returns:
        Dictionary containing:
        - root_causes: List of root cause metrics (ranked by importance)
        - causal_graph: Dict representation of the causal graph
        - edges: List of directed edges (cause -> effect)
        - status: Execution status
        - message: Status message
        - abnormal_kpi: The abnormal KPI used for analysis (if provided)
    """
    try:
        # Ensure alpha has a valid value
        if alpha is None:
            alpha = 0.05
            logger.info("PC alpha was None, using default 0.05")

        logger.info(f"Running PC analysis with {len(data.columns)} metrics, alpha={alpha}, abnormal_kpi={abnormal_kpi}")
        logger.info(f"Original data columns (first 10): {data.columns.tolist()[:10]}")

        # Prepare data - ensure numeric columns only, exclude 'time' column
        if 'time' in data.columns:
            data = data.drop(columns=['time'])
        numeric_data = data.select_dtypes(include=[np.number])
        if len(numeric_data.columns) < len(data.columns):
            logger.warning(f"Filtered {len(data.columns) - len(numeric_data.columns)} non-numeric columns")
            data = numeric_data

        # Remove constant columns
        before_cols = data.shape[1]
        data = data.loc[:, (data != data.iloc[0]).any()]
        after_cols = data.shape[1]
        if before_cols != after_cols:
            logger.info(f"Removed {before_cols - after_cols} constant columns")

        # Remove highly correlated columns to avoid singular matrix
        # Keep first of any pair with correlation > 0.99
        if data.shape[1] > 1:
            corr_matrix = data.corr().abs()
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.99)]
            if to_drop:
                logger.info(f"Removing {len(to_drop)} highly correlated columns (>0.99): {to_drop[:5]}...")
                data = data.drop(columns=to_drop)

        logger.info(f"Processed data columns (first 10): {data.columns.tolist()[:10]}")

        if data.shape[1] < 2:
            return {
                "root_causes": [],
                "causal_graph": {},
                "edges": [],
                "status": "error",
                "message": "Insufficient data columns after filtering (need at least 2)"
            }

        logger.info(f"PC analysis using {data.shape[1]} numeric columns")

        # Use processed data directly
        processed_data = data

        # Log column names for debugging
        logger.info(f"Processed data columns type: {type(processed_data.columns)}")
        logger.info(f"Processed data columns (all): {processed_data.columns.tolist()}")
        logger.info(f"First 10 column names: {processed_data.columns.tolist()[:10]}")

        # Try to use causal-learn PC algorithm
        if CAUSAL_LEARN_AVAILABLE:
            try:
                np_data = processed_data.to_numpy()
                node_names = processed_data.columns.tolist()

                # Run PC algorithm with node_names parameter
                cg = pc(
                    np_data,
                    alpha=alpha,
                    indep_test=fisherz if independent_test_method == "fisherz" else None,
                    node_names=node_names  # Pass column names to PC algorithm
                )

                # Get graph as NetworkX DiGraph
                cg.to_nx_graph()
                graph = cg.nx_graph

                # Relabel graph nodes from integer indices to metric names
                # cg.labels is built by causallearn: {0: "cpu_usage", 1: "mem_usage", ...}
                if hasattr(cg, 'labels') and cg.labels:
                    graph = nx.relabel_nodes(graph, cg.labels)
                else:
                    # Fallback: build mapping from our own node_names list
                    mapping = {i: name for i, name in enumerate(node_names)}
                    graph = nx.relabel_nodes(graph, mapping)
                logger.info(f"PC graph nodes relabeled: {list(graph.nodes())[:10]}...")

                # Extract edges and root causes
                edges = list(graph.edges())
                nodes = list(graph.nodes())

                # After relabeling, all nodes are metric name strings
                logger.info(f"PC graph nodes (first 10): {nodes[:10]}")

                # Find root causes based on abnormal_kpi
                root_cause_names = []

                if abnormal_kpi:
                    # Find ancestors of abnormal_kpi (potential root causes)
                    try:
                        if abnormal_kpi in nodes:
                            ancestors = list(nx.ancestors(graph, abnormal_kpi))
                            for ancestor in ancestors:
                                if graph.in_degree(ancestor) == 0:
                                    root_cause_names.append(ancestor)
                            for u, v in edges:
                                if v == abnormal_kpi and u not in root_cause_names:
                                    root_cause_names.append(u)
                            logger.info(f"PC analysis: Found {len(root_cause_names)} root causes affecting abnormal_kpi: {abnormal_kpi}")
                        else:
                            logger.warning(f"abnormal_kpi '{abnormal_kpi}' not found in graph nodes")
                            for node in nodes:
                                if graph.in_degree(node) == 0:
                                    root_cause_names.append(node)
                    except Exception as e:
                        logger.warning(f"Could not find ancestors for abnormal_kpi {abnormal_kpi}: {e}")
                        for node in nodes:
                            if graph.in_degree(node) == 0:
                                root_cause_names.append(node)
                else:
                    logger.info(f"PC: No abnormal_kpi specified, finding root nodes in graph")
                    for node in nodes:
                        if graph.in_degree(node) == 0:
                            root_cause_names.append(node)
                    logger.info(f"PC: Found {len(root_cause_names)} root causes: {root_cause_names}")

                # Build causal graph dict (nodes are already metric name strings)
                causal_edges = [[str(u), str(v)] for u, v in edges]

                causal_graph = {
                    "nodes": node_names,
                    "edges": causal_edges,
                    "node_names": node_names
                }

                logger.info(f"PC analysis completed. Found {len(root_cause_names)} root causes, {len(causal_edges)} edges")
                logger.info(f"PC root_causes: {root_cause_names}")
                logger.info(f"PC causal_graph nodes (first 10): {causal_graph['nodes'][:10]}")

                result = {
                    "root_causes": root_cause_names,
                    "causal_graph": causal_graph,
                    "edges": causal_edges,
                    "status": "success",
                    "message": f"{ALGORITHM_KE_FPC} analysis completed. Found {len(root_cause_names)} root causes and {len(causal_edges)} causal relationships."
                }

                if abnormal_kpi:
                    result["abnormal_kpi"] = abnormal_kpi

                return result

            except Exception as e:
                logger.warning(f"PC algorithm failed: {e}, falling back to correlation-based method")

        # Fallback: correlation-based implementation
        logger.info("Using correlation-based fallback for causal discovery")

        # Compute correlation matrix
        corr_matrix = processed_data.corr()

        # Find potential root causes based on correlation patterns
        # A variable is more likely to be a root cause if:
        # 1. It has high correlation with other variables
        # 2. It appears early in the correlation hierarchy

        # Calculate "root cause score" for each variable
        root_cause_scores = {}
        for col in processed_data.columns:
            # Score: sum of absolute correlations (excluding self)
            correlations = corr_matrix[col].abs()
            correlations[col] = 0  # Exclude self-correlation
            root_cause_scores[col] = correlations.sum()

        # Sort by score (higher score = more likely to be root cause)
        sorted_causes = sorted(root_cause_scores.items(), key=lambda x: x[1], reverse=True)

        # Select top 5 as root causes
        root_causes = [col for col, score in sorted_causes[:5]]

        # If abnormal_kpi is specified, prioritize metrics highly correlated with it
        if abnormal_kpi and abnormal_kpi in processed_data.columns:
            # Find metrics highly correlated with abnormal_kpi
            kpi_correlations = {}
            for col in processed_data.columns:
                if col != abnormal_kpi:
                    corr = corr_matrix.loc[abnormal_kpi, col]
                    if not pd.isna(corr):
                        kpi_correlations[col] = abs(corr)

            # Sort by correlation with abnormal_kpi
            sorted_by_kpi = sorted(kpi_correlations.items(), key=lambda x: x[1], reverse=True)

            # Use top correlated metrics as root causes
            if sorted_by_kpi:
                root_causes = [col for col, corr in sorted_by_kpi[:5]]
                logger.info(f"PC fallback: Using top correlated metrics with abnormal_kpi: {abnormal_kpi}")

        # Create edges based on high correlations
        edges = []
        correlation_threshold = 0.3

        for i, col1 in enumerate(processed_data.columns):
            for col2 in processed_data.columns[i+1:]:
                corr = corr_matrix.loc[col1, col2]
                if abs(corr) > correlation_threshold:
                    # Direction: higher score -> lower score (potential causal direction)
                    if root_cause_scores[col1] > root_cause_scores[col2]:
                        edges.append([col1, col2])
                    else:
                        edges.append([col2, col1])

        causal_graph = {
            "nodes": processed_data.columns.tolist(),
            "edges": edges,
            "method": "correlation_fallback"
        }

        logger.info(f"Correlation-based analysis completed. Found {len(root_causes)} root causes, {len(edges)} edges")
        logger.info(f"Correlation fallback root_causes: {root_causes}")
        logger.info(f"Correlation fallback nodes (first 10): {causal_graph['nodes'][:10]}")

        result = {
            "root_causes": root_causes,
            "causal_graph": causal_graph,
            "edges": edges,
            "status": "success",
            "message": f"{ALGORITHM_KE_FPC} analysis (correlation fallback) completed. Found {len(root_causes)} potential root causes and {len(edges)} causal relationships."
        }

        if abnormal_kpi:
            result["abnormal_kpi"] = abnormal_kpi

        return result

    except Exception as e:
        logger.error(f"PC analysis failed: {e}", exc_info=True)
        return {
            "root_causes": [],
            "causal_graph": {},
            "edges": [],
            "status": "error",
            "message": f"{ALGORITHM_KE_FPC} analysis failed: {str(e)}"
        }


def format_pc_results(result: Dict[str, Any]) -> str:
    """
    Format KE-FPC results for display.

    Args:
        result: Result dictionary from run_pc_analysis

    Returns:
        Formatted string with results
    """
    if result.get("status") == "error":
        return f"❌ {ALGORITHM_KE_FPC} Analysis Error: {result.get('message')}"

    root_causes = result.get("root_causes", [])
    causal_graph = result.get("causal_graph", {})
    edges = result.get("edges", [])

    output = [f"=== {ALGORITHM_KE_FPC} 因果发现分析结果 ===\n"]
    output.append(f"📊 分析状态: {result.get('status', 'unknown')}")
    output.append(f"🔍 发现根因数量: {len(root_causes)}")
    output.append(f"🔗 因果关系数量: {len(edges)}\n")

    if root_causes:
        output.append("**根因指标列表（潜在根因）**:")
        for i, cause in enumerate(root_causes[:10], 1):
            output.append(f"  {i}. {cause}")

        if len(root_causes) > 10:
            output.append(f"  ... 及其他 {len(root_causes) - 10} 个指标")

        output.append("")

    if edges:
        output.append("**故障传播图（因果链）**:")
        for i, edge in enumerate(edges[:15], 1):
            output.append(f"  {i}. {edge[0]} → {edge[1]}")

        if len(edges) > 15:
            output.append(f"  ... 及其他 {len(edges) - 15} 条边")

    return "\n".join(output)


def visualize_causal_graph(result: Dict[str, Any]) -> str:
    """
    Generate a text-based visualization of the causal graph.

    Args:
        result: Result dictionary from run_pc_analysis

    Returns:
        ASCII representation of the causal graph
    """
    if result.get("status") == "error":
        return "Error: Cannot visualize graph"

    edges = result.get("edges", [])
    if not edges:
        return "No causal relationships to visualize"

    # Build adjacency structure
    graph = {}
    for src, dst in edges:
        if src not in graph:
            graph[src] = []
        graph[src].append(dst)

    # Generate text representation
    output = ["\n=== 故障传播图结构 ===\n"]

    # Sort nodes alphabetically for consistent output
    for src in sorted(graph.keys()):
        targets = graph[src]
        output.append(f"{src}")
        for dst in targets:
            output.append(f"  └──► {dst}")
        output.append("")

    return "\n".join(output)
