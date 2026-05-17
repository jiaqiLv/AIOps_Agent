"""
Main entry point for the microservice causal discovery system.

This module orchestrates the complete pipeline:
1. Data loading
2. Preprocessing
3. Knowledge constraint generation
4. Causal discovery (PC algorithm)
5. Cascade orientation (Meek rules, time lag, IGCI)
6. Root cause analysis (RCA)
7. Evaluation (optional)
8. Result saving
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Any
import logging

from src.utils.config_manager import ConfigManager
from src.dataloader.loaders.csv_loader import CSVDataLoader
from src.dataloader.loaders.radice_loader import RADICELoader
from src.dataloader.container import DataContainer
from src.preprocessing.pipeline import PreprocessingPipeline
from src.knowledge.knowledge_manager import KnowledgeManager
from src.algorithms.factory import AlgorithmFactory
from src.algorithms.base_algorithm import AlgorithmResult
from src.orientation.cascade_orientator import CascadeOrientator
from src.rca.rca_engine import RCAEngine, RCAConfig
from src.evaluation.evaluator import EvaluationEngine
from src.output.result_saver import ResultSaver
from src.utils.logger import setup_logger

import networkx as nx
import pandas as pd


def _build_graph_from_dataframe(causal_graph: pd.DataFrame) -> nx.DiGraph:
    """Convert causal graph DataFrame to NetworkX DiGraph (directed edges only)."""
    graph = nx.DiGraph()
    node_names = causal_graph.index.tolist()
    graph.add_nodes_from(node_names)

    for i, src in enumerate(node_names):
        for j, dst in enumerate(node_names):
            if causal_graph.iloc[i, j] == -1:  # Directed edge only
                graph.add_edge(src, dst)

    return graph


def setup_system_logging(config_manager: ConfigManager) -> logging.Logger:
    """Setup system-wide logging."""
    log_config = config_manager.get_section('logging')
    log_level = getattr(logging, log_config.get('level', 'INFO'))
    log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = log_config.get('file')

    return setup_logger(level=log_level, log_file=log_file, format_string=log_format)


def load_data(config_manager: ConfigManager, logger: logging.Logger) -> DataContainer:
    """Load data using configured data loader."""
    logger.info("=" * 60)
    logger.info("STEP 1: Data Loading")
    logger.info("=" * 60)

    data_config = config_manager.get_section('data_loading')
    data_config['context'] = config_manager.get_section('context')

    # Select loader based on type
    loader_type = data_config.get('type', 'csv')

    if loader_type == 'radice':
        loader = RADICELoader(data_config)
    else:
        loader = CSVDataLoader(data_config)

    container = loader.load()

    logger.info(f"Loaded data: {container.get_summary()}")

    return container


def preprocess_data(config_manager: ConfigManager, container: DataContainer,
                   logger: logging.Logger) -> pd.DataFrame:
    """Preprocess the loaded data."""
    logger.info("=" * 60)
    logger.info("STEP 2: Preprocessing")
    logger.info("=" * 60)

    preprocessing_config = config_manager.get_section('preprocessing')

    pipeline = PreprocessingPipeline(preprocessing_config)
    processed_data = pipeline.run(container.get_metric_data())

    logger.info(f"Preprocessed data shape: {processed_data.shape}")

    return processed_data


def generate_constraints(config_manager: ConfigManager, container: DataContainer,
                        logger: logging.Logger) -> tuple:
    """
    Generate knowledge constraints and create causal-learn BackgroundKnowledge object.

    Args:
        config_manager: Configuration manager instance
        container: Data container with loaded data (may contain level_map from layer.txt)
        logger: Logger instance

    Returns:
        Tuple of (background_knowledge, level_map)
    """
    logger.info("=" * 60)
    logger.info("STEP 3: Knowledge Constraints")
    logger.info("=" * 60)

    constraints_path = config_manager.get('knowledge.constraints_path')

    if not Path(constraints_path).exists():
        logger.warning(f"Constraints file not found: {constraints_path}")
        return None, {}

    knowledge_manager = KnowledgeManager(constraints_path)
    metric_names = container.get_metric_names()

    # Check if level constraints are enabled (for ablation study)
    use_level_constraints = config_manager.get('knowledge.use_level_constraints', True)

    # Get level_map from layer.txt (if available, e.g., RADICE data)
    level_map_from_layer = container.metadata.get('level_map', {})

    # Get level_map from YAML configuration
    level_map_from_yaml = knowledge_manager.get_level_map(metric_names)

    # Merge level maps: layer.txt takes precedence over YAML
    if use_level_constraints:
        level_map = {**level_map_from_yaml, **level_map_from_layer}
        if level_map_from_layer:
            logger.info(f"Merged level_map: {len(level_map_from_yaml)} from YAML, {len(level_map_from_layer)} from layer.txt")
        else:
            logger.info(f"Using level_map from YAML: {len(level_map)} entries")
    else:
        level_map = {}
        logger.info("Level constraints DISABLED (ablation study mode)")

    # Generate causal-learn BackgroundKnowledge object
    background_knowledge = knowledge_manager.create_causal_learn_background_knowledge(
        metric_names=metric_names,
        level_map=level_map
    )

    # Log explicit constraints info
    explicit_forbidden = knowledge_manager.config.get('explicit_forbidden', [])
    explicit_required = knowledge_manager.config.get('explicit_required', [])
    logger.info(f"Created BackgroundKnowledge with {len(level_map)} tier assignments")
    logger.info(f"Explicit constraints: {len(explicit_forbidden)} forbidden, {len(explicit_required)} required")

    return background_knowledge, level_map


def run_causal_discovery(config_manager: ConfigManager, data: pd.DataFrame,
                        background_knowledge, level_map: dict,
                        logger: logging.Logger) -> dict:
    """
    Run causal discovery algorithm.

    Args:
        config_manager: Configuration manager instance
        data: Preprocessed data
        background_knowledge: causal-learn BackgroundKnowledge object
        level_map: Dictionary mapping node names to levels
        logger: Logger instance

    Returns:
        Algorithm results dictionary
    """
    logger.info("=" * 60)
    logger.info("STEP 4: Causal Discovery")
    logger.info("=" * 60)

    algorithm_config = config_manager.get_section('algorithm')
    algorithm_name = algorithm_config.get('name', 'pc')
    algorithm_params = algorithm_config.get('params', {})

    # Get output directory (already resolved by ConfigManager)
    output_dir = algorithm_config.get('output_dir')

    logger.info(f"Running {algorithm_name} algorithm")

    algorithm = AlgorithmFactory.create(algorithm_name, **algorithm_params)
    results = algorithm.run(
        data,
        background_knowledge=background_knowledge,
        level_map=level_map,
        output_dir=output_dir
    )

    logger.info(f"Causal discovery complete: {results.edge_count} edges, {results.skeleton_edge_count} skeleton edges")

    return results


def orient_graph(config_manager: ConfigManager, algorithm_results: AlgorithmResult,
                processed_data: pd.DataFrame, logger: logging.Logger) -> tuple:
    """Orient the causal graph using cascade orientation."""
    logger.info("=" * 60)
    logger.info("STEP 5: Cascade Orientation")
    logger.info("=" * 60)

    orientation_config = config_manager.get_section('orientation')

    orientator = CascadeOrientator(
        correlation_threshold=orientation_config.get('time_lag.correlation_threshold', 0.3),
        igci_confidence_threshold=orientation_config.get('igci.entropy_threshold', 0.01),
        max_time_lag=orientation_config.get('time_lag.max_lag', 10),
        max_iterations=orientation_config.get('cascade.max_iterations', 100)
    )

    # Get raw_graph from algorithm results (CausalGraph from PC algorithm)
    raw_graph = getattr(algorithm_results, 'raw_graph', None)
    if raw_graph is None:
        raise ValueError("Algorithm results must contain raw_graph (CausalGraph from PC algorithm)")

    # Get node names from skeleton graph
    node_names = algorithm_results.skeleton_graph.index.tolist()

    # Run cascade orientation on the CausalGraph
    oriented_cg = orientator.orient_graph(
        cg=raw_graph,
        data=processed_data,
        node_names=node_names,
        enable_time_lag=orientation_config.get('time_lag.enabled', True),
        enable_igci=orientation_config.get('igci.enabled', True)
    )

    # Convert CausalGraph to DataFrame for downstream use
    oriented_graph = orientator.causal_graph_to_dataframe(oriented_cg, node_names)

    # Compile metadata
    tracker_summary = orientator.get_tracker().get_summary()
    metadata = {
        'tracker_summary': tracker_summary,
        'orientation_details': orientator.get_tracker().get_orientations_as_list()
    }

    logger.info(f"Orientation complete: {tracker_summary['total_orientations']} orientations made")

    return oriented_graph, metadata


def run_rca(config_manager: ConfigManager, data: pd.DataFrame,
            container: DataContainer, causal_graph: pd.DataFrame,
            level_map: dict, logger: logging.Logger) -> dict:
    """Run root cause analysis."""
    logger.info("=" * 60)
    logger.info("STEP 6: Root Cause Analysis")
    logger.info("=" * 60)

    rca_config = config_manager.get_section('rca')

    if not rca_config.get('enabled', True):
        logger.info("RCA disabled, skipping")
        return None

    # Convert adjacency matrix to NetworkX graph
    graph = _build_graph_from_dataframe(causal_graph)
    logger.info(f"Built DiGraph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} directed edges")

    # Create RCA config
    rca_cfg = RCAConfig(
        max_shift=rca_config.get('max_shift', 1),
        max_width=rca_config.get('max_width', 2),
        shift_penalty=rca_config.get('shift_penalty', 0.004),
        smooth_penalty=rca_config.get('smooth_penalty', 0.01),
        min_similarity=rca_config.get('min_similarity', 0.5)
    )

    # Determine performance metric
    # Priority: RADICE ground truth symptom > config file > first column
    performance_metric = None

    # First, try to get from RADICE ground truth (highest priority)
    gt = container.metadata.get('ground_truth', {})
    logger.debug(f"Ground truth metadata keys: {list(gt.keys()) if gt else 'None'}")

    if gt and 'symptoms' in gt and gt['symptoms']:
        # Use the first symptom as the performance metric
        symptom_id = gt['symptoms'][0]
        logger.debug(f"Found symptom in ground_truth: {symptom_id}")

        # Find the column name corresponding to the symptom ID
        if symptom_id in data.columns:
            performance_metric = symptom_id
            logger.info(f"Using performance metric from root_cause.txt (symptom): {performance_metric}")
        else:
            logger.warning(f"Symptom '{symptom_id}' not found in data columns")

    # If not found in ground truth, try config file
    if not performance_metric:
        config_metric = rca_config.get('performance_metric', '').strip()
        if config_metric and config_metric in data.columns:
            performance_metric = config_metric
            logger.info(f"Using configured performance metric: {performance_metric}")
        elif config_metric:
            logger.warning(f"Configured performance metric '{config_metric}' not found in data columns")

    # Final fallback: use first column
    if not performance_metric:
        performance_metric = data.columns[0]
        logger.warning(f"Performance metric not found, using first column: {performance_metric}")

    # Run RCA
    engine = RCAEngine(rca_cfg)

    results = engine.analyze(
        data=data,
        causal_graph=graph,
        performance_metric=performance_metric,
        node_levels=level_map
    )

    logger.info(f"RCA complete: {len(results.root_causes)} root causes found")

    # Print summary
    engine.print_summary(results)

    return {
        'root_causes': results.root_causes,
        'subgraph': results.subgraph,
        'node_classifications': results.node_classifications,
        'correlation_results': results.correlation_results,
        'config': results.config,
        'performance_metric': performance_metric
    }


def run_evaluation(config_manager: ConfigManager, causal_graph: pd.DataFrame,
                   rca_results: dict, logger: logging.Logger) -> dict:
    """Run evaluation against ground truth."""
    logger.info("=" * 60)
    logger.info("STEP 7: Evaluation")
    logger.info("=" * 60)

    eval_config = config_manager.get_section('evaluation')

    if not eval_config.get('enabled', False):
        logger.info("Evaluation disabled, skipping")
        return None

    # Get ground_truth_path
    ground_truth_path = eval_config.get('ground_truth_path')

    # If not configured, try to build from data_loading path (for RADICE datasets)
    if not ground_truth_path:
        data_config = config_manager.get_section('data_loading')
        dataset_path = data_config.get('dataset_path', '')
        sample_name = data_config.get('sample_name', '')

        if dataset_path and sample_name:
            ground_truth_path = str(Path(dataset_path) / sample_name)
            logger.info(f"Auto-detected ground truth path: {ground_truth_path}")

    if not ground_truth_path:
        logger.warning("Ground truth path not configured")
        return None

    gt_path = Path(ground_truth_path)
    if not gt_path.exists():
        logger.warning(f"Ground truth path not found: {ground_truth_path}")
        return None

    # Convert causal graph to NetworkX
    graph = _build_graph_from_dataframe(causal_graph)
    logger.info(f"Built DiGraph for evaluation: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} directed edges")

    # Build RCA subgraph if available
    rca_subgraph = _build_rca_subgraph(rca_results.get('subgraph') if rca_results else None)
    if rca_subgraph:
        logger.info(f"Using RCA subgraph: {rca_subgraph.number_of_nodes()} nodes, {rca_subgraph.number_of_edges()} edges")

    # Run evaluation
    evaluator = EvaluationEngine()
    eval_results = evaluator.evaluate(
        causal_graph=graph,
        rca_subgraph=rca_subgraph,
        rca_results=rca_results,
        ground_truth_path=str(gt_path)
    )

    logger.info("Evaluation complete")

    # Print results
    if 'graph' in eval_results:
        g = eval_results['graph']
        logger.info(f"Graph: F1={g['f1']:.3f}, F1-S={g['skeleton_f1']:.3f}, SHD={g['shd']}")

    if 'subgraph' in eval_results:
        sg = eval_results['subgraph']
        logger.info(f"Subgraph: F1={sg['f1']:.3f}, F1-S={sg['skeleton_f1']:.3f}, SHD={sg['shd']}")

    if 'rca' in eval_results:
        r = eval_results['rca']
        logger.info(
            f"RCA (this sample): avg@5={r.get('avg@5', 0):.3f}, "
            f"acc@1={r.get('acc@1', 0):.3f}, acc@3={r.get('acc@3', 0):.3f}, "
            f"P={r.get('precision', 0):.3f}, R={r.get('recall', 0):.3f}"
        )

    return eval_results


def _build_rca_subgraph(subgraph_obj: Any) -> Optional[nx.DiGraph]:
    """Build RCA subgraph from various formats."""
    if subgraph_obj is None:
        return None
    if isinstance(subgraph_obj, nx.DiGraph):
        return subgraph_obj
    if isinstance(subgraph_obj, list):
        graph = nx.DiGraph()
        for edge in subgraph_obj:
            if isinstance(edge, (list, tuple)) and len(edge) >= 2:
                graph.add_edge(str(edge[0]), str(edge[1]))
        return graph
    return None


def save_results(config_manager: ConfigManager, processed_data: pd.DataFrame,
                algorithm_results: dict, oriented_graph: pd.DataFrame,
                orientation_metadata: dict, rca_results: dict,
                eval_results: dict, logger: logging.Logger) -> None:
    """Save all results."""
    logger.info("=" * 60)
    logger.info("STEP 8: Saving Results")
    logger.info("=" * 60)

    output_config = config_manager.get_section('output')

    # Get output directory (already resolved by ConfigManager)
    output_dir = output_config.get('base_dir')

    # Derive RCA output directory from output_dir
    # Pattern: data/output/{dataset}/{case}/{sample}/ -> data/rca_result/{dataset}/{case}/{sample}/
    output_path = Path(output_dir)
    if 'output' in output_path.parts:
        idx = output_path.parts.index('output')
        # Replace 'output' with 'rca_result'
        rca_output_dir = Path(*list(output_path.parts[:idx]) + ['rca_result'] + list(output_path.parts[idx+1:]))
    else:
        rca_output_dir = output_path.parent / 'rca_result' / output_path.name

    saver = ResultSaver(
        output_dir=output_dir,
        save_intermediate=output_config.get('save_intermediate', True),
        rca_output_dir=str(rca_output_dir)
    )

    # Compute correlation matrix if not already saved
    correlation_matrix = None
    if processed_data is not None:
        correlation_matrix = processed_data.corr(method='pearson')

    saver.save_all(
        processed_data=processed_data,
        algorithm_results=algorithm_results,
        oriented_graph=oriented_graph,
        orientation_metadata=orientation_metadata,
        rca_results=rca_results,
        eval_results=eval_results,
        correlation_matrix=correlation_matrix
    )

    logger.info(f"All results saved to: {output_dir}")


def main(config_path: str = "config/config.yaml"):
    """Main pipeline entry point."""
    # Initialize configuration (ConfigManager handles path resolution internally)
    config_manager = ConfigManager(config_path)

    # Setup logging
    logger = setup_system_logging(config_manager)

    logger.info("=" * 60)
    logger.info("Microservice Causal Discovery System v2.0")
    logger.info("=" * 60)
    logger.info(f"Configuration: {config_path}")
    logger.info(f"Dataset: {config_manager.get('context.dataset_name')}")
    logger.info(f"Case: {config_manager.get('context.case_name')}")

    # Start timing
    pipeline_start_time = time.time()
    step_times = {}

    try:
        # Step 1: Load data
        step_start = time.time()
        container = load_data(config_manager, logger)
        step_times['load_data'] = time.time() - step_start

        # Step 2: Preprocess data
        step_start = time.time()
        processed_data = preprocess_data(config_manager, container, logger)
        step_times['preprocess'] = time.time() - step_start

        # Step 3: Generate constraints
        step_start = time.time()
        background_knowledge, level_map = generate_constraints(
            config_manager, container, logger
        )
        step_times['constraints'] = time.time() - step_start

        # Step 4: Causal discovery
        step_start = time.time()
        algorithm_results = run_causal_discovery(
            config_manager, processed_data, background_knowledge, level_map, logger
        )
        step_times['causal_discovery'] = time.time() - step_start

        # Step 5: Orientation
        step_start = time.time()
        oriented_graph, orientation_metadata = orient_graph(
            config_manager, algorithm_results,
            processed_data, logger
        )
        step_times['orientation'] = time.time() - step_start

        # Step 6: Root cause analysis
        step_start = time.time()
        rca_results = run_rca(
            config_manager, processed_data, container, oriented_graph, level_map, logger
        )
        step_times['rca'] = time.time() - step_start

        # Step 7: Evaluation
        step_start = time.time()
        eval_results = run_evaluation(
            config_manager, oriented_graph, rca_results, logger
        )
        step_times['evaluation'] = time.time() - step_start

        # Step 8: Save results
        step_start = time.time()
        save_results(
            config_manager, processed_data, algorithm_results,
            oriented_graph, orientation_metadata, rca_results, eval_results, logger
        )
        step_times['save'] = time.time() - step_start

        # Calculate total time
        total_time = time.time() - pipeline_start_time

        # Print timing summary
        logger.info("=" * 60)
        logger.info("Pipeline completed successfully")
        logger.info("=" * 60)
        logger.info("Execution Time Summary:")
        logger.info(f"  Step 1 - Data Loading:        {step_times.get('load_data', 0):.2f}s")
        logger.info(f"  Step 2 - Preprocessing:       {step_times.get('preprocess', 0):.2f}s")
        logger.info(f"  Step 3 - Constraints:         {step_times.get('constraints', 0):.2f}s")
        logger.info(f"  Step 4 - Causal Discovery:   {step_times.get('causal_discovery', 0):.2f}s")
        logger.info(f"  Step 5 - Orientation:         {step_times.get('orientation', 0):.2f}s")
        logger.info(f"  Step 6 - RCA:                {step_times.get('rca', 0):.2f}s")
        logger.info(f"  Step 7 - Evaluation:         {step_times.get('evaluation', 0):.2f}s")
        logger.info(f"  Step 8 - Save Results:       {step_times.get('save', 0):.2f}s")
        logger.info("-" * 60)
        logger.info(f"  TOTAL TIME:                   {total_time:.2f}s")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Microservice Causal Discovery System")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                        help="Path to configuration file")
    args = parser.parse_args()

    main(args.config)
