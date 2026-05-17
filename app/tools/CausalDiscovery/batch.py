"""
Batch processing pipeline for RADICE datasets.

This module provides functionality to run the causal discovery and RCA pipeline
on all samples in a RADICE dataset.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
import json

import pandas as pd
import networkx as nx

from src.utils.config_manager import ConfigManager
from src.dataloader.loaders.radice_loader import RADICELoader
from src.preprocessing.pipeline import PreprocessingPipeline
from src.knowledge.knowledge_manager import KnowledgeManager
from src.algorithms.factory import AlgorithmFactory
from src.orientation.cascade_orientator import CascadeOrientator
from src.rca.rca_engine import RCAEngine, RCAConfig
from src.output.result_saver import ResultSaver

logger = logging.getLogger(__name__)


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


class BatchProcessor:
    """
    Batch processor for datasets.

    Runs the complete pipeline on all samples in a dataset:
    1. Load data and ground truth
    2. Preprocess
    3. Generate constraints (including layer.txt)
    4. Run causal discovery
    5. Orient edges
    6. Run RCA
    7. Evaluate
    8. Save results
    """

    def __init__(self, config_manager: ConfigManager, dataset_name: str):
        """
        Initialize the batch processor.

        Args:
            config_manager: ConfigManager instance with loaded configuration
            dataset_name: Dataset name (e.g., 'N5', 'N10', 'N15', 'N25')
        """
        self.config_manager = config_manager
        self.dataset_name = dataset_name

        # Get paths from configuration (already resolved by ConfigManager)
        data_config = config_manager.get_section('data_loading')
        batch_config = config_manager.get_section('batch')

        # dataset_path from config is already resolved with variable substitution
        # We just need to use it directly
        base_dataset_path = data_config.get('dataset_path', 'data/raw/RADICE')
        self.dataset_path = Path(base_dataset_path)

        # Get output directories from batch config (already resolved)
        self.output_base_dir = Path(batch_config.get('output_base_dir', 'data/output/RADICE'))
        self.rca_output_base_dir = Path(batch_config.get('rca_output_base_dir', 'data/rca_result'))
        self.processed_output_dir = Path(batch_config.get('processed_output_dir', 'data/processed'))

        self.logger = logging.getLogger(self.__class__.__name__)

        # List all samples
        self.samples = RADICELoader.list_samples(str(self.dataset_path))
        self.logger.info(f"Found {len(self.samples)} samples in {self.dataset_path}")

    def run_all(self, sample_indices: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Run pipeline on all (or selected) samples.

        Args:
            sample_indices: Optional list of sample indices to process
                           If None, processes all samples

        Returns:
            Summary of all results including dataset-level aggregated metrics
        """
        # Determine which samples to process
        if sample_indices is not None:
            samples_to_process = [self.samples[i] for i in sample_indices if 0 <= i < len(self.samples)]
        else:
            samples_to_process = self.samples

        self.logger.info(f"Processing {len(samples_to_process)} samples")

        all_results = {}
        summary = {
            'total_samples': len(samples_to_process),
            'successful': 0,
            'failed': 0,
            'results': {},
            'aggregate_metrics': None
        }

        # Create EvaluationEngine for dataset-level aggregation
        from src.evaluation.evaluator import EvaluationEngine
        evaluator = EvaluationEngine()

        for i, sample_name in enumerate(samples_to_process):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Processing sample {i+1}/{len(samples_to_process)}: {sample_name}")
            self.logger.info(f"{'='*60}")

            try:
                result = self.run_sample(sample_name)
                all_results[sample_name] = result
                summary['successful'] += 1
                summary['results'][sample_name] = 'success'

                # Add evaluation results to dataset evaluator
                if result.get('evaluation'):
                    evaluator.dataset_evaluator.add_sample_result(result['evaluation'])

                self.logger.info(f"Successfully processed: {sample_name}")

            except Exception as e:
                self.logger.error(f"Failed to process {sample_name}: {e}", exc_info=True)
                summary['failed'] += 1
                summary['results'][sample_name] = f'failed: {str(e)}'

        # Compute dataset-level aggregated metrics
        if summary['successful'] > 0:
            aggregate_metrics = evaluator.dataset_evaluator.compute_aggregate_metrics()
            summary['aggregate_metrics'] = aggregate_metrics

            # Print aggregate results
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Dataset-Level Aggregated Metrics ({summary['successful']} samples)")
            self.logger.info(f"{'='*60}")

            if 'graph_f1' in aggregate_metrics:
                self.logger.info(f"Graph F1:   {aggregate_metrics['graph_f1']:.3f}")
            if 'graph_skeleton_f1' in aggregate_metrics:
                self.logger.info(f"Graph F1-S: {aggregate_metrics['graph_skeleton_f1']:.3f}")
            if 'graph_shd_floor' in aggregate_metrics:
                self.logger.info(f"Graph SHD:  {aggregate_metrics['graph_shd_floor']}")
            elif 'graph_shd' in aggregate_metrics:
                self.logger.info(f"Graph SHD:  {aggregate_metrics['graph_shd']:.1f}")

            if 'subgraph_f1' in aggregate_metrics:
                self.logger.info(f"Subgraph F1:   {aggregate_metrics['subgraph_f1']:.3f}")
            if 'subgraph_skeleton_f1' in aggregate_metrics:
                self.logger.info(f"Subgraph F1-S: {aggregate_metrics['subgraph_skeleton_f1']:.3f}")
            if 'subgraph_shd_floor' in aggregate_metrics:
                self.logger.info(f"Subgraph SHD:  {aggregate_metrics['subgraph_shd_floor']}")

            if 'rca_precision' in aggregate_metrics:
                self.logger.info(f"RCA P:       {aggregate_metrics['rca_precision']:.3f}")
            if 'rca_recall' in aggregate_metrics:
                self.logger.info(f"RCA R:       {aggregate_metrics['rca_recall']:.3f}")
            if 'rca_avg@5' in aggregate_metrics:
                self.logger.info(f"RCA Avg@5:   {aggregate_metrics['rca_avg@5']:.3f}")

            self.logger.info(f"{'='*60}")

        # Save summary
        self._save_summary(summary)

        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Batch processing complete: {summary['successful']}/{summary['total_samples']} successful")
        self.logger.info(f"{'='*60}")

        return summary

    def run_sample(self, sample_name: str) -> Dict[str, Any]:
        """
        Run the complete pipeline on a single sample.

        Args:
            sample_name: Sample folder name (e.g., artificialResults_0)

        Returns:
            Dictionary with all results
        """
        # Update context with current sample
        self.config_manager.set('context.sample_name', sample_name)
        self.logger.info(f"Set context.sample_name to: {sample_name}")
        self.logger.info(f"Current context: {self.config_manager._config.get('context', {})}")

        self.config_manager._resolve_all_paths()  # Re-resolve paths with new sample

        # Check the resolved output_path
        resolved_output_path = self.config_manager.get('preprocessing.output_path')
        self.logger.info(f"Resolved preprocessing.output_path: {resolved_output_path}")

        # Step 1: Load data
        loader_config = self.config_manager.get_section('data_loading').copy()
        loader_config['dataset_path'] = str(self.dataset_path)
        loader_config['sample_name'] = sample_name
        loader_config['load_ground_truth'] = True

        loader = RADICELoader(loader_config)
        container = loader.load()

        original_data = container.get_metric_data()
        level_map_from_layer = container.metadata.get('level_map', {})
        ground_truth = container.metadata.get('ground_truth')

        # Step 2: Preprocess
        self.logger.info("Preprocessing data...")
        # Get fresh config after path resolution
        prep_config = self.config_manager._config.get('preprocessing', {})
        self.logger.info(f"Creating PreprocessingPipeline with output_path: {prep_config.get('output_path')}")
        pipeline = PreprocessingPipeline(prep_config)
        processed_data = pipeline.run(original_data)

        # Compute correlation matrix (for downstream use)
        correlation_matrix = processed_data.corr()

        # Note: Processed data and correlation matrix are saved by PreprocessingPipeline
        # to the path specified in preprocessing.output_path

        # Step 3: Generate constraints
        self.logger.info("Generating constraints...")
        metric_names = processed_data.columns.tolist()

        # Check if level constraints are enabled (for ablation study)
        knowledge_config = self.config_manager.get_section('knowledge')
        use_level_constraints = knowledge_config.get('use_level_constraints', True)

        # Use layer.txt levels if available
        if level_map_from_layer and use_level_constraints:
            self.logger.info(f"Using level_map from layer.txt: {len(level_map_from_layer)} entries")
        elif not use_level_constraints:
            self.logger.info("Level constraints DISABLED (ablation study mode)")
            level_map_from_layer = {}

        # Create BackgroundKnowledge object from layer.txt
        level_map = level_map_from_layer.copy() if use_level_constraints else {}
        background_knowledge = None

        if level_map:
            # Import BackgroundKnowledgeBuilder
            from src.knowledge.background_knowledge import BackgroundKnowledgeBuilder
            builder = BackgroundKnowledgeBuilder()
            background_knowledge = builder.build_background_knowledge(
                metric_names=metric_names,
                level_map=level_map
            )
            self.logger.info(f"Created BackgroundKnowledge with {len(level_map)} tier assignments")

        # Step 4: Run causal discovery
        self.logger.info("Running causal discovery...")
        algo_config = self.config_manager.get_section('algorithm')
        algorithm = AlgorithmFactory.create(
            algo_config.get('name', 'pc'),
            **algo_config.get('params', {})
        )

        algorithm_results = algorithm.run(
            processed_data,
            background_knowledge=background_knowledge,
            level_map=level_map,
            output_dir=str(self.output_base_dir / self.dataset_path.name / sample_name / 'intermediate')
        )

        # Step 5: Orientation
        self.logger.info("Running orientation...")
        orient_config = self.config_manager.get_section('orientation')
        orientator = CascadeOrientator(
            correlation_threshold=orient_config.get('time_lag.correlation_threshold', 0.3),
            igci_confidence_threshold=orient_config.get('igci.entropy_threshold', 0.01),
            max_time_lag=orient_config.get('time_lag.max_lag', 10),
            max_iterations=orient_config.get('cascade.max_iterations', 100)
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
            enable_time_lag=orient_config.get('time_lag.enabled', True),
            enable_igci=orient_config.get('igci.enabled', True)
        )

        # Convert CausalGraph to DataFrame for downstream use
        oriented_graph = orientator.causal_graph_to_dataframe(oriented_cg, node_names)

        # Get orientation metadata
        orientation_metadata = {
            'tracker_summary': orientator.get_tracker().get_summary(),
            'orientation_details': orientator.get_tracker().get_orientations_as_list()
        }

        # Step 6: RCA
        self.logger.info("Running RCA...")
        rca_config = self.config_manager.get_section('rca')
        rca_cfg = RCAConfig(
            max_shift=rca_config.get('max_shift', 1),
            max_width=rca_config.get('max_width', 2),
            shift_penalty=rca_config.get('shift_penalty', 0.004),
            smooth_penalty=rca_config.get('smooth_penalty', 0.01),
            min_similarity=rca_config.get('min_similarity', 0.5)
        )

        # Convert to NetworkX (only directed edges)
        graph = _build_graph_from_dataframe(oriented_graph)

        # Determine performance metric (use last node or symptom if available)
        node_names = oriented_graph.index.tolist()
        if ground_truth and ground_truth.get('symptoms'):
            performance_metric = ground_truth['symptoms'][0]
        else:
            performance_metric = node_names[-1]

        rca_engine = RCAEngine(rca_cfg)
        rca_result = rca_engine.analyze(
            data=processed_data,
            causal_graph=graph,
            performance_metric=performance_metric,
            node_levels=level_map
        )

        # Convert RCAResult to dict for compatibility with evaluation and saving
        rca_results = {
            'root_causes': rca_result.root_causes,
            'subgraph': rca_result.subgraph,
            'node_classifications': rca_result.node_classifications,
            'correlation_results': rca_result.correlation_results,
            'config': rca_result.config
        }

        # Add symptoms to rca_results
        if ground_truth and ground_truth.get('symptoms'):
            rca_results['symptoms'] = ground_truth['symptoms']

        # Step 7: Evaluate (using EvaluationEngine for consistency with run.py)
        eval_results = None
        if ground_truth:
            self.logger.info("Running evaluation...")

            # Build ground truth path
            gt_path = self.dataset_path / sample_name

            # Build RCA subgraph if available
            rca_subgraph = _build_rca_subgraph(rca_results.get('subgraph'))

            # Run evaluation using EvaluationEngine
            from src.evaluation.evaluator import EvaluationEngine
            evaluator = EvaluationEngine()
            eval_results = evaluator.evaluate(
                causal_graph=graph,
                rca_subgraph=rca_subgraph,
                rca_results=rca_results,
                ground_truth_path=str(gt_path)
            )

            self.logger.info("Evaluation complete")

            # Print results
            if 'graph' in eval_results:
                g = eval_results['graph']
                self.logger.info(f"Graph: F1={g['f1']:.3f}, F1-S={g['skeleton_f1']:.3f}, SHD={g['shd']}")

            if 'subgraph' in eval_results:
                sg = eval_results['subgraph']
                self.logger.info(f"Subgraph: F1={sg['f1']:.3f}, F1-S={sg['skeleton_f1']:.3f}, SHD={sg['shd']}")

            if 'rca' in eval_results:
                r = eval_results['rca']
                self.logger.info(
                    f"RCA (this sample): avg@5={r.get('avg@5', 0):.3f}, "
                    f"acc@1={r.get('acc@1', 0):.3f}, acc@3={r.get('acc@3', 0):.3f}, "
                    f"P={r.get('precision', 0):.3f}, R={r.get('recall', 0):.3f}"
                )

        # Step 8: Save results
        self.logger.info("Saving results...")

        # Build output directory for this sample
        # output_base_dir is like "data/output/RADICE", dataset_path.name is "N15"
        # So final path is: data/output/RADICE/N15/artificialResults_0/
        final_output_dir = self.output_base_dir / self.dataset_path.name / sample_name
        final_output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Saving results to: {final_output_dir}")

        # Save final graph
        oriented_graph.to_csv(final_output_dir / 'final_graph.csv', index=True)

        # Save RCA results to rca_result folder
        # rca_output_base_dir is like "data/rca_result/RADICE", dataset_path.name is "N15"
        # So final path is: data/rca_result/RADICE/N15/artificialResults_0/
        rca_output_dir = self.rca_output_base_dir / self.dataset_path.name / sample_name

        saver = ResultSaver(str(final_output_dir), rca_output_dir=str(rca_output_dir))
        saver.save_rca_results(rca_results)

        # Save evaluation results to rca_result directory
        if eval_results:
            eval_path = rca_output_dir / 'evaluation_results.json'
            rca_output_dir.mkdir(parents=True, exist_ok=True)
            with open(eval_path, 'w') as f:
                json.dump(eval_results, f, indent=2)

        return {
            'sample_name': sample_name,
            'processed_data_shape': processed_data.shape,
            'edge_count': graph.number_of_edges(),
            'root_causes': rca_results['root_causes'],
            'evaluation': eval_results
        }

    def _save_summary(self, summary: Dict[str, Any]) -> None:
        """Save batch processing summary."""
        summary_path = self.rca_output_base_dir / self.dataset_path.name / 'batch_summary.json'
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Saved batch summary to: {summary_path}")


def main():
    """Main entry point for batch processing."""
    import argparse

    parser = argparse.ArgumentParser(description='RADICE Batch Processing')
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,  # No default, read from config if not provided
        choices=['N5', 'N10', 'N15', 'N25'],
        help='RADICE dataset name (N5, N10, N15, N25). If not specified, uses config file value.'
    )
    parser.add_argument(
        '--indices',
        type=str,
        default=None,
        help='Optional: comma-separated sample indices to process (e.g., "0,1,2")'
    )

    args = parser.parse_args()

    # Load configuration
    config_manager = ConfigManager(args.config)

    # Use dataset from command line if provided, otherwise use config file value
    dataset_name = args.dataset if args.dataset else config_manager.get('context.case_name')

    config_manager.set('context.case_name', dataset_name)
    config_manager._resolve_all_paths()

    # Setup logging
    log_config = config_manager.get_section('logging')
    log_level = getattr(logging, log_config.get('level', 'INFO'))
    log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = log_config.get('file')

    # Setup root logger to ensure all modules log at the same level
    import sys
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(log_format)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        from pathlib import Path as PathLib
        log_path = PathLib(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    logger = logging.getLogger(__name__)

    # Parse sample indices if provided
    sample_indices = None
    if args.indices:
        sample_indices = [int(x.strip()) for x in args.indices.split(',')]

    # Create processor and run
    processor = BatchProcessor(config_manager, dataset_name)

    print("=" * 60)
    print(f"RADICE Batch Processing: {dataset_name}")
    print(f"Dataset path: {processor.dataset_path}")
    print(f"Samples: {len(processor.samples)}")
    if sample_indices:
        print(f"Processing indices: {sample_indices}")
    print("=" * 60)

    summary = processor.run_all(sample_indices=sample_indices)

    # Print final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Total samples: {summary['total_samples']}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")

    if summary.get('aggregate_metrics'):
        agg = summary['aggregate_metrics']
        print("\nAggregate Metrics:")
        if 'graph_f1' in agg:
            print(f"  Graph F1:    {agg['graph_f1']:.3f}")
        if 'graph_skeleton_f1' in agg:
            print(f"  Graph F1-S:  {agg['graph_skeleton_f1']:.3f}")
        if 'graph_shd_floor' in agg:
            print(f"  Graph SHD:   {agg['graph_shd_floor']}")
        if 'rca_precision' in agg:
            print(f"  RCA P:       {agg['rca_precision']:.3f}")
        if 'rca_recall' in agg:
            print(f"  RCA R:       {agg['rca_recall']:.3f}")
        if 'rca_avg@5' in agg:
            print(f"  RCA Avg@5:   {agg['rca_avg@5']:.3f}")

    print("=" * 60)


if __name__ == "__main__":
    main()
