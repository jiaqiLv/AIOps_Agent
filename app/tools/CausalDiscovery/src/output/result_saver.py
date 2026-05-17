"""
Result saver for saving all intermediate and final results.

This module handles saving all outputs from the causal discovery pipeline.
"""

import os
import json
import pandas as pd
import networkx as nx
from typing import Dict, Any, Optional, Union
from pathlib import Path
import logging

from src.algorithms.base_algorithm import AlgorithmResult

logger = logging.getLogger(__name__)


class ResultSaver:
    """
    Saves all results from the causal discovery pipeline.

    Output directories:
    - Causal discovery results: data/output/{dataset}/{case}/{sample}/
      * final_graph.csv - Final oriented causal graph
      * intermediate/ - Algorithm intermediate results
        - skeleton_graph.csv - Skeleton graph (undirected edges)
        - independence_tests.csv - CI test results
        - separation_sets.json - Separation sets
        - algorithm_stats.json - Algorithm statistics

    - Orientation results: data/output/{dataset}/{case}/{sample}/orientation/
      * orientation_details.csv - Edge orientation details
      * orientation_summary.json - Orientation summary statistics

    - RCA & Evaluation results: data/rca_result/{dataset}/{case}/{sample}/
      * evaluation_results.json - Evaluation metrics (if enabled)
      * root_cause.txt - RADICE format root causes
      * root_causes.csv - Root causes list (CSV format)
      * adjusted_correlation.csv - Adjusted correlation scores
      * subgraph.csv - RADICE format fault propagation subgraph

    Note: Preprocessed data and correlation matrix are saved by the
    preprocessing pipeline to data/processed/{dataset}/{case}/{sample}/
    """

    def __init__(self, output_dir: str, save_intermediate: bool = True,
                 rca_output_dir: Optional[str] = None):
        """
        Initialize the result saver.

        Args:
            output_dir: Base directory for causal discovery output files (already resolved)
            save_intermediate: Whether to save intermediate results
            rca_output_dir: Optional custom path for RCA results
                        If None, defaults to data/rca_result/{dataset}/{case}/{sample}/
        """
        self.save_intermediate = save_intermediate
        self.logger = logging.getLogger(self.__class__.__name__)

        # output_dir is already resolved by ConfigManager
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Intermediate results directory
        self.intermediate_dir = self.output_dir / "intermediate"
        if save_intermediate:
            self.intermediate_dir.mkdir(exist_ok=True)

        # Orientation results directory (NEW)
        self.orientation_dir = self.output_dir / "orientation"
        self.orientation_dir.mkdir(parents=True, exist_ok=True)

        # RCA output directory
        if rca_output_dir:
            # rca_output_dir is already resolved by ConfigManager
            self.rca_output_dir = Path(rca_output_dir)
        else:
            # Derive from output_dir: data/output/... -> data/rca_result/...
            parts = self.output_dir.parts
            if 'output' in parts:
                idx = parts.index('output')
                # Combine: parts before 'output' + 'rca_result' + parts after 'output'
                base_parts = list(parts[:idx]) + ['rca_result'] + list(parts[idx+1:])
                self.rca_output_dir = Path(*base_parts)
            else:
                # Fallback: rca_results as subdirectory of output_dir
                self.rca_output_dir = self.output_dir / 'rca_results'

        self.rca_output_dir.mkdir(parents=True, exist_ok=True)

    def save_all(self,
                processed_data: Optional[pd.DataFrame] = None,
                algorithm_results: Optional[Dict[str, Any]] = None,
                oriented_graph: Optional[pd.DataFrame] = None,
                rca_results: Optional[Dict[str, Any]] = None,
                eval_results: Optional[Dict[str, Any]] = None,
                orientation_metadata: Optional[Dict[str, Any]] = None,
                correlation_matrix: Optional[pd.DataFrame] = None) -> None:
        """
        Save all results.

        Note: Preprocessed data and correlation matrix are saved to their
        configured output paths, not duplicated here.

        Args:
            processed_data: Preprocessed data (saved to preprocessing output path)
            algorithm_results: Results from causal discovery algorithm
            oriented_graph: Final oriented causal graph
            rca_results: Root cause analysis results
            eval_results: Evaluation results
            orientation_metadata: Orientation metadata
            correlation_matrix: Pearson correlation matrix (saved to preprocessing output path)
        """
        self.logger.info(f"Saving causal discovery results to: {self.output_dir}")
        self.logger.info(f"Orientation results to: {self.orientation_dir}")
        self.logger.info(f"RCA results will be saved to: {self.rca_output_dir}")

        # Save algorithm results
        if algorithm_results is not None:
            self.save_algorithm_results(algorithm_results)

        # Save oriented graph
        if oriented_graph is not None:
            self.save_final_graph(oriented_graph)

        # Save orientation details
        if orientation_metadata is not None:
            self.save_orientation_details(orientation_metadata)

        # Save RCA results to dedicated directory
        if rca_results is not None:
            self.save_rca_results(rca_results)

        # Save evaluation results
        if eval_results is not None:
            self.save_evaluation_results(eval_results)

        self.logger.info("All results saved successfully")

    def save_algorithm_results(self, results: Union[AlgorithmResult, Dict[str, Any]]) -> None:
        """Save algorithm results."""
        if not self.save_intermediate:
            return

        # Handle AlgorithmResult object
        if hasattr(results, 'skeleton_graph'):
            # Save skeleton graph
            path = self.intermediate_dir / "skeleton_graph.csv"
            results.skeleton_graph.to_csv(path)
            self.logger.info(f"Saved skeleton graph: {path}")

            # Save independence tests
            if hasattr(results, 'independence_tests') and results.independence_tests:
                path = self.intermediate_dir / "independence_tests.csv"
                pd.DataFrame(results.independence_tests).to_csv(path, index=False)
                self.logger.info(f"Saved independence tests: {path}")

            # Save separation sets
            if hasattr(results, 'separation_sets') and results.separation_sets:
                path = self.intermediate_dir / "separation_sets.json"
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(results.separation_sets, f, indent=2, ensure_ascii=False)
                self.logger.info(f"Saved separation sets: {path}")
        elif isinstance(results, dict):
            # Handle dictionary format (for backward compatibility)
            if 'skeleton_graph' in results:
                path = self.intermediate_dir / "skeleton_graph.csv"
                results['skeleton_graph'].to_csv(path)
                self.logger.info(f"Saved skeleton graph: {path}")

            if 'independence_tests' in results and results['independence_tests']:
                path = self.intermediate_dir / "independence_tests.csv"
                pd.DataFrame(results['independence_tests']).to_csv(path, index=False)
                self.logger.info(f"Saved independence tests: {path}")

            if 'separation_sets' in results:
                path = self.intermediate_dir / "separation_sets.json"
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(results['separation_sets'], f, indent=2, ensure_ascii=False)
                self.logger.info(f"Saved separation sets: {path}")

    def save_final_graph(self, graph: pd.DataFrame, filename: str = "final_graph.csv") -> None:
        """Save final causal graph."""
        path = self.output_dir / filename
        graph.to_csv(path)
        self.logger.info(f"Saved final graph: {path}")

    def save_orientation_details(self, metadata: Dict[str, Any]) -> None:
        """Save orientation details to the orientation directory."""
        if 'orientation_details' in metadata:
            path = self.orientation_dir / "orientation_details.csv"
            pd.DataFrame(metadata['orientation_details']).to_csv(path, index=False)
            self.logger.info(f"Saved orientation details: {path}")

        if 'tracker_summary' in metadata:
            path = self.orientation_dir / "orientation_summary.json"
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(metadata['tracker_summary'], f, indent=2, ensure_ascii=False)
            self.logger.info(f"Saved orientation summary: {path}")

    def save_rca_results(self, results: Dict[str, Any]) -> None:
        """
        Save root cause analysis results to the dedicated RCA output directory.

        Args:
            results: RCA results dictionary
        """
        # Use the dedicated RCA output directory
        rca_dir = self.rca_output_dir
        rca_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Saving RCA results to: {rca_dir}")

        # Save root_cause.txt (RADICE format)
        if 'root_causes' in results:
            # Format: first line = root causes (space-separated)
            #         second line = symptoms (space-separated)
            root_causes_line = ' '.join(str(rc) for rc in results['root_causes'])

            # Get symptoms (intermediate nodes or symptoms from subgraph)
            symptoms = []
            if 'symptoms' in results:
                symptoms = results['symptoms']
            elif 'node_classifications' in results:
                symptoms = [name for name, cls in results['node_classifications'].items()
                           if cls == 'intermediate' or cls == 'symptom']
            symptoms_line = ' '.join(str(s) for s in symptoms)

            path = rca_dir / "root_cause.txt"
            with open(path, 'w', encoding='utf-8') as f:
                f.write(root_causes_line + '\n')
                if symptoms_line:
                    f.write(symptoms_line + '\n')
            self.logger.info(f"Saved root_cause.txt: {path}")

        # Also save as CSV for easy reading
        if 'root_causes' in results:
            path = rca_dir / "root_causes.csv"
            pd.DataFrame({'root_cause': results['root_causes']}).to_csv(path, index=False)
            self.logger.info(f"Saved root causes: {path}")

        # Save correlation results
        if 'correlation_results' in results:
            corr_data = []
            for name, data in results['correlation_results'].items():
                if isinstance(data, tuple) and len(data) >= 3:
                    score, corr, penalty = data[0], data[1], data[2]
                else:
                    score, corr, penalty = data, 0, 0
                corr_data.append({
                    'metric': name,
                    'score': score,
                    'correlation': corr,
                    'penalty': penalty
                })

            if corr_data:
                path = rca_dir / "adjusted_correlation.csv"
                pd.DataFrame(corr_data).to_csv(path, index=False)
                self.logger.info(f"Saved adjusted correlation: {path}")

        # Save fault propagation graph (subgraph.csv - RADICE format)
        if 'subgraph' in results:
            self._save_graph_radice_format(results['subgraph'], rca_dir / "subgraph.csv")
            self.logger.info(f"Saved fault propagation subgraph: {rca_dir / 'subgraph.csv'}")

    def save_evaluation_results(self, results: Dict[str, Any], filename: str = "evaluation_results.json") -> None:
        """Save evaluation results to rca_result directory."""
        path = self.rca_output_dir / filename
        self._make_serializable_and_save(results, path)
        self.logger.info(f"Saved evaluation results: {path}")

    def _save_graph_radice_format(self, graph: nx.DiGraph, output_path: Path) -> None:
        """
        Save graph in RADICE format (src dst per line).

        Args:
            graph: NetworkX DiGraph
            output_path: Path to save the file
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            for src, dst in graph.edges():
                f.write(f"{src} {dst}\n")
            # Empty line at end (RADICE format)
            f.write('\n')

    def _make_serializable_and_save(self, obj: Any, path: Path) -> None:
        """Make object serializable and save to JSON."""
        def convert(o):
            if isinstance(o, dict):
                return {k: convert(v) for k, v in o.items()}
            elif isinstance(o, (list, tuple)):
                return [convert(item) for item in o]
            elif isinstance(o, (int, float, str, bool, type(None))):
                return o
            else:
                return str(o)

        serializable = convert(obj)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)