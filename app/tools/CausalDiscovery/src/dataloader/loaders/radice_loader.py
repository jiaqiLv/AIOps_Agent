"""
RADICE dataset data loader.

This module provides functionality to load RADICE format datasets which include:
- data.csv: Time series metric data
- layer.txt: Level constraints for metrics
- edges.txt: Ground truth causal graph
- subgraph.txt: Ground truth fault propagation subgraph
- root_cause.txt: Ground truth root causes and symptoms
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import logging

from src.dataloader.loaders.base_loader import BaseLoader
from src.dataloader.container import DataContainer

logger = logging.getLogger(__name__)


class RADICELoader(BaseLoader):
    """
    Data loader for RADICE format datasets.

    RADICE dataset structure:
    dataset_name/
    ├── artificialResults_0/
    │   ├── data.csv         # Time series data (first row is header)
    │   ├── layer.txt        # Level constraints: "node_id:level node_id:level ..."
    │   ├── edges.txt        # Ground truth edges: "src dst" per line
    │   ├── subgraph.txt     # Fault propagation subgraph edges
    │   └── root_cause.txt   # Root causes and symptoms
    ├── artificialResults_1/
    └── ...

    The loader reads:
    - data.csv: Metric time series data (first row is header)
    - layer.txt: Level constraints (optional, for background knowledge)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the RADICE data loader.

        Args:
            config: Configuration dictionary containing:
                - type: Must be "radice"
                - dataset_path: Path to the RADICE dataset folder (e.g., data/raw/{dataset_name}/{case_name})
                - sample_name: Sample folder name (e.g., artificialResults_0)
                - load_ground_truth: Whether to load ground truth files (default: True)
        """
        super().__init__(config)
        # dataset_path and sample_name are already resolved by ConfigManager
        self.dataset_path = config.get('dataset_path', '')
        self.sample_name = config.get('sample_name', '')
        self.load_ground_truth = config.get('load_ground_truth', True)

        logger.info(f"Initialized RADICELoader for dataset: {self.dataset_path}, sample: {self.sample_name}")

    def load(self) -> DataContainer:
        """
        Load data from the RADICE sample folder.

        Returns:
            DataContainer with the loaded data and metadata
        """
        # Resolve sample path (dataset_path already resolved by ConfigManager)
        sample_path = Path(self.dataset_path) / self.sample_name

        if not sample_path.exists():
            raise FileNotFoundError(f"Sample folder not found: {sample_path}")

        logger.info(f"Loading RADICE data from: {sample_path}")

        # Load data.csv
        data_path = Path(sample_path) / 'data.csv'
        if not data_path.exists():
            raise FileNotFoundError(f"data.csv not found in: {sample_path}")

        df = pd.read_csv(data_path)  # First row is header

        logger.info(f"Loaded data.csv: shape={df.shape}, columns={list(df.columns)}")

        # Load layer.txt if exists
        level_map = self._load_layer_txt(sample_path)

        # Load ground truth files if requested
        ground_truth = None
        if self.load_ground_truth:
            ground_truth = self._load_ground_truth(sample_path)

        # Create data container
        container = DataContainer(
            metric_data=df,
            timestamp_column=None,
            timestamp_index=False,
            metadata={
                'source_path': str(sample_path),
                'dataset_name': Path(self.dataset_path).name,
                'sample_name': self.sample_name,
                'level_map': level_map,
                'ground_truth': ground_truth
            }
        )

        logger.info(f"Loaded RADICE data: {df.shape[0]} samples, {df.shape[1]} metrics")

        return container

    def _load_layer_txt(self, sample_path: str) -> Dict[str, int]:
        """
        Load level constraints from layer.txt.

        Format: "node_id:level node_id:level ..."
        Example: "0:0 1:1 2:2 3:3 4:4"

        Args:
            sample_path: Path to the sample folder

        Returns:
            Dictionary mapping node names to levels
        """
        layer_path = Path(sample_path) / 'layer.txt'

        if not layer_path.exists():
            logger.warning(f"layer.txt not found in: {sample_path}")
            return {}

        with open(layer_path, 'r', encoding='utf-8') as f:
            line = f.readline().strip()

        level_map = {}
        if line:
            parts = line.split()
            for part in parts:
                if ':' in part:
                    node_id, level = part.split(':')
                    level_map[str(node_id)] = int(level)

        logger.info(f"Loaded layer.txt: {len(level_map)} level assignments")
        return level_map

    def _load_ground_truth(self, sample_path: str) -> Dict[str, Any]:
        """
        Load ground truth files.

        Args:
            sample_path: Path to the sample folder

        Returns:
            Dictionary with ground truth data
        """
        gt = {}

        # Load edges.txt
        edges_path = Path(sample_path) / 'edges.txt'
        if edges_path.exists():
            gt['edges'] = self._load_edges_file(edges_path)
            logger.info(f"Loaded edges.txt: {len(gt['edges'])} edges")

        # Load subgraph.txt
        subgraph_path = Path(sample_path) / 'subgraph.txt'
        if subgraph_path.exists():
            gt['subgraph_edges'] = self._load_edges_file(subgraph_path)
            logger.info(f"Loaded subgraph.txt: {len(gt['subgraph_edges'])} edges")

        # Load root_cause.txt
        root_cause_path = Path(sample_path) / 'root_cause.txt'
        if root_cause_path.exists():
            gt['root_causes'], gt['symptoms'] = self._load_root_cause_file(root_cause_path)
            logger.info(f"Loaded root_cause.txt: {len(gt['root_causes'])} root causes, {len(gt['symptoms'])} symptoms")

        return gt if gt else None

    def _load_edges_file(self, edges_path: Path) -> List[List[str]]:
        """
        Load edges from edges.txt or subgraph.txt.

        Format: "src dst" per line (empty line at end)
        Example:
            0 1
            1 2
            1 3

        Args:
            edges_path: Path to the edges file

        Returns:
            List of [source, target] pairs
        """
        edges = []
        with open(edges_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    edges.append([str(parts[0]), str(parts[1])])
        return edges

    def _load_root_cause_file(self, rc_path: Path) -> Tuple[List[str], List[str]]:
        """
        Load root causes and symptoms from root_cause.txt.

        Format (single line):
            First item: Root cause ID
            Second item: Symptom/Performance metric ID
        Example:
            6 9
        Where: 6 = root cause, 9 = symptom (performance metric)

        Args:
            rc_path: Path to the root_cause.txt file

        Returns:
            Tuple of (root_causes, symptoms) as lists of strings
        """
        with open(rc_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()

        parts = content.split()
        root_causes = []
        symptoms = []

        if len(parts) >= 1:
            # First item is root cause
            root_causes = [str(parts[0])]

        if len(parts) >= 2:
            # Second item is symptom (performance metric)
            symptoms = [str(parts[1])]

        logger.info(f"Loaded root_cause.txt: root_cause={root_causes}, symptom={symptoms}")

        return root_causes, symptoms

    @staticmethod
    def list_samples(dataset_path: str) -> List[str]:
        """
        List all sample folders in a RADICE dataset.

        Args:
            dataset_path: Path to the RADICE dataset folder

        Returns:
            List of sample folder names sorted numerically
        """
        dataset_dir = Path(dataset_path)
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset folder not found: {dataset_path}")

        # Find all artificialResults_* folders
        samples = []
        for item in dataset_dir.iterdir():
            if item.is_dir() and item.name.startswith('artificialResults_'):
                samples.append(item.name)

        # Sort numerically by the number after artificialResults_
        def extract_number(name):
            try:
                return int(name.split('_')[1])
            except (IndexError, ValueError):
                return 0

        samples.sort(key=extract_number)
        return samples
