"""
Configuration Manager for causal discovery system.

This module provides a centralized configuration management system that:
1. Loads YAML configuration files
2. Supports variable substitution (e.g., {dataset}, {case})
3. Provides convenient access to nested configuration values
4. Validates configuration against schemas
"""

import os
import re
import yaml
from typing import Any, Dict, Optional, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Configuration manager for loading and accessing YAML configuration files.

    Supports variable substitution using {placeholder} syntax that can be
    replaced with values from other parts of the configuration.

    All path fields are automatically resolved to absolute paths after loading.
    """

    # Path fields that need to be resolved (dot-notation paths)
    PATH_FIELDS = {
        'data_loading.path': 'input',
        'data_loading.dataset_path': 'input',
        'knowledge.constraints_path': 'input',
        'evaluation.ground_truth_path': 'input',
        'preprocessing.output_path': 'output',
        'output.base_dir': 'output',
        'algorithm.output_dir': 'output',
        'evaluation.output_path': 'output',
        'logging.file': 'output',
        # Batch processing paths
        'batch.output_base_dir': 'output',
        'batch.rca_output_base_dir': 'output',
        'batch.processed_output_dir': 'output',
    }

    def __init__(self, config_path: Optional[str] = None, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize the configuration manager.

        Args:
            config_path: Path to the YAML configuration file
            config_dict: Optional dictionary to use as configuration (overrides file loading)
        """
        self._config: Dict[str, Any] = {}
        self._config_path: Optional[str] = None
        self._path_resolver = None
        self._original_config_str: Optional[str] = None  # Store original template

        if config_dict is not None:
            self._config = config_dict
        elif config_path is not None:
            self.load(config_path)

    class PathResolver:
        """
        Resolves relative paths to absolute paths based on project root.

        The project root is the parent directory of 'src' (e.g., ZH_CausalDiscovery-zpz/).
        All relative paths in configuration are resolved relative to this root directory.
        """

        def __init__(self, reference_file: str):
            """
            Initialize the path resolver.

            Args:
                reference_file: Absolute path to a reference file (typically __file__)
            """
            self._project_root = self._find_project_root(reference_file)

        def _find_project_root(self, reference_file: str) -> Path:
            """
            Find the project root directory.

            The project root is the parent directory containing 'src'.

            Args:
                reference_file: Absolute path to a reference file

            Returns:
                Path to the project root directory
            """
            ref_path = Path(reference_file).resolve()

            # Check if reference file is directly in src
            if ref_path.name == 'src':
                return ref_path.parent.resolve()

            # Navigate up from the reference file to find src
            current = ref_path
            while current != current.parent:
                if current.name == 'src':
                    return current.parent.resolve()
                current = current.parent

            # Fallback: assume project root is two levels up from reference file
            # (handles case where reference is in src/utils/)
            return ref_path.parent.parent.resolve()

        @property
        def project_root(self) -> Path:
            """Get the project root directory."""
            return self._project_root

        def resolve_path(self, path: str, context: Optional[Dict[str, str]] = None) -> str:
            """
            Resolve a path to an absolute path.

            First applies placeholder substitution, then converts relative
            paths to absolute paths relative to the project root.

            Args:
                path: Path string (may contain placeholders like {dataset_name})
                context: Optional context dict for placeholder substitution

            Returns:
                Resolved absolute path
            """
            resolved = path

            # Step 1: Apply placeholder substitution
            if context:
                for key, value in context.items():
                    placeholder = f"{{{key}}}"
                    resolved = resolved.replace(placeholder, str(value))

            # Step 2: Convert relative path to absolute
            if not Path(resolved).is_absolute():
                resolved = str(self._project_root / resolved)

            return resolved

    def load(self, config_path: str) -> 'ConfigManager':
        """
        Load configuration from a YAML file.

        Args:
            config_path: Path to the YAML configuration file

        Returns:
            Self for method chaining

        Raises:
            FileNotFoundError: If the configuration file doesn't exist
            yaml.YAMLError: If the YAML file is malformed
        """
        # Initialize path resolver (if not already initialized)
        if self._path_resolver is None:
            self._path_resolver = self.PathResolver(__file__)

        # Resolve the config path properly
        resolved_path = self._resolve_config_path(config_path)
        self._config_path = resolved_path

        if not os.path.exists(resolved_path):
            raise FileNotFoundError(f"Configuration file not found: {resolved_path}")

        with open(resolved_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f) or {}

        # Store original template for re-substitution
        self._original_config_str = yaml.dump(self._config)

        # Apply variable substitution
        self._apply_substitution()

        # Resolve all path fields to absolute paths
        self._resolve_all_paths()

        logger.info(f"Loaded configuration from: {resolved_path}")
        return self

    def _resolve_config_path(self, config_path: str) -> str:
        """
        Resolve configuration file path to absolute path.

        All relative paths are resolved relative to the project root directory.

        Args:
            config_path: Input path (can be relative or absolute)

        Returns:
            Resolved absolute path
        """
        # Initialize path resolver if needed
        if self._path_resolver is None:
            self._path_resolver = self.PathResolver(__file__)

        # If already absolute, return as is
        if os.path.isabs(config_path):
            return config_path

        # Resolve relative path against project root
        resolved_path = str(self._path_resolver.project_root / config_path)

        return resolved_path

    def _apply_substitution(self) -> None:
        """
        Apply variable substitution throughout the configuration.

        Replaces {placeholder} patterns with values from the 'context' section
        of the configuration.

        Always starts from the original template to ensure placeholders
        are correctly replaced with current context values.
        """
        context = self._config.get('context', {})
        if not context:
            return

        # Start from original template (before any substitution)
        if self._original_config_str is None:
            # Fallback: if no original template, use current config
            config_str = yaml.dump(self._config)
        else:
            config_str = self._original_config_str

        # Define mapping from short placeholders to context keys
        placeholder_map = {
            'dataset': 'dataset_name',
            'case': 'case_name',
            'sample': 'sample_name'
        }

        # Replace {key} patterns with context values
        for placeholder, context_key in placeholder_map.items():
            if context_key in context:
                pattern = re.compile(r'\{' + placeholder + r'\}')
                config_str = pattern.sub(str(context[context_key]), config_str)

        # Also allow direct key matching (e.g., {dataset_name})
        for key, value in context.items():
            pattern = re.compile(r'\{' + key + r'\}')
            config_str = pattern.sub(str(value), config_str)

        self._config = yaml.safe_load(config_str)

        # Log for debugging
        logger.debug(f"Applied substitution with context: {context}")

    def _resolve_all_paths(self) -> None:
        """
        Resolve all configured path fields to absolute paths.

        This method:
        1. Re-applies variable substitution with current context values
        2. Resolves relative paths to absolute paths
        """
        if self._path_resolver is None:
            self._path_resolver = self.PathResolver(__file__)

        # First, re-apply substitution with current context values
        self._apply_substitution()

        # Get context for placeholder substitution
        context = self._config.get('context', {})

        for field_path, field_type in self.PATH_FIELDS.items():
            value = self.get(field_path)
            if value and isinstance(value, str):
                resolved = self._path_resolver.resolve_path(value, context)
                self.set(field_path, resolved)
                logger.info(f"Resolved path: {field_path} -> {resolved}")

        # Log preprocessing output_path specifically for debugging
        prep_output = self.get('preprocessing.output_path')
        if prep_output:
            logger.info(f"preprocessing.output_path after resolution: {prep_output}")

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value by dot-separated key path.

        Args:
            key_path: Dot-separated path to the configuration value
                     (e.g., 'algorithm.params.alpha')
            default: Default value if key is not found

        Returns:
            Configuration value or default

        Examples:
            >>> config.get('algorithm.params.alpha')
            0.05
            >>> config.get('context.dataset_name')
            'online_boutique'
        """
        keys = key_path.split('.')
        value = self._config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def set(self, key_path: str, value: Any) -> None:
        """
        Set a configuration value by dot-separated key path.

        Args:
            key_path: Dot-separated path to the configuration value
            value: Value to set
        """
        keys = key_path.split('.')
        config = self._config

        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        # Set the value
        config[keys[-1]] = value

    def get_section(self, section_name: str) -> Dict[str, Any]:
        """
        Get an entire configuration section.

        Args:
            section_name: Name of the section to retrieve

        Returns:
            Dictionary containing the section's configuration,
            or empty dict if section doesn't exist
        """
        return self._config.get(section_name, {})

    def to_dict(self) -> Dict[str, Any]:
        """
        Return the entire configuration as a dictionary.

        Returns:
            Complete configuration dictionary
        """
        return self._config.copy()

    @property
    def config_path(self) -> Optional[str]:
        """Path to the configuration file."""
        return self._config_path
