"""
RDT-based data encoding utilities for sdvaluation.

This module provides utilities for loading encoding configurations and
transforming datasets using RDT (Reversible Data Transforms).
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import yaml
from rdt import HyperTransformer
from rdt.transformers import (
    FloatFormatter,
    FrequencyEncoder,
    LabelEncoder,
    OneHotEncoder,
    OrderedUniformEncoder,
    UniformEncoder,
    UnixTimestampEncoder,
)

logger = logging.getLogger(__name__)

# Registry mapping transformer names to classes
TRANSFORMER_REGISTRY = {
    "UniformEncoder": UniformEncoder,
    "OrderedUniformEncoder": OrderedUniformEncoder,
    "LabelEncoder": LabelEncoder,
    "OneHotEncoder": OneHotEncoder,
    "FrequencyEncoder": FrequencyEncoder,
    "UnixTimestampEncoder": UnixTimestampEncoder,
    "FloatFormatter": FloatFormatter,
}


def load_encoding_config(config_path: Path) -> Dict[str, Any]:
    """
    Load and validate encoding configuration from YAML file.

    The configuration file should have the following structure:
    ```yaml
    sdtypes:
      column1: numerical
      column2: categorical
      ...

    transformers:
      column1:
        type: UniformEncoder
        params:
          param1: value1
      column2:
        type: LabelEncoder
      ...
    ```

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Dictionary containing:
            - sdtypes: Mapping of column names to data types
            - transformers: Mapping of column names to instantiated transformers
            - config_path: Original config file path

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config structure is invalid
        KeyError: If transformer type is not in registry
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load YAML
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Validate structure
    if not isinstance(config, dict):
        raise ValueError("Config must be a dictionary")

    if "sdtypes" not in config:
        raise ValueError("Config must contain 'sdtypes' key")

    if "transformers" not in config:
        raise ValueError("Config must contain 'transformers' key")

    sdtypes = config["sdtypes"]
    transformer_specs = config["transformers"]

    if not isinstance(sdtypes, dict):
        raise ValueError("'sdtypes' must be a dictionary")

    if not isinstance(transformer_specs, dict):
        raise ValueError("'transformers' must be a dictionary")

    # Instantiate transformers
    transformers = {}
    for column, spec in transformer_specs.items():
        if not isinstance(spec, dict):
            raise ValueError(
                f"Transformer spec for '{column}' must be a dictionary"
            )

        transformer_type = spec.get("type")
        if not transformer_type:
            raise ValueError(
                f"Transformer spec for '{column}' must include 'type'"
            )

        if transformer_type not in TRANSFORMER_REGISTRY:
            raise KeyError(
                f"Unknown transformer type '{transformer_type}'. "
                f"Available: {list(TRANSFORMER_REGISTRY.keys())}"
            )

        # Get transformer class
        transformer_class = TRANSFORMER_REGISTRY[transformer_type]

        # Get parameters (if any)
        params = spec.get("params", {})
        if not isinstance(params, dict):
            raise ValueError(
                f"Transformer params for '{column}' must be a dictionary"
            )

        # Instantiate transformer
        try:
            transformer = transformer_class(**params)
            transformers[column] = transformer
            logger.debug(
                f"Instantiated {transformer_type} for column '{column}' "
                f"with params: {params}"
            )
        except Exception as e:
            raise ValueError(
                f"Failed to instantiate {transformer_type} for '{column}': {e}"
            )

    logger.info(
        f"Loaded encoding config from {config_path}: "
        f"{len(sdtypes)} columns, {len(transformers)} transformers"
    )

    return {
        "sdtypes": sdtypes,
        "transformers": transformers,
        "config_path": config_path,
    }


class RDTDatasetEncoder:
    """
    Dataset encoder using RDT HyperTransformer.

    Handles fitting and transforming datasets according to a configuration
    that specifies column types (sdtypes) and transformers.

    Attributes:
        config: Configuration dictionary with sdtypes and transformers
        hypertransformer: RDT HyperTransformer instance
        _is_fitted: Whether the encoder has been fitted

    Example:
        ```python
        config = load_encoding_config("encoding_config.yaml")
        encoder = RDTDatasetEncoder(config)
        encoder.fit(training_data)
        encoded_data = encoder.transform(training_data)
        ```
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the encoder with a configuration.

        Args:
            config: Configuration dictionary from load_encoding_config()
                   Must contain 'sdtypes' and 'transformers' keys.

        Raises:
            ValueError: If config is missing required keys
        """
        if "sdtypes" not in config:
            raise ValueError("Config must contain 'sdtypes' key")
        if "transformers" not in config:
            raise ValueError("Config must contain 'transformers' key")

        self.config = config
        self.hypertransformer: Optional[HyperTransformer] = None
        self._is_fitted = False

        logger.info(
            f"Initialized RDTDatasetEncoder with {len(config['sdtypes'])} columns"
        )

    def fit(self, training_data: pd.DataFrame) -> None:
        """
        Fit the encoder on training data.

        Creates and configures a HyperTransformer with the specified
        sdtypes and transformers, then fits it on the training data.

        Args:
            training_data: DataFrame to fit the encoder on

        Raises:
            ValueError: If training data is missing required columns
        """
        # Validate that all configured columns are present
        configured_columns = set(self.config["sdtypes"].keys())
        data_columns = set(training_data.columns)

        missing_columns = configured_columns - data_columns
        if missing_columns:
            raise ValueError(
                f"Training data missing configured columns: {missing_columns}"
            )

        # Create HyperTransformer
        self.hypertransformer = HyperTransformer()

        # Configure sdtypes
        self.hypertransformer.set_config(
            config={
                "sdtypes": self.config["sdtypes"],
                "transformers": self.config["transformers"],
            }
        )

        logger.info(f"Fitting encoder on {len(training_data)} samples...")

        # Fit on training data
        self.hypertransformer.fit(training_data)

        self._is_fitted = True
        logger.info("Encoder fitting complete")

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using the fitted encoder.

        Args:
            data: DataFrame to transform

        Returns:
            Transformed DataFrame with encoded features

        Raises:
            RuntimeError: If encoder hasn't been fitted yet
            ValueError: If data is missing required columns
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Encoder must be fitted before transforming. Call fit() first."
            )

        # Validate columns
        configured_columns = set(self.config["sdtypes"].keys())
        data_columns = set(data.columns)

        missing_columns = configured_columns - data_columns
        if missing_columns:
            raise ValueError(
                f"Data missing configured columns: {missing_columns}"
            )

        logger.debug(f"Transforming {len(data)} samples...")

        # Transform
        transformed = self.hypertransformer.transform(data)

        logger.debug(
            f"Transformation complete: {len(data.columns)} -> "
            f"{len(transformed.columns)} columns"
        )

        return transformed

    def reverse_transform(self, encoded_data: pd.DataFrame) -> pd.DataFrame:
        """
        Reverse transform encoded data back to original format.

        Args:
            encoded_data: Encoded DataFrame to reverse transform

        Returns:
            DataFrame in original format

        Raises:
            RuntimeError: If encoder hasn't been fitted yet
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Encoder must be fitted before reverse transforming. "
                "Call fit() first."
            )

        logger.debug(f"Reverse transforming {len(encoded_data)} samples...")

        # Reverse transform
        decoded = self.hypertransformer.reverse_transform(encoded_data)

        logger.debug(
            f"Reverse transformation complete: {len(encoded_data.columns)} -> "
            f"{len(decoded.columns)} columns"
        )

        return decoded

    @property
    def is_fitted(self) -> bool:
        """Check if the encoder has been fitted."""
        return self._is_fitted

    def get_feature_names(self) -> list:
        """
        Get the names of transformed features.

        Returns:
            List of feature names after transformation

        Raises:
            RuntimeError: If encoder hasn't been fitted yet
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Encoder must be fitted before getting feature names. "
                "Call fit() first."
            )

        # Get output columns from the hypertransformer
        return list(self.hypertransformer.get_config()["sdtypes"].keys())
