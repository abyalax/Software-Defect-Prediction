"""Dataset loader for Defectors dataset."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from software_defects.config import DatasetConfig


class DefectorsDatasetLoader:
    """Load and validate Defectors dataset files."""

    def __init__(self, config: DatasetConfig) -> None:
        """Initialize the dataset loader.

        Args:
            config: Dataset configuration containing paths and settings.
        """
        self._config = config

    def load_jit_train(self) -> pd.DataFrame:
        """Load JIT bug prediction training data.

        Returns:
            DataFrame containing training data.
        """
        path = self._config.raw_dir / "defectors" / "jit_bug_prediction_splits" / "random" / "train.parquet.gzip"
        return self._load_parquet(path)

    def load_jit_test(self) -> pd.DataFrame:
        """Load JIT bug prediction test data.

        Returns:
            DataFrame containing test data.
        """
        path = self._config.raw_dir / "defectors" / "jit_bug_prediction_splits" / "random" / "test.parquet.gzip"
        return self._load_parquet(path)

    def load_jit_val(self) -> pd.DataFrame:
        """Load JIT bug prediction validation data.

        Returns:
            DataFrame containing validation data.
        """
        path = self._config.raw_dir / "defectors" / "jit_bug_prediction_splits" / "random" / "val.parquet.gzip"
        return self._load_parquet(path)

    def load_line_train(self) -> pd.DataFrame:
        """Load line-level bug prediction training data.

        Returns:
            DataFrame containing training data.
        """
        path = self._config.raw_dir / "defectors" / "line_bug_prediction_splits" / "random" / "train.parquet.gzip"
        return self._load_parquet(path)

    def load_line_test(self) -> pd.DataFrame:
        """Load line-level bug prediction test data.

        Returns:
            DataFrame containing test data.
        """
        path = self._config.raw_dir / "defectors" / "line_bug_prediction_splits" / "random" / "test.parquet.gzip"
        return self._load_parquet(path)

    def load_line_val(self) -> pd.DataFrame:
        """Load line-level bug prediction validation data.

        Returns:
            DataFrame containing validation data.
        """
        path = self._config.raw_dir / "defectors" / "line_bug_prediction_splits" / "random" / "val.parquet.gzip"
        return self._load_parquet(path)

    def load_all_jit(self) -> pd.DataFrame:
        """Load all JIT splits combined.

        Returns:
            DataFrame containing all JIT data.
        """
        dfs = [self.load_jit_train(), self.load_jit_test(), self.load_jit_val()]
        return pd.concat(dfs, ignore_index=True)

    def load_all_line(self) -> pd.DataFrame:
        """Load all line-level splits combined.

        Returns:
            DataFrame containing all line-level data.
        """
        dfs = [self.load_line_train(), self.load_line_test(), self.load_line_val()]
        return pd.concat(dfs, ignore_index=True)

    def _load_parquet(self, path: Path) -> pd.DataFrame:
        """Load a parquet file with optional row limit.

        Args:
            path: Path to the parquet file.

        Returns:
            DataFrame containing the loaded data.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        df = pd.read_parquet(path)

        if self._config.row_limit is not None:
            df = df.head(self._config.row_limit)

        return df
