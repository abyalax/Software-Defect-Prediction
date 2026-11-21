"""Verify the tokenized model inputs that the data pipeline wrote for one split scenario."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from software_defects.data.defectors.split_generator import (
    GROUP_KFOLD_N_FOLDS,
    GROUP_KFOLD_SCENARIOS,
    DefectorsSplitGenerator,
    SplitScenario,
)
from software_defects.data.defectors.tokenizer import DefectorsTokenizer
from software_defects.reporting.reporter import ReportWritter
from software_defects.utils import get_device

if TYPE_CHECKING:
    from software_defects.config import DatasetConfig

SPLITS = ("train", "val", "test")
CHECK_COLUMNS = ["over_limit", "special_token_errors", "path_not_intact", "token_count_mismatch"]


class TokenizationVerifier:
    """Re-encode the processed model inputs and check them against the preprocessing contract.

    Per scenario, model and split it reports the share of truncated rows and verifies that:

    * no input is longer than the model limit (special tokens included),
    * every input starts and ends with the model's special tokens,
    * the file path is the untouched first line of every input (unless it was shortened on purpose),
    * the stored ``token_count`` equals the re-encoded length.
    """

    def __init__(
        self,
        config: DatasetConfig,
        input_dir: Path,
        tokenizers: dict[str, DefectorsTokenizer] | None = None,
    ) -> None:
        """Initialize the verifier.

        Args:
            config: Dataset configuration.
            input_dir: Directory containing the processed parquet files from the pipeline.
            tokenizers: Tokenizer per model key; defaults to the models in ``config.tokenizer_models``.
        """
        self._config = config
        self._input_dir = Path(input_dir)
        self._tokenizers = tokenizers or {
            key: DefectorsTokenizer(config, model_name=name, max_length=config.tokenizer_max_length)
            for key, name in config.tokenizer_models.items()
        }
        self._reporter = ReportWritter(
            config=config,
            domain_data="defectors",
            pipeline="data_pipeline",
            class_name="TokenizationVerifier",
        )

    def run(self, scenario: SplitScenario, fold: int | None = None) -> pd.DataFrame:
        """Verify one scenario (every fold for Group K-Fold unless ``fold`` is given) and save the report.

        Args:
            scenario: Split scenario to verify.
            fold: Fold index, only used by the Group K-Fold scenarios.

        Returns:
            One row per (fold, model, split) with the verification counts.

        Raises:
            FileNotFoundError: If the processed files of the scenario do not exist.
        """
        self._reporter.display_section(f"Tokenization verification: {scenario.value}")

        # Detect and report device
        device = get_device()
        self._reporter.display_text(f"Device: {device}")
        self._reporter.display_text()

        rows: list[dict] = []
        for fold_index in self._folds(scenario, fold):
            scenario_dir = DefectorsSplitGenerator.scenario_dir(self._input_dir, scenario, fold_index)
            for model_key, tokenizer in self._tokenizers.items():
                for split in SPLITS:
                    path = scenario_dir / model_key / f"{scenario.value}_{split}.parquet"
                    if not path.exists():
                        raise FileNotFoundError(
                            f"Tokenized file not found: {path}. Run `run-pipeline --scenario {scenario.value}` first "
                            "(without --no-tokenize)."
                        )
                    row = self.verify_frame(pd.read_parquet(path), tokenizer)
                    rows.append({"fold": fold_index, "model": model_key, "split": split, **row})

        report = pd.DataFrame(rows)
        self._reporter.display_table(report)
        failures = int(report[CHECK_COLUMNS].sum().sum())
        if failures:
            self._reporter.display_warning(f"{failures} verification failure(s); see the table above")
        else:
            self._reporter.display_text("All checks passed.")
        self._reporter.save_csv(report, f"tokenization_{scenario.value}.csv")
        self._reporter.update_summary(
            {"device": device, "scenario": scenario.value, "failures": failures, "rows": rows}
        )
        self._reporter.save_reports(f"tokenization_{scenario.value}.txt", f"tokenization_{scenario.value}.json")
        return report

    def verify_frame(self, df: pd.DataFrame, tokenizer: DefectorsTokenizer) -> dict:
        """Verify one tokenized split frame.

        Args:
            df: Processed frame with ``model_input``, ``filepath``, ``token_count`` and the preprocessing columns.
            tokenizer: Tokenizer of the model the frame was built for.

        Returns:
            Dictionary with the row count, truncation shares and the number of violations per check.
        """
        hf = tokenizer.load_tokenizer()
        max_length = tokenizer.max_length
        head_ids, tail_ids = self._special_layout(hf)
        texts = df["model_input"].tolist()
        paths = df["filepath"].astype(str).tolist()
        encoded = hf(texts, truncation=False, padding=False, return_attention_mask=False, verbose=False)["input_ids"]
        path_ids = hf(paths, add_special_tokens=False, return_attention_mask=False, verbose=False)["input_ids"]
        check_path = self._config.diff_input.include_path

        over_limit = special_errors = path_errors = count_errors = 0
        for ids, p_ids, stored, shortened, text, path in zip(
            encoded, path_ids, df["token_count"], df["path_shortened"], texts, paths, strict=True
        ):
            over_limit += len(ids) > max_length
            count_errors += len(ids) != int(stored)
            special_errors += not self._has_special_tokens(ids, head_ids, tail_ids)
            if check_path and not shortened:
                first = len(head_ids)
                intact = text.startswith(path + "\n") and ids[first : first + len(p_ids)] == p_ids
                path_errors += not intact

        def pct(column: str) -> float:
            return float(df[column].mean() * 100) if column in df.columns and len(df) else 0.0

        total_changed = int(df["changed_total"].sum())
        return {
            "rows": len(df),
            "truncated_pct": pct("truncated"),
            "pre_cut_pct": pct("pre_cut"),
            "fallback_pct": pct("fallback_truncation"),
            "path_shortened_pct": pct("path_shortened"),
            "changed_kept_pct": float(df["changed_kept"].sum() / total_changed * 100) if total_changed else 100.0,
            "mean_tokens": float(df["token_count"].mean()) if len(df) else 0.0,
            "over_limit": int(over_limit),
            "special_token_errors": int(special_errors),
            "path_not_intact": int(path_errors),
            "token_count_mismatch": int(count_errors),
        }

    @staticmethod
    def _special_layout(hf: object) -> tuple[list[int], list[int]]:
        """Return the special token ids the model puts before and after the content of a sequence."""
        probe = "x = 1"
        with_special = hf(probe, return_attention_mask=False, verbose=False)["input_ids"]
        content = hf(probe, add_special_tokens=False, return_attention_mask=False, verbose=False)["input_ids"]
        for head in range(len(with_special) - len(content) + 1):
            if with_special[head : head + len(content)] == content:
                return with_special[:head], with_special[head + len(content) :]
        raise ValueError("Cannot locate the content tokens inside the encoded probe")

    @staticmethod
    def _has_special_tokens(ids: list[int], head_ids: list[int], tail_ids: list[int]) -> bool:
        """Check that ``ids`` begins and ends with the model's special tokens."""
        starts = ids[: len(head_ids)] == head_ids
        ends = not tail_ids or ids[-len(tail_ids) :] == tail_ids
        return starts and ends

    @staticmethod
    def _folds(scenario: SplitScenario, fold: int | None) -> list[int]:
        if scenario in GROUP_KFOLD_SCENARIOS:
            return [fold] if fold is not None else list(range(GROUP_KFOLD_N_FOLDS))
        if fold is not None:
            raise ValueError("--fold is only valid for the Group_KFold scenarios")
        return [0]
