"""Tests for hunk-aware truncation, the 'path + diff' input builder and the tokenization verifier."""

from __future__ import annotations

import random
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
from transformers import AutoTokenizer

from software_defects.config import DatasetConfig, DiffInputConfig
from software_defects.data.defectors.diff_input_builder import DiffInputBuilder
from software_defects.data.defectors.diff_truncator import HunkAwareTruncator
from software_defects.data.defectors.pipeline import DefectorsDataPipeline
from software_defects.data.defectors.split_generator import SplitScenario
from software_defects.data.defectors.tokenization_verifier import CHECK_COLUMNS, TokenizationVerifier
from software_defects.data.defectors.tokenizer import DefectorsTokenizer
from tests.conftest import write_raw_jit


def char_truncator(**kwargs: float) -> HunkAwareTruncator:
    return HunkAwareTruncator(lambda lines: [len(line) for line in lines], **kwargs)


def cost(text: str) -> int:
    """Cost of a rendered diff under the character truncator (line length plus one per line break)."""
    return sum(len(line) + 1 for line in text.split("\n"))


def make_diff(hunk_sizes: list[int], context: int = 2) -> str:
    lines: list[str] = []
    for hunk, size in enumerate(hunk_sizes):
        lines.append(f"@@ -{hunk * 10},{size} +{hunk * 10},{size} @@ def fn{hunk}():")
        lines.extend(f" context{hunk}_{i} = {i}" for i in range(context))
        lines.extend(f"+added{hunk}_{i} = {i}" if i % 2 == 0 else f"-removed{hunk}_{i} = {i}" for i in range(size))
        lines.extend(f" tail{hunk}_{i} = {i}" for i in range(context))
    return "\n".join(lines) + "\n"


class TestHunkAwareTruncator:
    def test_diff_within_budget_is_unchanged(self) -> None:
        diff = make_diff([3, 2])
        result = char_truncator().truncate(diff, budget=10_000)

        assert result.text == diff
        assert not result.truncated
        assert (result.hunks_total, result.hunks_kept) == (2, 2)
        assert (result.changed_total, result.changed_kept) == (5, 5)

    def test_result_fits_budget_and_keeps_every_hunk_header(self) -> None:
        diff = make_diff([30, 30, 30])
        result = char_truncator().truncate(diff, budget=600)

        assert result.truncated
        assert cost(result.text) <= 600
        assert result.hunks_kept == 3
        for hunk in range(3):
            assert f"@@ -{hunk * 10},30 +{hunk * 10},30 @@ def fn{hunk}():" in result.text

    def test_changed_lines_win_over_context(self) -> None:
        diff = make_diff([6], context=10)
        result = char_truncator().truncate(diff, budget=cost(diff) // 2)

        kept = result.text.split("\n")
        assert result.changed_kept == result.changed_total == 6
        assert any(line.startswith("+added0_0") for line in kept)
        assert sum(line.startswith(" ") for line in kept) < 20

    def test_small_hunk_is_not_starved_by_a_large_one(self) -> None:
        diff = make_diff([2, 200], context=0)
        result = char_truncator().truncate(diff, budget=500)

        small_hunk_lines = [line for line in result.text.split("\n") if "added0_" in line or "removed0_" in line]
        assert len(small_hunk_lines) == 2
        assert result.changed_kept < result.changed_total

    def test_order_of_lines_is_preserved(self) -> None:
        diff = make_diff([20, 20])
        result = char_truncator().truncate(diff, budget=500)

        original = diff.split("\n")
        positions = [original.index(line) for line in result.text.split("\n")]
        assert positions == sorted(positions)

    def test_many_hunks_respect_the_header_share(self) -> None:
        diff = make_diff([1] * 200, context=0)
        result = char_truncator(header_share=0.5).truncate(diff, budget=800)

        header_cost = sum(len(line) + 1 for line in result.text.split("\n") if line.startswith("@@"))
        assert cost(result.text) <= 800
        assert header_cost <= 400
        assert result.hunks_kept < result.hunks_total

    def test_one_huge_line_is_clipped_instead_of_dropped(self) -> None:
        diff = "@@ -1 +1 @@\n+" + "x" * 5000 + "\n"
        result = char_truncator().truncate(diff, budget=200)

        assert result.changed_kept == 1
        assert cost(result.text) <= 200

    def test_text_before_the_first_hunk_is_handled(self) -> None:
        diff = "--- a/x.py\n+++ b/x.py\n" + make_diff([4])
        result = char_truncator().truncate(diff, budget=120)

        assert cost(result.text) <= 120
        assert "@@ -0,4 +0,4 @@ def fn0():" in result.text

    def test_empty_diff(self) -> None:
        result = char_truncator().truncate("", budget=10)
        assert (result.text, result.truncated, result.hunks_total) == ("", False, 0)

    def test_random_diffs_never_exceed_the_budget(self) -> None:
        rng = random.Random(7)
        for _ in range(200):
            diff = make_diff([rng.randint(1, 25) for _ in range(rng.randint(1, 8))], context=rng.randint(0, 4))
            budget = rng.randint(30, 900)
            result = char_truncator().truncate(diff, budget=budget)
            assert cost(result.text) <= budget or not result.text.strip(), (budget, result.text)

    def test_count_structure(self) -> None:
        assert HunkAwareTruncator.count_structure(make_diff([3, 4])) == (2, 7)

    def test_invalid_header_share_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="header_share"):
            char_truncator(header_share=0)


class TestDiffInputBuilder:
    @staticmethod
    def builder(tokenizer, max_length: int = 128, **config: object) -> DiffInputBuilder:
        wrapper = DefectorsTokenizer(DatasetConfig(Path("."), "defectors"), max_length=max_length)
        wrapper._tokenizer = tokenizer
        return DiffInputBuilder(
            count_tokens=wrapper.count_tokens,
            special_tokens=tokenizer.num_special_tokens_to_add(pair=False),
            max_length=max_length,
            config=DiffInputConfig(**config),
        )

    def test_short_diff_keeps_path_and_whole_diff(self, roberta_like_tokenizer) -> None:
        diff = make_diff([2])
        [item] = self.builder(roberta_like_tokenizer, max_length=512).build(["pkg/a.py"], [diff])

        assert item.text == f"pkg/a.py\n{diff}"
        assert not item.truncated
        assert (item.hunks_total, item.hunks_kept, item.changed_total, item.changed_kept) == (1, 1, 2, 2)

    def test_long_diff_fits_the_limit_and_path_stays_first(self, roberta_like_tokenizer) -> None:
        builder = self.builder(roberta_like_tokenizer, max_length=128)
        diffs = [make_diff([40, 40, 40]), make_diff([300]), make_diff([5] * 30)]
        paths = ["airflow/models/dag.py", "a/very/deep/dir/tree/file.py", "x.py"]

        for path, diff, item in zip(paths, diffs, builder.build(paths, diffs), strict=True):
            ids = roberta_like_tokenizer(item.text)["input_ids"]
            assert len(ids) <= 128
            assert item.truncated
            assert not item.fallback
            assert item.text.startswith(path + "\n")
            assert item.full_tokens > 128 - 2
            assert item.changed_kept < item.changed_total
            assert diff.startswith("@@")

    def test_extreme_diff_is_pre_cut_and_still_fits(self, roberta_like_tokenizer) -> None:
        builder = self.builder(roberta_like_tokenizer, max_length=128, pre_cut_chars=2000)
        diff = make_diff([2000])
        assert len(diff) > 20_000

        [item] = builder.build(["big.py"], [diff])

        assert item.pre_cut
        assert item.truncated
        assert len(roberta_like_tokenizer(item.text)["input_ids"]) <= 128
        assert item.text.startswith("big.py\n")
        assert item.changed_total == 2000

    def test_overlong_path_is_shortened_from_the_left(self, roberta_like_tokenizer) -> None:
        builder = self.builder(roberta_like_tokenizer, max_length=128, path_max_tokens=16)
        long_path = "/".join(["some_very_long_directory_name"] * 8) + "/final_module.py"

        [item] = builder.build([long_path], [make_diff([40])])

        assert item.path_shortened
        assert item.text.startswith(".../")
        assert item.text.split("\n")[0].endswith("final_module.py")
        assert len(roberta_like_tokenizer(item.text)["input_ids"]) <= 128

    def test_path_can_be_left_out_for_the_ablation(self, roberta_like_tokenizer) -> None:
        builder = self.builder(roberta_like_tokenizer, max_length=512, include_path=False)
        diff = make_diff([2])
        [item] = builder.build(["pkg/a.py"], [diff])
        assert item.text == diff

    def test_mismatched_inputs_are_rejected(self, roberta_like_tokenizer) -> None:
        with pytest.raises(ValueError, match="same length"):
            self.builder(roberta_like_tokenizer).build(["a.py"], [])

    def test_random_inputs_never_exceed_the_limit(self, roberta_like_tokenizer) -> None:
        rng = random.Random(3)
        builder = self.builder(roberta_like_tokenizer, max_length=96, pre_cut_chars=4000)
        diffs = [
            make_diff([rng.randint(1, 60) for _ in range(rng.randint(1, 6))], context=rng.randint(0, 3))
            for _ in range(60)
        ]
        paths = [f"pkg{i}/mod{i % 5}.py" for i in range(len(diffs))]

        for item in builder.build(paths, diffs):
            assert len(roberta_like_tokenizer(item.text)["input_ids"]) <= 96
            assert not item.fallback


@pytest.fixture
def tokenized_pipeline(
    config: DatasetConfig, project_root: Path, synthetic_jit: pd.DataFrame, monkeypatch, roberta_like_tokenizer
):
    """Run the pipeline with the offline tokenizer for both models and a tight limit so truncation happens."""
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: roberta_like_tokenizer)
    config = replace(config, tokenizer_max_length=96)
    write_raw_jit(project_root, synthetic_jit)
    out = project_root / "processed"
    data = DefectorsDataPipeline(config).run_pipeline(SplitScenario.WITHIN_DOMAIN_ML_DL, output_dir=out, tokenize=True)
    return config, out, data


class TestTokenizedPipeline:
    def test_every_split_and_model_respects_the_limit(self, tokenized_pipeline) -> None:
        _, _, data = tokenized_pipeline

        assert set(data) == {"codebert", "codet5"}
        for model_frames in data.values():
            for frame in model_frames.values():
                assert frame["token_count"].max() <= 96
                pairs = zip(frame["model_input"], frame["filepath"], strict=True)
                assert all(text.startswith(path + "\n") for text, path in pairs)
                assert frame["truncated"].any()
                assert not frame["fallback_truncation"].any()

    def test_truncated_flag_is_no_longer_always_false(self, tokenized_pipeline) -> None:
        _, _, data = tokenized_pipeline
        frame = data["codebert"]["train"]
        assert frame["truncated"].sum() == (frame["full_tokens"] + 2 > 96).sum()

    def test_verifier_passes_on_pipeline_output(self, tokenized_pipeline) -> None:
        config, out, _ = tokenized_pipeline

        report = TokenizationVerifier(config, out).run(SplitScenario.WITHIN_DOMAIN_ML_DL)

        assert len(report) == 2 * 3
        assert report[CHECK_COLUMNS].to_numpy().sum() == 0
        assert (report["truncated_pct"] > 0).all()

    def test_verifier_detects_a_lost_path(self, tokenized_pipeline, roberta_like_tokenizer) -> None:
        config, out, data = tokenized_pipeline
        broken = data["codebert"]["test"].copy()
        broken["model_input"] = broken["model_input"].str.split("\n", n=1).str[1]

        tokenizer = DefectorsTokenizer(config, max_length=96)
        tokenizer._tokenizer = roberta_like_tokenizer
        result = TokenizationVerifier(config, out).verify_frame(broken, tokenizer)

        assert result["path_not_intact"] == len(broken)
        assert result["token_count_mismatch"] > 0

    def test_verifier_reports_missing_files(self, config: DatasetConfig, project_root: Path) -> None:
        with pytest.raises(FileNotFoundError, match="run-pipeline"):
            TokenizationVerifier(config, project_root / "processed").run(SplitScenario.ML_TO_NON_ML)


def _real_tokenizers_cached() -> bool:
    try:
        for name in ("microsoft/codebert-base", "Salesforce/codet5-small"):
            AutoTokenizer.from_pretrained(name, local_files_only=True)
    except Exception:
        return False
    return True


@pytest.mark.skipif(not _real_tokenizers_cached(), reason="CodeBERT/CodeT5 tokenizers are not in the local cache")
@pytest.mark.parametrize("name", ["microsoft/codebert-base", "Salesforce/codet5-small"])
def test_real_tokenizers_keep_path_and_limit(name: str) -> None:
    tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
    wrapper = DefectorsTokenizer(DatasetConfig(Path("."), "defectors"), model_name=name, max_length=512)
    wrapper._tokenizer = tokenizer
    diffs = [make_diff([400, 400]), make_diff([3]), make_diff([2000])]
    paths = ["airflow/providers/google/cloud/hooks/bigquery.py", "a.py", "tests/test_x.py"]

    items = wrapper.input_builder().build(paths, diffs)

    for path, item in zip(paths, items, strict=True):
        ids = tokenizer(item.text)["input_ids"]
        path_ids = tokenizer(path, add_special_tokens=False)["input_ids"]
        head = len(tokenizer("")["input_ids"]) - len(tokenizer("", add_special_tokens=False)["input_ids"]) - 1
        head = max(head, 0)
        assert len(ids) <= 512
        assert not item.fallback
        assert ids[head : head + len(path_ids)] == path_ids
