"""Smoke tests of the Transformer classifiers and the shared training loop (tiny random models, no download)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from transformers import AutoTokenizer, RobertaConfig, T5Config  # noqa: E402

from software_defects.cli.parser import ParserArgs  # noqa: E402
from software_defects.config import DatasetConfig  # noqa: E402
from software_defects.data.defectors.pipeline import DefectorsDataPipeline  # noqa: E402
from software_defects.data.defectors.split_generator import SplitScenario  # noqa: E402
from software_defects.data.defectors.tokenizer import DefectorsTokenizer  # noqa: E402
from software_defects.models.classifier_config import ClassifierConfig, TrainingConfig  # noqa: E402
from software_defects.models.classifier_trainer import ClassifierTrainer  # noqa: E402
from software_defects.models.codebert_classifier import CodeBertClassifier  # noqa: E402
from software_defects.models.codet5_small_classifier import CodeT5SmallClassifier  # noqa: E402
from software_defects.models.length_bucket_loader import LengthBucketLoader  # noqa: E402
from software_defects.models.training_experiment import TrainingExperiment  # noqa: E402
from tests.conftest import write_raw_jit  # noqa: E402

PAD, EOS, VOCAB = 1, 2, 400  # ids of the offline roberta-like tokenizer fixture


def tiny_roberta() -> RobertaConfig:
    return RobertaConfig(
        vocab_size=VOCAB, hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
        max_position_embeddings=130, pad_token_id=PAD, bos_token_id=0, eos_token_id=EOS,
    )  # fmt: skip


def tiny_t5() -> T5Config:
    return T5Config(
        vocab_size=VOCAB, d_model=32, d_kv=8, d_ff=64, num_layers=2, num_decoder_layers=2, num_heads=4,
        pad_token_id=PAD, eos_token_id=EOS, decoder_start_token_id=PAD,
    )  # fmt: skip


def build_codebert(config: ClassifierConfig) -> CodeBertClassifier:
    return CodeBertClassifier.from_backbone_config(config, tiny_roberta())


def build_codet5(config: ClassifierConfig) -> CodeT5SmallClassifier:
    return CodeT5SmallClassifier.from_backbone_config(config, tiny_t5())


def synthetic_samples(n: int = 48, seed: int = 0) -> tuple[list[list[int]], np.ndarray]:
    """Samples whose label is decided by a marker token, so a tiny model can pick it up."""
    rng = np.random.default_rng(seed)
    ids, labels = [], []
    for i in range(n):
        label = i % 2
        body = rng.integers(5, 100, size=int(rng.integers(4, 40))).tolist()
        body[0] = 150 if label else 151
        ids.append([0, *body, EOS])
        labels.append(label)
    return ids, np.array(labels)


def loaders(batch_size: int = 8) -> tuple[LengthBucketLoader, LengthBucketLoader, np.ndarray]:
    ids, labels = synthetic_samples()
    train = LengthBucketLoader(ids, labels, batch_size, PAD, shuffle=True, pool_factor=2, seed=1)
    val = LengthBucketLoader(ids, labels, 16, PAD, shuffle=False)
    return train, val, labels


class TestLengthBucketLoader:
    def test_batches_are_padded_to_their_own_longest_sample(self) -> None:
        train, _, _ = loaders()
        for batch, indices in train:
            lengths = batch["attention_mask"].sum(dim=1)
            assert batch["input_ids"].shape[1] == int(lengths.max())
            assert (batch["input_ids"][batch["attention_mask"] == 0] == PAD).all()
            assert len(indices) == len(batch["labels"])

    def test_every_sample_is_seen_exactly_once_per_epoch(self) -> None:
        train, val, labels = loaders()
        for loader in (train, val):
            seen = np.concatenate([indices for _, indices in loader])
            assert sorted(seen.tolist()) == list(range(len(labels)))

    def test_bucketing_wastes_less_padding_than_random_batches(self) -> None:
        ids, labels = synthetic_samples(400)

        def padded_tokens(bucketing: bool) -> int:
            loader = LengthBucketLoader(ids, labels, 8, PAD, True, bucketing, pool_factor=25, seed=3)
            return sum(batch["input_ids"].numel() for batch, _ in loader)

        assert padded_tokens(True) < padded_tokens(False)

    def test_epoch_changes_the_order_and_seed_makes_it_reproducible(self) -> None:
        train, _, _ = loaders()
        first = [b.tolist() for b in train.batch_indices()]
        assert first == [b.tolist() for b in train.batch_indices()]
        train.set_epoch(2)
        assert first != [b.tolist() for b in train.batch_indices()]


@pytest.mark.parametrize("build", [build_codebert, build_codet5], ids=["codebert", "codet5"])
class TestClassifiers:
    def test_logits_shape_and_padding_independence(self, build) -> None:
        torch.manual_seed(0)
        model = build(ClassifierConfig(pretrained_name="unused")).eval()
        ids, _ = synthetic_samples(4)
        short = torch.tensor([ids[0]])
        width = len(ids[0]) + 5
        padded = torch.full((1, width), PAD)
        padded[0, : len(ids[0])] = short[0]
        mask = (torch.arange(width) < len(ids[0])).long().unsqueeze(0)

        with torch.no_grad():
            alone = model(short, torch.ones_like(short))
            with_padding = model(padded, mask)

        assert alone.shape == (1, 2)
        assert torch.allclose(alone, with_padding, atol=1e-4)

    def test_gradient_checkpointing_runs_a_training_step(self, build) -> None:
        model = build(ClassifierConfig(pretrained_name="unused", gradient_checkpointing=True)).train()
        batch, _ = next(iter(loaders()[0]))
        loss = torch.nn.functional.cross_entropy(model(batch["input_ids"], batch["attention_mask"]), batch["labels"])
        loss.backward()
        assert all(p.grad is not None for p in model.out_proj.parameters())


def test_codet5_head_reads_the_last_eos_state() -> None:
    model = build_codet5(ClassifierConfig(pretrained_name="unused"))
    hidden = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
    input_ids = torch.tensor([[0, 7, EOS, PAD, PAD], [0, 7, 8, 9, EOS]])
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])

    picked = model._last_eos_state(input_ids, mask, hidden)

    assert torch.equal(picked[0], hidden[0, 2])
    assert torch.equal(picked[1], hidden[1, 4])


class TestTrainer:
    def trainer(self, tmp_path: Path, log=lambda _: None, **overrides) -> ClassifierTrainer:
        torch.manual_seed(0)
        config = TrainingConfig(epochs=2, batch_size=8, gradient_accumulation_steps=2, learning_rate=5e-3, **overrides)
        model = build_codebert(ClassifierConfig(pretrained_name="unused"))
        return ClassifierTrainer(
            model, config, torch.device("cpu"), tmp_path / "m_best.pt", tmp_path / "m_last.pt", log=log
        )

    def test_fit_records_history_resources_and_checkpoints(self, tmp_path: Path) -> None:
        train, val, labels = loaders()

        result = self.trainer(tmp_path).fit(train, val, labels)

        assert [record.epoch for record in result.history] == [1, 2]
        assert result.best_epoch in (1, 2)
        assert 0.0 <= result.best_val_pr_auc <= 1.0
        assert result.throughput_samples_per_second > 0
        assert result.peak_vram_mib is None  # CPU run
        assert result.n_parameters > 0
        assert (tmp_path / "m_best.pt").exists() and (tmp_path / "m_last.pt").exists()

    def test_balanced_class_weights(self, tmp_path: Path) -> None:
        weight = self.trainer(tmp_path)._compute_class_weight(np.array([0, 0, 0, 1]))
        assert weight.tolist() == pytest.approx([4 / 6, 4 / 2])
        assert self.trainer(tmp_path, class_weight="none")._compute_class_weight(np.array([0, 1])) is None

    def test_resume_continues_after_an_interrupted_run(self, tmp_path: Path) -> None:
        train, val, labels = loaders()

        def crash_in_epoch_two(line: str) -> None:
            if "epoch 2/2" in line:
                raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            self.trainer(tmp_path, crash_in_epoch_two).fit(train, val, labels)

        result = self.trainer(tmp_path).fit(train, val, labels, resume=True)

        assert result.resumed_from_epoch == 1
        assert [record.epoch for record in result.history] == [1, 2]
        assert result.epochs_run == 2

    def test_non_finite_loss_is_reported(self, tmp_path: Path) -> None:
        train, val, labels = loaders()
        trainer = self.trainer(tmp_path)
        with torch.no_grad():
            trainer._model.out_proj.weight.fill_(float("nan"))
        with pytest.raises(FloatingPointError, match="bf16"):
            trainer.fit(train, val, labels)


def test_training_config_rejects_bad_values() -> None:
    with pytest.raises(ValueError, match="precision"):
        TrainingConfig(precision="fp8")
    with pytest.raises(ValueError, match="at least 1"):
        TrainingConfig(batch_size=0)
    assert TrainingConfig(batch_size=4, gradient_accumulation_steps=8).effective_batch_size == 32


def test_split_parquet_lives_under_data_processed_split() -> None:
    config = DatasetConfig(project_root=Path("/p"), dataset_name="defectors")
    assert config.processed_dir == Path("/p/data/processed/split/defectors")
    assert config.model_dir == Path("/p/artifacts/model/defectors")


def test_train_command_parses_with_protocol_defaults() -> None:
    args = ParserArgs.build_parser().parse_args(
        ["train", "--dataset", "defectors", "--scenario", "ML_to_NonML", "--model", "codet5"]
    )
    assert (args.command, args.model, args.seeds, args.epochs) == ("train", "codet5", None, 3)


def test_default_seeds_follow_the_protocol() -> None:
    assert TrainingExperiment.default_seeds(SplitScenario.ML_TO_NON_ML) == [42, 43, 44]
    assert TrainingExperiment.default_seeds(SplitScenario.GROUP_KFOLD_NON_ML) == [42, 43]


@pytest.mark.parametrize(
    ("model_key", "build"), [("codebert", build_codebert), ("codet5", build_codet5)], ids=["codebert", "codet5"]
)
def test_experiment_end_to_end_smoke(
    model_key, build, config: DatasetConfig, project_root: Path, synthetic_jit, monkeypatch, roberta_like_tokenizer
) -> None:
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: roberta_like_tokenizer)
    config = replace(config, tokenizer_max_length=96)
    write_raw_jit(project_root, synthetic_jit)
    out = config.processed_dir
    DefectorsDataPipeline(config).run_pipeline(SplitScenario.WITHIN_DOMAIN_ML_DL, output_dir=out, tokenize=True)
    tokenizer = DefectorsTokenizer(config, max_length=96)
    tokenizer._tokenizer = roberta_like_tokenizer

    experiment = TrainingExperiment(
        config,
        model_key,
        out,
        training_config=TrainingConfig(
            epochs=1,
            batch_size=4,
            gradient_accumulation_steps=2,
            max_length=96,
            max_train_samples=24,
            max_val_samples=12,
            precision="fp32",
        ),  # fmt: skip
        model_factory=build,
        tokenizer=tokenizer,
    )
    results = experiment.run(SplitScenario.WITHIN_DOMAIN_ML_DL, seeds=[1, 2])

    assert len(results) == 2
    assert set(results["model"]) == {model_key}
    assert (results["n_train"] == 24).all() and (results["n_val"] == 12).all()
    checkpoints = {path.name for path in config.model_dir.iterdir()}
    assert f"{model_key}_Within_Domain_ML_DL_seed1_best.pt" in checkpoints
    registry = [json.loads(line) for line in config.run_registry_path.read_text().splitlines()]
    assert [entry["seed"] for entry in registry] == [1, 2]
    assert registry[0]["training_config"]["max_length"] == 96 and len(registry[0]["history"]) == 1


def test_experiment_reports_missing_tokenized_files(config: DatasetConfig, project_root: Path) -> None:
    experiment = TrainingExperiment(config, "codebert", project_root / "none", tokenizer=DefectorsTokenizer(config))
    with pytest.raises(FileNotFoundError, match="run-pipeline"):
        experiment.run(SplitScenario.ML_TO_NON_ML, seeds=[1])


def test_unknown_model_is_rejected(config: DatasetConfig, project_root: Path) -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        TrainingExperiment(config, "gpt", project_root)


def test_benchmark_smoke_and_run_time_estimate(config: DatasetConfig) -> None:
    from software_defects.models.model_benchmark import ModelBenchmark

    # Use the real manifests of the repository for the estimate, a tiny random model for the timing
    real = replace(config, project_root=Path(__file__).resolve().parents[1])
    benchmark = ModelBenchmark(
        real, "codebert", TrainingConfig(precision="fp32"), ClassifierConfig(pretrained_name="x"), build_codebert
    )
    benchmark._reporter = type(benchmark._reporter)(
        config=config, domain_data="defectors", pipeline="b", class_name="B"
    )

    results = benchmark.run([1, 2], seq_len=64, steps=1, epochs=1)

    assert results["oom"].tolist() == [False, False]
    assert (results["train_samples_per_second"] > 0).all()
    assert ModelBenchmark.max_batch_size(results) == 2
    estimate = benchmark.estimate(results, epochs=1)
    assert estimate["runs"].sum() == 35
    assert (estimate["hours"] > 0).all()
