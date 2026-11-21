# Defectors Dataset Commands

This document describes all available commands for processing and analyzing the Defectors dataset.

## Prerequisites

### 1. Download the Dataset

Download [Defectors: A Large Scale Python Dataset for Defect Prediction](https://zenodo.org/records/7708984) from Zenodo and place it under `data/raw/` with a domain-specific folder:

```
data/raw/defectors/
```

The dataset should contain:
- `defectors.zip` - Main dataset with JIT and line-level splits
- `bug_inducing_commits.zip` - Bug-inducing commits metadata
- `filtered_bug_inducing_commits.yaml` - Filtered commits
- `repo_links.yaml` - Repository links

### 2. Extract Dataset Files

Extract the main dataset zip file:

```Shell
python -c "import zipfile; zipfile.ZipFile('data/raw/defectors/defectors.zip').extractall('data/raw/defectors/defectors')"
```

This will create the following structure:
```
data/raw/defectors/defectors/
├── jit_bug_prediction_splits/
│   ├── random/
│   │   ├── train.parquet.gzip
│   │   ├── test.parquet.gzip
│   │   └── val.parquet.gzip
│   └── time/
│       ├── train.parquet.gzip
│       ├── test.parquet.gzip
│       └── val.parquet.gzip
└── line_bug_prediction_splits/
    ├── random/
    │   ├── train.parquet.gzip
    │   ├── test.parquet.gzip
    │   └── val.parquet.gzip
    └── time/
        ├── train.parquet.gzip
        ├── test.parquet.gzip
        └── val.parquet.gzip
```

### 3. Install Dependencies

Install required dependencies:

```Shell
uv sync
```

### 4. Domain Mapping

Repository domains are defined in a single file: `artifacts/mapping/repository_mapping.yaml`.
Keys must match the `repo` column exactly (case-sensitive, e.g. `spaCy`, `core`).
Repositories under `ambiguous_repositories` and `unmapped_repositories` are excluded from all
ML vs non-ML experiments.

The same file holds the test holdout in `holdout_repositories` (`ml_dl` and `non_ml` lists). It is the only place
that decides which repositories are tested in `Repository_Level_Split`, `Within_Domain_ML_DL` and
`Within_Domain_Non_ML`, and it is reported by checkpoints 8 and 11. Current holdout:

- ML/DL: `jax`, `lightning` (about 30% of the ML/DL rows)
- Non-ML: `airflow`, `ansible`, `black`, `celery` (about 33% of the Non-ML rows)

Every holdout repository must be mapped to its own domain, and a domain can never be held out completely
(otherwise loading fails with a clear error).

### 5. Tokenizer Configuration (Optional)

The data pipeline uses multiple tokenizers by default for model comparison:
- **CodeBERT**: `microsoft/codebert-base`
- **CodeT5-small**: `Salesforce/codet5-small`

To customize tokenizer models, modify `tokenizer_models` in `src/software_defects/config.py`:

```python
tokenizer_models: dict[str, str] = {
    "codebert": "microsoft/codebert-base",
    "codet5": "Salesforce/codet5-small",
    # Add more models as needed
}
```

The first run requires internet access to download tokenizers. They are cached locally afterwards.

---

## Exploratory Commands

### 1. Run Dataset Exploratory

Run the complete dataset exploratory with all 18 checkpoints:

```Shell
uv run python -m software_defects inspect-data --dataset defectors
```

Run specific checkpoints (e.g., checkpoints 1, 3, and 5-7):

```Shell
uv run python -m software_defects inspect-data --dataset defectors --checkpoints 1,3,5-7
```

The exploratory performs the following checkpoints:

- **Checkpoint 1: Dataset Integrity**
  - Parquet file readability
  - Record counts (JIT: 213,420 total, Line-level: 213,420 total)
  - Column verification: `datetime`, `commit`, `repo`, `filepath`, `content`, `methods`, `lines`
  - Data quality checks (empty rows, null values)

- **Checkpoint 2: Label Validity**
  - Label column: `lines` (numpy array of line numbers)
  - Label interpretation: empty array = 0 (defect-free), non-empty array = 1 (defective)
  - Label distribution:
    - Train: 52.86% defect-free, 47.14% defective
    - Test: 87.54% defect-free, 12.46% defective
    - Val: 87.54% defect-free, 12.46% defective

- **Checkpoint 3: JIT Input Validity**
  - Content column contains Git diffs
  - Empty content: 1.43%
  - Content length stats (train):
    - Mean: 1,885 chars
    - Median: 810 chars
    - P90: 3,956 chars
    - P95: 6,329 chars
    - P99: 14,951 chars

- **Checkpoint 4: Repository and Domain Validity**
  - 24 repositories used (7 ML/DL, 17 Non-ML) plus `freqtrade` (1 record, unmapped, excluded): 25 `repo` values in the files
  - Top repositories: ansible (39,060), sentry (33,506), core (33,292)
  - 81,778 unique commits
  - Mean files per commit: 2.37

- **Checkpoint 5: Leakage and Unit Analysis**
  - No duplicate (repo+commit+filepath) records
  - Repository overlap: 24 out of 24 used repositories overlap between train and test
  - WARNING: Current splits have repository overlap - need repository-level split for valid evaluation

- **Checkpoint 6: Cross-domain Feasibility**
  - ML/DL repositories: 7 (lightning, jax, transformers, ray, scikit-learn, yolov5, spaCy)
  - Non-ML repositories: 17
  - Label distribution by domain (train):
    - ML/DL: 48.70% defective
    - Non-ML: 46.81% defective

- **Checkpoint 7: Record Reconciliation**
  - Freqtrade status verification (1 record in train only)
  - Repository overlap analysis across splits (train, test, val)
  - Repository-level distribution table generation
  - Sample count, defective count, defective percentage per repository and domain
  - CSV output with repository_distribution.csv

- **Checkpoint 8: Split Strategy Evaluation**
  - Current split analysis (random vs repository-level)
  - Data leakage identification from repository overlap
  - Configured repository-level holdout, read from `holdout_repositories` in the mapping YAML
    (2 ML/DL repos and 4 Non-ML repos for test)
  - Projected sample distribution over train + val + test, after the same filtering as the pipeline
    (freqtrade and empty-content rows removed)
  - JSON output with split_recommendations.json (`configured_repository_level_split`)

- **Checkpoint 9: Label Distribution Analysis**
  - Global label distribution statistics
  - Per-domain label distribution (ML/DL vs Non-ML)
  - Per-repository label distribution
  - CSV output with label_distribution.csv including summary rows for GLOBAL, ML_DL_DOMAIN, and NON_ML_DOMAIN

- **Checkpoint 10: Repository Mapping Analysis**
  - Classification of all repositories with domain assignments
  - Justifications for each repository's domain classification
  - Repository-level statistics (sample count, unique commits)
  - CSV output with repository_mapping.csv

- **Checkpoint 11: Repository-Level Split Analysis**
  - Runs the real `DefectorsSplitGenerator`, so the numbers equal what the pipeline trains on
  - ML to Non-ML cross-domain split
  - Non-ML to ML cross-domain split
  - Repository-level split with the configured holdout
  - Within-domain splits for ML/DL and Non-ML
  - CSV output with split_report.csv (train, val and test samples per scenario)

- **Checkpoint 12: Token Distribution Analysis**
  - Token length statistics (mean, median, percentiles) for multiple models
  - Token overflow analysis for CodeBERT (microsoft/codebert-base, 512 token limit)
  - Token overflow analysis for CodeT5-small (Salesforce/codet5-small, 512 token limit)
  - CSV output with token_analysis.csv

- **Checkpoint 13: Shortcut Signal Check**
  - AUC calculation for single features predicting labels (diff_length, added_lines, removed_lines, num_hunks)
  - AUC per split (train, test, val)
  - AUC per domain (ML/DL vs Non-ML)
  - Label prevalence by file extension analysis
  - CSV output with shortcut_signal_auc.csv

- **Checkpoint 18: Run Report** (always runs last, whatever the checkpoint order requested)
  - Compilation of all warnings from the other checkpoints
  - Combined JSON report with metadata: timestamp, git commit, commit date, branch, `git_dirty`,
    library versions, SHA-256 of the three JIT parquet files and of the repository mapping, and the
    checkpoints that ran
  - A warning is raised when the working tree is dirty, so commit first and run `inspect-data` after
    the commit; the recorded commit then describes the code that produced the report
  - Structured warnings with severity levels
  - Markdown summary report
  - JSON output with combined_run_report.json
  - Markdown output with combined_run_report.md

- **Checkpoint 14: Path, Files-per-Commit and Commit-Time Shortcut Probe** (P0.6)
  - Single-signal ROC-AUC for path depth/length, file name length, test-file and `__init__.py` flags,
    files changed in the same commit, commit day, hour and weekday, as pooled AUC and as
    within-repository AUC (the pooled value mixes in prevalence differences between repositories)
  - Path-only and file-name-only TF-IDF + Logistic Regression probes under 5-fold cross-validation
    grouped by repository (PR-AUC and lift over prevalence on unseen repositories)
  - Prevalence by files-per-commit bucket and by commit year and domain
  - CSV output with path_metadata_signal_auc.csv, path_text_probe.csv, files_in_commit_prevalence.csv,
    commit_year_prevalence.csv

- **Checkpoint 15: Repository Identity Probe** (P0.7)
  - Repository identification from the diff text and from the file path (linear SVM, 5-fold grouped by
    commit, at most 3,000 rows per repository): accuracy, lift over the majority rate, macro-F1, per-repo recall
  - Domain identification (ML/DL vs non-ML) from the diff text on repositories never seen in training
  - CSV output with repository_identity_probe.csv, repository_identity_recall.csv

- **Checkpoint 16: Duplicate and Near-Duplicate Diffs** (P0.8)
  - Exact duplicates (hash of the raw diff), normalized duplicates (changed lines only, whitespace and hunk
    offsets ignored) and near duplicates (MinHash with banding, verified at Jaccard >= 0.8)
  - Cross-repository, cross-domain and label-conflict groups
  - Share of test rows that also occur on the training side for every experiment layout
  - CSV output with duplicate_key_summary.csv, duplicate_split_exposure.csv

- **Checkpoint 17: Prevalence Shift** (P0.9)
  - Prevalence per repository with a Wilson 95% interval
  - Train/val/test prevalence and test-minus-train shift for every scenario and every Group K-Fold fold,
    produced by the real split generator; shifts of 10 percentage points or more raise a warning
  - CSV output with prevalence_per_repo.csv, prevalence_shift_scenarios.csv

**Run only the new probes**:

```Shell
uv run python -m software_defects inspect-data --dataset defectors --checkpoints 14-18
```

**Optional arguments**:
- `--project-root` : Project root that contains `data/raw/`. Defaults to the current working directory.
- `--dataset` : Dataset folder name in `data/raw/` (e.g., `defectors`). Required.
- `--nrows` : Read only the first N rows of the file. Defaults to all rows.
- `--checkpoints` : Comma-separated list of checkpoint numbers to run (e.g., '1,3,5-7'). Runs all 18 checkpoints if not specified.

**Artifacts**:
- `repository_distribution.csv` - Repository-level statistics (sample count, defective count, defective percentage, domain)
- `split_recommendations.json` - Split strategy recommendations with projected sample counts
- `label_distribution.csv` - Global, per-domain, and per-repository label distribution
- `repository_mapping.csv` - Classification of repositories with justifications
- `split_report.csv` - Repository-level split simulation for ML → non-ML and vice versa
- `token_analysis.csv` - Token length distribution for CodeBERT (microsoft/codebert-base) and CodeT5-small (Salesforce/codet5-small)
- `shortcut_signal_auc.csv` - AUC scores for shortcut features per split and domain
- `path_metadata_signal_auc.csv`, `path_text_probe.csv`, `files_in_commit_prevalence.csv`, `commit_year_prevalence.csv` - Checkpoint 14
- `repository_identity_probe.csv`, `repository_identity_recall.csv` - Checkpoint 15
- `duplicate_key_summary.csv`, `duplicate_split_exposure.csv` - Checkpoint 16
- `prevalence_per_repo.csv`, `prevalence_shift_scenarios.csv` - Checkpoint 17
- `combined_run_report.json` - Combined report with metadata, warnings, and checkpoint summaries
- `combined_run_report.md` - Markdown summary of the combined run report
- `dataset_exploratory.txt` - Full exploratory report
- `dataset_exploratory.json` - JSON summary

The exploratory includes 18 checkpoints:
- Checkpoint 1: Dataset integrity
- Checkpoint 2: Label validity
- Checkpoint 3: JIT input validity
- Checkpoint 4: Repository and domain validity
- Checkpoint 5: Leakage and unit analysis
- Checkpoint 6: Cross-domain feasibility
- Checkpoint 7: Record reconciliation and repository distribution
- Checkpoint 8: Split strategy evaluation
- Checkpoint 9: Label distribution analysis (global, per domain, per repository)
- Checkpoint 10: Repository mapping with classification and justification
- Checkpoint 11: Repository-level split simulation (ML ↔ non-ML)
- Checkpoint 12: Token distribution for CodeBERT (microsoft/codebert-base) and CodeT5-small (Salesforce/codet5-small)
- Checkpoint 13: Shortcut signal check (AUC for diff features)
- Checkpoint 14: Path, files-per-commit and commit-time shortcut probe
- Checkpoint 15: Repository identity probe
- Checkpoint 16: Duplicate and near-duplicate diffs
- Checkpoint 17: Prevalence shift per repository, scenario and fold
- Checkpoint 18: Run report (combined warnings, git commit and data hashes; always runs last)

---

## Data Pipeline Commands

### 2. Run Data Processing Pipeline

Generate dataset splits with optional tokenization and commit-level aggregation:

```Shell
uv run python -m software_defects run-pipeline --dataset defectors --scenario <SCENARIO> [OPTIONS]
```

**Required arguments**:
- `--dataset` : Dataset folder name in `data/raw/` (e.g., `defectors`). Required.
- `--scenario` : Split scenario to use. Required. Choices:
  - `ML_to_NonML` - Train on ML/DL, test on Non-ML
  - `NonML_to_ML` - Train on Non-ML, test on ML/DL
  - `Repository_Level_Split` - Repository-level holdout (mixed domains)
  - `Within_Domain_ML_DL` - Within-domain ML/DL holdout
  - `Within_Domain_Non_ML` - Within-domain Non-ML holdout

**Optional arguments**:
- `--project-root` : Project root that contains `data/raw/`. Defaults to the current working directory.
- `--output-dir` : Directory to save processed data. If not specified, data is not saved.
- `--commit-level` : Aggregate data to commit-level instead of file-level.
- `--no-tokenize` : Skip tokenization step.
- `--fold` : Fold index (0-4), only for the Group K-Fold scenarios. If omitted, all 5 folds are generated.

### Pipeline Features

The data pipeline performs the following steps:
1. **Load Data**: Combine train, test, and val splits from the original dataset
2. **Generate Split**: Apply the specified split scenario with repository-level holdout
3. **Aggregate** (optional): Aggregate file-level data to commit-level using max() rule for labels
4. **Tokenize** (optional): Apply HuggingFace tokenizers for multiple models (CodeBERT and CodeT5-small)
5. **Save** (optional): Save processed data to Parquet files with model-specific directories

List Scenario
ML_to_NonML
NonML_to_ML
Repository_Level_Split
Within_Domain_ML_DL
Within_Domain_Non_ML
Group_KFold_ML_DL
Group_KFold_Non_ML

### Usage Examples

### ML to Non-ML (Cross-domain)
```Shell
uv run python -m software_defects run-pipeline --dataset defectors --scenario ML_to_NonML
```
### Non-ML to ML (Cross-domain)
```Shell
uv run python -m software_defects run-pipeline --dataset defectors --scenario NonML_to_ML
```
### Repository Level Split (Mixed domains)
```Shell
uv run python -m software_defects run-pipeline --dataset defectors --scenario Repository_Level_Split
```
### Within-Domain ML/DL
```Shell
uv run python -m software_defects run-pipeline --dataset defectors --scenario Within_Domain_ML_DL
```
### Within-Domain Non-ML
```Shell
uv run python -m software_defects run-pipeline --dataset defectors --scenario Within_Domain_Non_ML
```
### Group K-Fold ML/DL (Within-domain)
Generates all 5 folds, one processed directory per fold (`.../Group_KFold_ML_DL/fold_0` to `fold_4`):
```Shell
uv run python -m software_defects run-pipeline --dataset defectors --scenario Group_KFold_ML_DL
```
### Group K-Fold Non-ML (Within-domain)
```Shell
uv run python -m software_defects run-pipeline --dataset defectors --scenario Group_KFold_Non_ML
```
### A single fold only
```Shell
uv run python -m software_defects run-pipeline --dataset defectors --scenario Group_KFold_ML_DL --fold 2
```

## Pipeline Output Format

### Directory Structure
When `--output-dir` is specified with tokenization enabled:
```
{output_dir}/{scenario}/
├── {scenario}_train.parquet          # Raw (non-tokenized) train data
├── {scenario}_val.parquet            # Raw (non-tokenized) val data
├── {scenario}_test.parquet           # Raw (non-tokenized) test data
├── codebert/                         # CodeBERT-tokenized data
│   ├── {scenario}_train.parquet
│   ├── {scenario}_val.parquet
│   └── {scenario}_test.parquet
└── codet5/                           # CodeT5-small-tokenized data
    ├── {scenario}_train.parquet
    ├── {scenario}_val.parquet
    └── {scenario}_test.parquet
```

For `Group_KFold_ML_DL` and `Group_KFold_Non_ML` the same layout is repeated inside `{output_dir}/{scenario}/fold_{n}/`.

**Note**: The raw parquet files (at the root of the scenario directory) are used by the baseline experiment (TF-IDF + Logistic Regression). The model-specific subdirectories (codebert/ and codet5/) contain tokenized data for deep learning models.

### File-Level Output
Columns:
- `datetime`, `commit`, `repo`, `filepath`, `content`, `methods`, `lines` (original)
- `label` (binary: 0/1)
- `token_count` (if tokenized)
- `truncated` (if tokenized)

### Commit-Level Output
Columns:
- `commit`, `label`, `content`, `repo`, `lines`
- `content` is concatenated diffs from all files in commit
- `lines` is merged list of buggy line numbers
- `label` is max of file-level labels

### Pipeline Artifacts

The pipeline saves the following artifacts under `artifacts/report/defectors/data_pipeline/DefectorsDataPipeline/<timestamp>/`:
- `pipeline_<scenario>.txt` - Text report with processing statistics
- `pipeline_<scenario>.json` - JSON summary with structured data including:
  - Split information (scenario, sample counts, defective percentages, repository lists)
  - Tokenization statistics for each model (mean/median tokens, truncation percentages)
  - Output paths for raw and model-specific tokenized data

### Split Scenarios

#### ML_to_NonML
- **Train**: All ML/DL repositories (7 repos)
- **Test**: All Non-ML repositories (19 repos)
- **Use case**: Cross-domain evaluation from ML/DL to general software

#### NonML_to_ML
- **Train**: All Non-ML repositories (19 repos)
- **Test**: All ML/DL repositories (7 repos)
- **Use case**: Cross-domain evaluation from general software to ML/DL

#### Repository_Level_Split
- **Train**: Mixed domains with repository holdout
- **Test**: Holdout repositories from the mapping YAML (2 ML/DL + 4 Non-ML)
- **Use case**: Balanced repository-level evaluation with no overlap

#### Within_Domain_ML_DL
- **Train**: ML/DL repositories excluding holdout
- **Test**: Holdout ML/DL repositories from the mapping YAML (`jax`, `lightning`)
- **Use case**: Within-domain ML/DL evaluation

#### Within_Domain_Non_ML
- **Train**: Non-ML repositories excluding holdout
- **Test**: Holdout Non-ML repositories from the mapping YAML (`airflow`, `ansible`, `black`, `celery`)
- **Use case**: Within-domain Non-ML evaluation

### Tokenization Notes

- **Supported models**: CodeBERT (microsoft/codebert-base) and CodeT5-small (Salesforce/codet5-small)
- **Default max_length**: 512 (the CodeBERT limit; the same length is used for CodeT5-small so the comparison is fair)
- **Model-specific tokenization**: Each model uses its own tokenizer for proper comparison
- **Output**: Tokenized data saved in model-specific subdirectories (codebert/ and codet5/)
- **Truncation**: Enabled by default for sequences exceeding max_length
- **First run**: Requires internet access to download tokenizers (cached afterwards)
- **Token statistics**: Mean, median, percentiles, and overflow percentage reported for each model
- **Configuration**: Tokenizer models can be customized via `DatasetConfig.tokenizer_models` in config.py

### Commit-Level Aggregation

- **Aggregation rule**: Commit is defective if any file-change is defective (max of file labels)
- **Content**: Concatenated diffs from all files in the commit
- **Lines**: Merged list of buggy line numbers from all files
- **Size reduction**: Reduces dataset size by ~2.4x (based on average files per commit)

---

## Baseline Commands

### 3. Run Baselines (TF-IDF + Logistic Regression, diff-size only)

Run a baseline experiment on processed data from the pipeline. `--model tfidf` (default) is TF-IDF + Logistic Regression.
`--model length` uses only four diff-size features (diff length, added lines, removed lines, hunks) and is the
reference for shortcut signals: a model that reads code should beat it clearly.

```Shell
uv run python -m software_defects run-baseline --dataset defectors --scenario <SCENARIO> [OPTIONS]
```

**Required arguments**:
- `--dataset` : Dataset folder name in `data/raw/` (e.g., `defectors`). Required.
- `--scenario` : Split scenario to evaluate. Required. Choices:
  - `ML_to_NonML` - Train on ML/DL, test on Non-ML
  - `NonML_to_ML` - Train on Non-ML, test on ML/DL
  - `Repository_Level_Split` - Repository-level holdout (mixed domains)
  - `Within_Domain_ML_DL` - Within-domain ML/DL holdout
  - `Within_Domain_Non_ML` - Within-domain Non-ML holdout
  - `Group_KFold_ML_DL` - Group K-Fold for ML/DL (5 folds)
  - `Group_KFold_Non_ML` - Group K-Fold for Non-ML (5 folds)
  - `Cross_Domain_Default` - Cross-domain with default settings

**Optional arguments**:
- `--project-root` : Project root that contains `data/raw/`. Defaults to the current working directory.
- `--input-dir` : Directory containing processed parquet files from pipeline. If not specified, defaults to `data/processed/defectors`.
- `--model` : `tfidf` (default) or `length` (diff-size features only).
- `--seeds` : Comma-separated random seeds for multiple runs (default: `42`). Example: `42,43,44`.
- `--equalize-train-size` : Subsample the training set to this many rows for size comparisons.
- `--exclude-repos` : Comma-separated list of repositories to exclude from the analysis (e.g., `openpilot`). Useful for sensitivity analysis.

### Prerequisites

The baseline command requires that the data pipeline has been run first to generate processed parquet files:

```Shell
# Step 1: Run the pipeline to generate processed data
uv run python -m software_defects run-pipeline --dataset defectors --scenario Within_Domain_ML_DL --output-dir data/processed/defectors

# Step 2: Run the baseline on the processed data
uv run python -m software_defects run-baseline --dataset defectors --scenario Within_Domain_ML_DL --input-dir data/processed/defectors
```

If the processed parquet files are not found, the baseline will fail with an error message instructing you to run the pipeline first.

### Usage Examples

### Single seed run
```Shell
uv run python -m software_defects run-baseline --dataset defectors --scenario Within_Domain_ML_DL
```

### Multiple seeds for robustness
```Shell
uv run python -m software_defects run-baseline --dataset defectors --scenario Within_Domain_ML_DL --seeds 42,43,44
```

### With custom input directory
```Shell
uv run python -m software_defects run-baseline --dataset defectors --scenario ML_to_NonML --input-dir data/processed/my_experiments
```

### With train size equalization
```Shell
uv run python -m software_defects run-baseline --dataset defectors --scenario Within_Domain_ML_DL --equalize-train-size 10000
```

### Baseline Features

The baseline experiment performs the following:
1. **Load Processed Data**: Reads train/val/test parquet files from the pipeline output
2. **TF-IDF Vectorization**: Converts code diffs to TF-IDF features
3. **Logistic Regression**: Trains L2-regularized logistic regression with hyperparameter tuning
4. **Threshold Selection**: Selects optimal classification threshold on validation data
5. **Evaluation**: Computes PR-AUC, F1, MCC, ROC-AUC on test data
6. **Per-Repository Metrics**: Evaluates performance per repository
7. **Multiple Runs**: Supports multiple random seeds for robustness

### Baseline Artifacts

The baseline saves the following artifacts under `artifacts/report/defectors/baseline/<TfidfLogRegBaseline|LengthLogRegBaseline>/<timestamp>/`:
- `baseline_<scenario>.txt` - Text report with per-run statistics
- `baseline_<scenario>.json` - JSON summary with structured data
- `baseline_results.csv` - Per-run results with metrics (PR-AUC, F1, MCC, ROC-AUC, train time, inference time)
- `baseline_per_repo.csv` - Per-repository metrics for each run

### Output Format

#### baseline_results.csv
Columns:
- `scenario` - Split scenario name
- `model` - Baseline model (`tfidf` or `length`)
- `fold` - Fold number (0 for non-KFold scenarios, 0-4 for Group K-Fold)
- `seed` - Random seed used
- `n_train` - Number of training samples
- `n_val` - Number of validation samples
- `n_test` - Number of test samples
- `train_repos` - Semicolon-separated list of training repositories
- `test_repos` - Semicolon-separated list of test repositories
- `train_defective_pct` - Percentage of defective samples in training set
- `test_defective_pct` - Percentage of defective samples in test set
- `pr_auc` - Precision-Recall AUC
- `prevalence_baseline_pr_auc` - PR-AUC of prevalence baseline
- `f1` - F1 score
- `mcc` - Matthews Correlation Coefficient
- `roc_auc` - ROC AUC
- `threshold` - Optimal classification threshold
- `best_c` - Best regularization parameter C
- `n_features` - Number of TF-IDF features
- `train_seconds` - Training time in seconds
- `inference_seconds` - Inference time in seconds

#### baseline_per_repo.csv
Columns:
- `scenario` - Split scenario name
- `fold` - Fold number
- `seed` - Random seed used
- `repo` - Repository name
- `n_test` - Number of test samples from this repository
- `prevalence` - Percentage of defective samples in this repository
- `pr_auc` - PR-AUC for this repository
- `f1` - F1 score for this repository
- `mcc` - MCC for this repository
- `roc_auc` - ROC AUC for this repository

### Group K-Fold Scenarios

For `Group_KFold_ML_DL` and `Group_KFold_Non_ML` scenarios:
- The experiment runs 5 folds (repository-based cross-validation)
- Each fold reads its own processed split (`.../{scenario}/fold_{n}/`), so run the pipeline for all 5 folds first
- Each fold tests different repositories; every repository is tested in exactly one fold
- Results are aggregated across all folds for mean and standard deviation

### Error Handling

If the processed parquet files are not found, the baseline will display an error message like:
```
Processed train file not found: data/processed/defectors/Within_Domain_ML_DL/Within_Domain_ML_DL_train.parquet
Please run the data pipeline first: uv run python -m software_defects run-pipeline --dataset defectors --scenario Within_Domain_ML_DL --output-dir data/processed/defectors
```

---