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

---

## Exploratory Commands

### 1. Run Dataset Exploratory

Run the complete dataset exploratory with all 12 checkpoints:

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
  - 25 repositories total
  - Top repositories: ansible (39,060), sentry (33,506), core (33,292)
  - 81,778 unique commits
  - Mean files per commit: 2.37

- **Checkpoint 5: Leakage and Unit Analysis**
  - No duplicate (repo+commit+filepath) records
  - Repository overlap: 24 out of 25 repositories overlap between train and test
  - WARNING: Current splits have repository overlap - need repository-level split for valid evaluation

- **Checkpoint 6: Cross-domain Feasibility**
  - ML/DL repositories: 6 (lightning, jax, transformers, ray, scikit-learn, yolov5)
  - Non-ML repositories: 19
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
  - Recommended split strategies:
    - Balanced Repository-Level Split: holdout 2 ML/DL repos and 4 Non-ML repos for test
    - Within-Domain Splits: holdout repos within the same domain
    - Cross-Domain Splits: train on one domain, test on the other
  - Projected sample distributions for each recommended strategy
  - JSON output with split_recommendations.json

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
  - Simulation of different split scenarios
  - ML to Non-ML cross-domain split
  - Non-ML to ML cross-domain split
  - Repository-level split with holdout
  - Within-domain splits for ML/DL and Non-ML
  - CSV output with split_report.csv

- **Checkpoint 12: Token Distribution Analysis**
  - Token length statistics (mean, median, percentiles)
  - Token overflow analysis for CodeBERT (512 token limit)
  - Token overflow analysis for CodeT5-small (512 token limit)
  - CSV output with token_analysis.csv

**Optional arguments**:
- `--project-root` : Project root that contains `data/raw/`. Defaults to the current working directory.
- `--dataset` : Dataset folder name in `data/raw/` (e.g., `defectors`). Required.
- `--nrows` : Read only the first N rows of the file. Defaults to all rows.
- `--checkpoints` : Comma-separated list of checkpoint numbers to run (e.g., '1,3,5-7'). Runs all 12 checkpoints if not specified.

**Artifacts**:
- `repository_distribution.csv` - Repository-level statistics (sample count, defective count, defective percentage, domain)
- `split_recommendations.json` - Split strategy recommendations with projected sample counts
- `label_distribution.csv` - Global, per-domain, and per-repository label distribution
- `repository_mapping.csv` - Classification of repositories with justifications
- `split_report.csv` - Repository-level split simulation for ML → non-ML and vice versa
- `token_analysis.csv` - Token length distribution for CodeBERT and CodeT5-small
- `dataset_exploratory.txt` - Full exploratory report
- `dataset_exploratory.json` - JSON summary

The exploratory includes 12 checkpoints:
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
- Checkpoint 12: Token distribution for CodeBERT and CodeT5-small

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

### Pipeline Features

The data pipeline performs the following steps:
1. **Load Data**: Combine train, test, and val splits from the original dataset
2. **Generate Split**: Apply the specified split scenario with repository-level holdout
3. **Aggregate** (optional): Aggregate file-level data to commit-level using max() rule for labels
4. **Tokenize** (optional): Apply HuggingFace tokenizer (default: CodeBERT with max_length=512)
5. **Save** (optional): Save processed data to Parquet files

### Usage Examples

#### Generate split with tokenization
```Shell
uv run python -m software_defects run-pipeline \
  --dataset defectors \
  --scenario ML_to_NonML \
  --output-dir data/processed/defectors_ml_to_non_ml
```

#### Generate commit-level split without tokenization
```Shell
uv run python -m software_defects run-pipeline \
  --dataset defectors \
  --scenario Repository_Level_Split \
  --output-dir data/processed/defectors_repo_level \
  --commit-level \
  --no-tokenize
```

#### Generate within-domain split
```Shell
uv run python -m software_defects run-pipeline \
  --dataset defectors \
  --scenario Within_Domain_ML_DL \
  --output-dir data/processed/defectors_within_ml
```

### Output Format

#### File-Level Output
Columns:
- `datetime`, `commit`, `repo`, `filepath`, `content`, `methods`, `lines` (original)
- `label` (binary: 0/1)
- `token_count` (if tokenized)
- `truncated` (if tokenized)

#### Commit-Level Output
Columns:
- `commit`, `label`, `content`, `repo`, `lines`
- `content` is concatenated diffs from all files in commit
- `lines` is merged list of buggy line numbers
- `label` is max of file-level labels

### Pipeline Artifacts

The pipeline saves the following artifacts under `artifacts/report/defectors/data_pipeline/DefectorsDataPipeline/<timestamp>/`:
- `pipeline_<scenario>.txt` - Text report with processing statistics
- `pipeline_<scenario>.json` - JSON summary with structured data

### Split Scenarios

#### ML_to_NonML
- **Train**: All ML/DL repositories (6 repos)
- **Test**: All Non-ML repositories (18 repos)
- **Use case**: Cross-domain evaluation from ML/DL to general software

#### NonML_to_ML
- **Train**: All Non-ML repositories (18 repos)
- **Test**: All ML/DL repositories (6 repos)
- **Use case**: Cross-domain evaluation from general software to ML/DL

#### Repository_Level_Split
- **Train**: Mixed domains with repository holdout
- **Test**: Holdout repositories (1 ML/DL + 3 Non-ML)
- **Use case**: Balanced repository-level evaluation with no overlap

#### Within_Domain_ML_DL
- **Train**: ML/DL repositories excluding holdout
- **Test**: Holdout ML/DL repositories (1-2 repos)
- **Use case**: Within-domain ML/DL evaluation

#### Within_Domain_Non_ML
- **Train**: Non-ML repositories excluding holdout
- **Test**: Holdout Non-ML repositories (3-4 repos)
- **Use case**: Within-domain Non-ML evaluation

### Tokenization Notes

- **Default model**: microsoft/codebert-base
- **Default max_length**: 512 (CodeBERT/RoBERTa supports at most 512 tokens)
- **Truncation**: Enabled by default for sequences exceeding max_length
- **First run**: Requires internet access to download tokenizer (cached afterwards)
- **Token statistics**: Mean, median, percentiles, and overflow percentage reported in output

### Commit-Level Aggregation

- **Aggregation rule**: Commit is defective if any file-change is defective (max of file labels)
- **Content**: Concatenated diffs from all files in the commit
- **Lines**: Merged list of buggy line numbers from all files
- **Size reduction**: Reduces dataset size by ~2.4x (based on average files per commit)
