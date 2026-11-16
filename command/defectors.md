This is place for documenting command that for run process/pipeline/something else.
if some workflow is multiple domain, create file again based on domain.

## Defectors Dataset Investigation Commands

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

### 4. Run Dataset Investigation

Run the full dataset investigation following TODO.md checkpoints:

```Shell
uv run python -m software_defects inspect-data --dataset defectors
```

The investigation performs the following checkpoints:

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

Optional arguments:

- `--project-root` : Project root that contains `data/raw/`. Defaults to the current working directory.
- `--dataset` : Dataset folder name in `data/raw/` (e.g., `defectors`). Required.
- `--nrows` : Read only the first N rows of the file. Defaults to all rows.

### 5. Artifacts

The investigation saves the following artifacts under `artifacts/report/defectors/dataset_investigation/DefectorsDatasetExplorer/<timestamp>/`:

- `dataset_investigation.txt` - Text report with all checkpoint results
- `dataset_investigation.json` - JSON summary with structured data

### 6. Run Advanced Analysis

Run advanced analysis to generate CSV reports for label distribution, repository mapping, split simulation, and token analysis:

```Shell
uv run python -m software_defects advanced-analysis --dataset defectors
```

The advanced analysis generates the following CSV files under `artifacts/report/defectors/advanced_analysis/`:

- **label_distribution.csv** - Global, per-domain, and per-repository label distribution
  - Columns: repository, domain, total_samples, defective, defect_free, defective_percentage
  - Includes GLOBAL, ML_DL_DOMAIN, and NON_ML_DOMAIN summaries

- **repository_mapping.csv** - Classification of 25 repositories with justifications
  - Columns: repository, domain, justification, total_samples, unique_commits
  - Domains: ml_dl (ML/DL repositories), non_ml (Non-ML repositories)

- **split_report.csv** - Repository-level split simulation for ML → non-ML and vice versa
  - Columns: scenario, train_domain, test_domain, train_repositories, test_repositories, train_samples, test_samples, train_defective_pct, test_defective_pct, train_repos_list, test_repos_list
  - Scenarios:
    - ML_to_NonML: Train on ML/DL, test on Non-ML
    - NonML_to_ML: Train on Non-ML, test on ML/DL
    - Repository_Level_Split: Mixed train/test with repository holdout

- **token_analysis.csv** - Token length distribution for CodeBERT and CodeT5-small
  - Columns: model, max_sequence_length, mean_tokens, median_tokens, std_tokens, min_tokens, max_tokens, p25_tokens, p50_tokens, p75_tokens, p90_tokens, p95_tokens, p99_tokens, overflow_count, overflow_percentage
  - Shows token statistics and overflow beyond 512-token limit

The advanced analysis also saves the following artifacts under `artifacts/report/defectors/advanced_analysis/DefectorsAdvancedAnalysis/<timestamp>/`:

- `advanced_analysis.txt` - Text report with all analysis results
- `advanced_analysis.json` - JSON summary with structured data

### Key Findings

1. **Dataset Size**: 213,420 records (matches publication: 213,419)
2. **Label Balance**: Reasonably balanced in train (47% defective), imbalanced in test/val (12% defective)
3. **Repository Overlap**: Current splits have significant repository overlap - requires repository-level split for valid cross-domain evaluation
4. **Domain Coverage**: Both ML/DL (6 repos) and non-ML (19 repos) domains are well-represented
5. **Content Quality**: Only 1.43% empty content, reasonable token lengths for model processing
