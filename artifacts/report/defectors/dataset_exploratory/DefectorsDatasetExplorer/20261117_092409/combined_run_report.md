# Defectors Dataset Validation Report

Generated: 2026-11-17T09:26:56.385277
Git Commit: 7062d3c08545bf9d047ee0b1d21a4d4a40f784a3

## Summary

- Total Warnings: 10

## Warnings

- **W001**:   commit_message column NOT found in JIT dataset
- **W002**:   WARNING: This may indicate double-counting if both are used
- **W003**:   Found 100 overlapping samples in first 100 rows
- **W004**:   Could not convert datetime column
- **W005**:   WARNING: Commits overlap between train and test (files from same commit in different splits)!
- **W006**:   WARNING: Datetime ranges overlap between train and test!
- **W007**:   WARNING: All repositories overlap between train and test!
- **W008**:   This is a random record-level split, not repository-level split.
- **W009**:   This causes potential data leakage.
- **W010**:   Could not load tokenizer Salesforce/codet5-small: Input must be a List[Union[str, AddedToken]]

## Checkpoint Summary

See the JSON report for detailed checkpoint results.