# Defectors Dataset Validation Report

Generated: 2026-11-17T07:31:59.928074
Git Commit: 611727dd8f3c52a04f41afd9713b2a381aac9828

## Summary

- Total Warnings: 11

## Warnings

- **W001**:   commit_message column NOT found in JIT dataset
- **W002**:   WARNING: This may indicate double-counting if both are used
- **W003**:   Found 100 overlapping samples in first 100 rows
- **W004**:   Could not convert datetime column
- **W005**:   WARNING: Commits overlap between train and test (files from same commit in different splits)!
- **W006**:   WARNING: Datetime ranges overlap between train and test!
- **W007**:   UNMAPPED repositories in data: {'yolov5', 'transformers', 'redash', 'poetry', 'celery', 'localstack', 'sentry', 'scikit-learn', 'lightning', 'pandas', 'core', 'jax', 'django', 'airflow', 'django-rest-framework', 'ansible', 'openpilot', 'scrapy', 'freqtrade', 'numpy', 'ray', 'pipenv', 'black', 'cpython', 'spaCy'}
- **W008**:   MISMATCH: 0 != 213,420
- **W009**:   WARNING: All repositories overlap between train and test!
- **W010**:   This is a random record-level split, not repository-level split.
- **W011**:   This causes potential data leakage.

## Checkpoint Summary

See the JSON report for detailed checkpoint results.