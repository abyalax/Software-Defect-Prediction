# Defectors Dataset Validation Report

Generated: 2026-11-21T16:36:12.068674
Git commit: de369e9d5ed23549db18970d852c40f831f6e9d1 (main, 2026-11-21T16:31:13+07:00)
Working tree dirty: True

## Summary

- Checkpoints run: checkpoint_14, checkpoint_15, checkpoint_16, checkpoint_17
- Total warnings: 7

## Warnings

- **W001**: Shortcut candidate: files_in_commit within-repo AUC 0.624
- **W002**: Repository is easy to guess (repo_from_diff_content): accuracy 0.949 vs majority 0.063
- **W003**: Repository is easy to guess (repo_from_file_path): accuracy 0.985 vs majority 0.063
- **W004**: NonML_to_ML: 1.25% of test rows have a normalized_key twin in train
- **W005**: Prevalence shift in Group_KFold_Non_ML fold 0: train 46.6% -> test 36.6% (-10.0 pp); compare PR-AUC against prevalence
- **W006**: Prevalence shift in Group_KFold_Non_ML fold 2: train 40.0% -> test 60.0% (+20.0 pp); compare PR-AUC against prevalence
- **W007**: Working tree has uncommitted changes; commit first so the recorded commit describes this run

## Checkpoint Summary

See the JSON report for detailed checkpoint results.
