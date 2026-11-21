# Dataset Scope (single wording for the form, the thesis and the README)

Use these numbers and this wording everywhere. The source of truth is
`artifacts/mapping/repository_mapping.yaml` plus the `repository_status.csv` of `inspect-data` checkpoint 7.

| Quantity                                                           | Value                                  |
| ------------------------------------------------------------------ | -------------------------------------- |
| File-change records in the released JIT files (train + test + val) | 213,420                                |
| `repo` values in the released files                                | 25                                     |
| Repositories used in the experiments                               | 24 (7 ML/DL + 17 non-ML)               |
| Repository excluded                                                | `freqtrade` (1 record, domain unknown) |
| Records after dropping `freqtrade`                                 | 213,419                                |
| Records after also dropping empty diffs (the pipeline input)       | 210,156                                |

Canonical sentence (Indonesian, for the form and the thesis):

> Dataset Defectors berisi 213.420 catatan perubahan file Python. Berkas yang dirilis memuat 25 repositori;
> penelitian ini menggunakan 24 repositori (7 Machine Learning/Deep Learning dan 17 non-Machine Learning) dan
> mengeluarkan `freqtrade` karena hanya memiliki satu catatan dan domainnya tidak dapat ditentukan. Setelah
> membuang diff kosong, 210.156 catatan tersisa sebagai input eksperimen.

Canonical sentence (English):

> The Defectors dataset contains 213,420 Python file-change records. The released files carry 25 repositories;
> this study uses 24 (7 ML/DL and 17 non-ML) and excludes `freqtrade`, which has a single record and no
> determinable domain. After removing empty diffs, 210,156 records remain as the experiment input.

Note: the original paper and earlier drafts say "24 repositories" for the whole dataset. The released files
contain `freqtrade` as a 25th identifier, so "24 used + freqtrade excluded" is the wording that matches the data.
