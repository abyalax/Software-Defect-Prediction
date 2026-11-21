Read this before writing code
File Structure
```Shell
└── 📁boilerplate
    └── 📁artifacts ('all here is example expected structure')
        └── 📁model
            └── 📁exampledomain
                ├── lstm_ae_w1_best.pt   # format naming [modelname]_[windowconfig]_best.pt
                ├── lstm_ae_w10_best.pt
                ├── lstm_ae_w3_best.pt
                ├── lstm_ae_w5_best.pt
        └── 📁report
            └── 📁exampledomain
                └── 📁baseline
                    └── 📁BaselineEvaluator
                        └── 📁20261104_145907
                            ├── ablation_report.json
                            ├── ablation_report.txt
                └── 📁exploratory
                    └── 📁ExampleDatasetsExplorer
                        └── 📁20261104_100038
                            ├── dataset_report.json
                            ├── dataset_report.txt
                        └── 📁20261104_100136
                            ├── dataset_report.json
                            ├── dataset_report.txt
                └── 📁model_training
                    └── 📁ExampleModelTrainingPipeline
                        └── 📁20261104_164946
                            ├── training_report.json
                            ├── training_report.txt
                        └── 📁20261104_165221
                            ├── training_report.json
                            ├── training_report.txt
        └── 📁temp
    └── 📁data ('all here is example expected structure')
        └── 📁raw 
            └── 📁example_datasets_domain
                ├── example_datasets.csv
                └── example_datasets.zip
        └── 📁processed
    └── 📁scripts
    └── 📁src
        └── 📁boilerplate ('on here, choose based on context project, organized file by action or pipelines or modules or data_domain')
            └── 📁organized_by_action   
            └── 📁organized_by_modules   
            └── 📁organized_by_pipelines
            └── 📁cli
                ├── __init__.py
                ├── cli.py
                ├── command.py
                ├── parser.py
                └── runner.py
            ├── __main__.py
            └── config.py
    ├── .gitignore
    ├── .python-version
    ├── pyproject.toml
    ├── README.md
    └── uv.lock
```
Principal Design
OOP
- Single Responsibility: setiap class memiliki tanggung jawab yang jelas.
- One Main Class per Module: setiap file Python memiliki satu class utama. Pengecualian hanya untuk dataclass, enum, dan tipe data pendukung.
- Explicit Typing: semua parameter dan return value function/method memiliki type annotation.
- Private Helpers: fungsi internal menggunakan prefix _, misalnya _normalize_target().
- CLI as Entry Point: cli.py hanya mengurus argument parsing dan dispatch command, not main proces execution.
- class biasa: memiliki behavior atau proses.
- @dataclass(frozen=True): merepresentasikan data yang tidak perlu dimodifikasi.
- Enum: merepresentasikan kumpulan nilai konstan.
- Private methods: implementasi internal suatu class.
- Public methods: behavior yang memang boleh digunakan modul lain.
- Tidak perlu membuat dataclass untuk setiap variabel atau hasil operasi sederhana.
- Hindari class yang terlalu besar.
- Jangan memaksakan OOP.
- Function internal yang sederhana boleh menjadi private method. 
- Dataclass hanya digunakan ketika memang ada kebutuhan merepresentasikan data.

Pattern
- Domain Separation: data loading, exploration, metrics, dan reporting dipisahkan.
- Pisahkan konfigurasi dan data.
- dataset raw harus di proses menjadi parquet sebelum masuk ke heavy operation, save it under processed/.
- Path dataset, nama kolom target, parameter model, dan konfigurasi training melalui configuration object
- output long execution must be report and save on file 

Avoid
- Long line that more than define on pyproject.toml for ruff
- business logic di CLI
- membuat utility module yang terlalu umum.
- Commit and push whatever yoo do, just keep it on local changes

Flow Data and Execution (example)
CLI
 │
 └── inspect-data
       │
       └── DatasetExplorer
              │
              ├── DatasetLoader
              │      └── load()
              │
              ├── DatasetExplorer
              │      └── inspect()
              │
              └── ConsoleReporter
                     ├── display_section()
                     ├── display_text()
                     └── display_table()

| Aturan             | Konvensi                            |
| ------------------ | ----------------------------------- |
| Class utama        | `PascalCase`                        |
| Dataclass          | `PascalCase`                        |
| Public method      | `snake_case`                        |
| Private method     | `_snake_case`                       |
| Constant           | `UPPER_SNAKE_CASE`                  |
| Function arguments | Wajib type annotation               |
| Return value       | Wajib type annotation               |
| Class attributes   | Wajib type annotation               |
| Docstring          | Wajib untuk public class dan method |
| Imports            | Absolute imports                    |
| Formatting         | Ruff                                |
| Testing            | Pytest                              |