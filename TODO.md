## P0. Data Exploration & Signal Probing
- [x] P0.1 Integritas dataset, rekonsiliasi record, dan distribusi label
- [x] P0.2 Validasi label dan input JIT
- [x] P0.3 Pemetaan domain dan kelayakan cross-domain
- [x] P0.4 Analisis token (CodeBERT 36.6% sampel > 512 token, CodeT5-small 25.5%)
- [x] P0.5 Probing shortcut ukuran diff (ROC-AUC fitur tunggal sekitar 0.62–0.63)
- [x] P0.6 Probing shortcut path/nama file (path kini ikut jadi input), jumlah file per commit, dan waktu commit (checkpoint 15; hasil menunggu run)
- [x] P0.7 Probing identitas repo: seberapa mudah repo asal ditebak dari isi diff (checkpoint 16; hasil menunggu run)
- [x] P0.8 Cek duplikat dan near-duplicate diff lintas repo (hash content) (checkpoint 17; hasil menunggu run)
- [x] P0.9 Analisis pergeseran prevalensi per repo dan per fold (checkpoint 18; hasil menunggu run)
- [x] P0.10 Jalankan ulang `inspect-data` setelah commit agar laporan mencatat commit terbaru (laporan kini mencatat commit, flag dirty, hash data; run ulang setelah commit patch ini)
- [x] P0.11 Samakan redaksi jumlah repositori di form dan skripsi (24 dipakai + freqtrade dikeluarkan; redaksi baku di `docs/DATASET_SCOPE.md`)

## P1. Data Cleaning & Preprocessing
- [x] P1.1 Buang freqtrade dan baris content kosong (210,156 baris tersisa)
- [x] P1.2 Label biner dari `lines` dan unit analisis file-level
- [x] P1.3 Tokenisasi CodeBERT dan CodeT5-small di pipeline dengan max_length 512
- [x] P1.4 Implementasi truncation hunk-aware (pertahankan hunk header, prioritaskan baris +/-, bagi budget token antar hunk; `diff_truncator.py`)
- [x] P1.5 Implementasi input "path file + diff lengkap" untuk kedua model (path selalu di awal dan tidak terpotong; `diff_input_builder.py`)
- [x] P1.6 Pra-potong diff berukuran ekstrem sebelum tokenisasi agar pipeline cepat
- [x] P1.7 Verifikasi tokenisasi per scenario (persen terpotong di train/val/test, special token, path utuh) dan unit test (perintah `verify-tokenization`; angka pada tokenizer asli menunggu run)
- [x] P1.8 Dokumentasikan semua langkah preprocessing untuk bab metodologi (`docs/PREPROCESSING.md`)

## P2. Data Splitting
- [x] P2.1 Holdout repo eksplisit di YAML (ML/DL: jax, lightning; non-ML: airflow, ansible, black, celery)
- [x] P2.2 Tujuh scenario: ML_to_NonML, NonML_to_ML, Repository_Level_Split, Within_Domain (ML/DL dan non-ML), Group_KFold (ML/DL dan non-ML, 5 fold)
- [x] P2.3 Val per commit dari sisi train, tanpa overlap repo/commit antara train dan test
- [x] P2.4 Test unit generator split (25 test lolos)
- [x] P2.5 Terapkan `remove-leave-repo-out.patch` (sudah teraplikasi di commit `2bdceb5`; `split_generator.py` tidak lagi memuat scenario leave-repo-out)
- [x] P2.6 Simpan manifest split (id baris per split per scenario dan fold) beserta hash-nya (`split_manifest.py`, perintah `build-split-manifest` dan `verify-split-manifest`; 15 manifest di `artifacts/split_manifest/defectors/`; hasil menunggu run)
- [x] P2.7 Bersihkan artefak lama (`20261117_*`, run parsial `20261120_112058`, dan laporan pipeline lama; laporan pipeline `20261117_*` dan `20261120_*` dihapus karena belum memuat `diff_input`, yang tersisa hanya run `20261121_*`)

## P3. Model Training
- [x] P3.1 Baseline TF-IDF + Logistic Regression di 7 scenario
- [x] P3.2 Baseline panjang diff di 7 scenario
- [x] P3.3 Implementasi `CodeBertClassifier` (satu kelas utama per modul, config via object; `codebert_classifier.py`, `classifier_config.py`)
- [x] P3.4 Implementasi `CodeT5SmallClassifier` dengan encoder-decoder penuh (gaya T5ForSequenceClassification; head membaca state decoder di `</s>` terakhir; `codet5_small_classifier.py`)
- [x] P3.5 Loop training bersama: dynamic padding dengan length bucketing, mixed precision (bf16/fp16), gradient accumulation, gradient checkpointing, dukungan class weight (`classifier_trainer.py`, `length_bucket_loader.py`)
- [x] P3.6 Perintah CLI `train` (parser, runner, command) dan smoke test pada subset kecil (`--max-train-samples`; 22 test di `tests/test_training.py` dengan model acak kecil, tanpa unduhan)
- [x] P3.7 Benchmark lokal RTX 3050 6GB untuk kedua model (throughput, VRAM puncak, batch size maksimum, input 512 token) (perintah `benchmark-model` siap; angka GPU menunggu run di mesin RTX 3050)
- [x] P3.8 Susun estimasi total jam untuk 70 run (35 per model) plus tuning dari hasil benchmark, lalu laporkan untuk ditinjau (`run_time_estimate.csv` dihitung otomatis oleh `benchmark-model` dari ukuran train di manifest; jalankan untuk kedua model dan jumlahkan, tuning ditambah di P4.2)
- [x] P3.9 Checkpoint model terbaik dan fitur resume di `artifacts/model/defectors/` (`*_best.pt` dan `*_last.pt`, opsi `--resume`; folder `artifacts/model/*` di-ignore git)
- [x] P3.10 Catat waktu latih dan resource per run (puncak VRAM, throughput) (kolom di `training_results.csv` dan `run_registry.jsonl`)
- [ ] P3.11 Fine-tuning 7 scenario: 3 seed untuk 5 scenario non-fold, 2 seed per fold untuk Group K-Fold, dicatat di registri run (kode, seed bawaan, dan registri `run_registry.jsonl` sudah siap; run penuh menunggu GPU dan hasil P3.7/P3.8)

## P4. Model Validation & Hyperparameter Tuning
- [x] P4.1 Baseline: C dipilih lewat PR-AUC val dan threshold dari val
- [ ] P4.2 Rancang ruang hyperparameter (learning rate, effective batch size, epoch, warmup, weight decay) sesuai budget dari benchmark
- [ ] P4.3 Tuning terpisah per scenario dan simpan konfigurasi terbaiknya
- [ ] P4.4 Early stopping berdasarkan PR-AUC val (val per commit) dan kurva train/val loss untuk cek overfit
- [ ] P4.5 Pilih threshold dari val dengan metode yang sama seperti baseline
- [ ] P4.6 Audit protokol: hyperparameter dibekukan sebelum menyentuh test

## P5. Model Testing & Evaluation
- [x] P5.1 Evaluator PR-AUC, F1, MCC, ROC-AUC beserta hasil per repo (dipakai baseline)
- [ ] P5.2 Evaluasi test CodeBERT dan CodeT5-small di 7 scenario
- [ ] P5.3 Metrik efisiensi: waktu inferensi per sampel, throughput, puncak VRAM, jumlah parameter
- [ ] P5.4 Hasil per repo dan per fold untuk model Transformer
- [ ] P5.5 Metrik pendamping: lift PR-AUC terhadap prevalensi dan F1 relatif terhadap tebak-semua-defective
- [ ] P5.6 Hitung generalization gap (ΔPR-AUC, ΔMCC) dan retention ratio dari within ke cross-domain
- [ ] P5.7 Interval kepercayaan lewat cluster bootstrap per commit dan uji berpasangan antar model (koreksi perbandingan ganda)
- [ ] P5.8 Simpan skor prediksi per baris agar analisis ulang tanpa retraining

## P6. Ablation Study
Menguji kontribusi komponen model atau fitur.
- [ ] P6.1 Ablasi input: diff penuh vs hanya baris tambah vs hanya baris hapus, dengan dan tanpa path file
- [ ] P6.2 Ablasi truncation (hunk-aware vs head vs head+tail), dengan evaluasi terpisah untuk diff ≤ 512 dan > 512 token
- [ ] P6.3 Evaluasi terkontrol panjang (bin panjang diff) untuk memisahkan pemahaman kode dari shortcut ukuran
- [ ] P6.4 Ablasi komponen model: fine-tune penuh vs encoder beku vs sebagian layer beku
- [ ] P6.5 Ablasi fitur baseline: TF-IDF unigram vs n-gram, token vs karakter, dan hibrida TF-IDF + fitur ukuran
- [ ] P6.6 Ablasi ukuran train: versi disamakan ke ukuran train terkecil dibanding hasil utama apa adanya
- [ ] P6.7 Analisis sensitivitas openpilot (`--exclude-repos openpilot`) untuk baseline dan Transformer

## P7. Final Evaluation & Comparison
- [ ] P7.1 Tabel final tiga model plus baseline length di semua scenario (rata-rata ± std dan CI)
- [ ] P7.2 RM1: apakah CodeBERT dan CodeT5-small mengalahkan TF-IDF dan baseline length secara signifikan
- [ ] P7.3 RM2: perbandingan ML vs non-ML pada within-domain dengan kontrol prevalensi dan ukuran data
- [ ] P7.4 RM3: generalisasi lintas domain dua arah (gap dan retention per model)
- [ ] P7.5 RM4: tabel efisiensi komputasi dan trade-off performa terhadap biaya
- [ ] P7.6 Analisis heterogenitas per repo dan error analysis kualitatif (contoh FP/FN menurut panjang diff)
- [ ] P7.7 Rekomendasi model berdasarkan performa dan efisiensi
- [ ] P7.8 Registri run: tiap angka final dapat ditelusuri ke commit, seed, dan konfigurasi

## P8. Reporting & Visualization
- [ ] P8.1 Plot metrik per scenario dan model dengan CI, heatmap domain latih × domain uji, kurva PR dan ROC
- [ ] P8.2 Plot distribusi panjang diff vs label, dot plot per repo, scatter performa vs waktu inferensi
- [ ] P8.3 Perintah CLI `report` yang menggabungkan semua hasil tanpa hardcode dan mengabaikan run lama
- [ ] P8.4 Tabel markdown/LaTeX untuk bab Hasil dan lampiran per-repo
- [ ] P8.5 Perbarui dokumentasi (`command/defectors.md` masih berisi angka lama di checkpoint 4–6) dan panduan reproduksi
- [ ] P8.6 Draf bab Metodologi, Hasil, dan Pembahasan, termasuk threats to validity (pemetaan domain, test `jax` kecil, pergeseran prevalensi, shortcut ukuran, val per commit yang tidak mencerminkan pergeseran domain, satu dataset Python, dua model)
- [ ] P8.7 Abstrak penelitian

## P9. Summary & Conclusion
- [ ] P9.1 Jawaban eksplisit untuk empat rumusan masalah berbasis angka final
- [ ] P9.2 Simpulan tentang generalisasi lintas domain, termasuk jika hasilnya tidak jauh dari baseline length
- [ ] P9.3 Perbarui form/metodologi agar menyebut baseline length yang ditambahkan (form hanya menyebut TF-IDF + Logistic Regression)
- [ ] P9.4 Keterbatasan dan saran penelitian lanjutan (domain adaptation, fitur metadata JIT, bahasa lain)
- [ ] P9.5 Paket reproduksi: tag git, `uv.lock`, hash dataset, manifest split, urutan command