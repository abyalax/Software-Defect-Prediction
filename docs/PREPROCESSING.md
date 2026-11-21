# Prapemrosesan Data / Data Preprocessing

Dokumen ini mencatat seluruh langkah prapemrosesan (preprocessing) dari data mentah Defectors sampai input model
CodeBERT dan CodeT5-small. Setiap angka parameter di bawah berasal dari `DatasetConfig` dan `DiffInputConfig`
di `src/software_defects/config.py`, sehingga dokumen ini dan kode tidak boleh berbeda. Ringkasan jumlah
repositori dan rekaman ada di `docs/DATASET_SCOPE.md`.

## 1. Alur Ringkas / Overview

```
JIT parquet (train + test + val, 213,420 baris)
  │  1. filter_data            buang freqtrade dan diff kosong            → 210,156 baris
  │  2. label biner            label = 1 jika kolom `lines` tidak kosong
  │  3. split                  tujuh scenario, val per commit dari sisi train
  │  4. pre-cut                diff > 16,000 karakter dipotong hunk-aware per karakter
  │  5. path + diff            baris pertama = path file, sisanya = diff
  │  6. truncation hunk-aware  hanya jika input > 512 token (token khusus ikut dihitung)
  │  7. tokenisasi             tokenizer CodeBERT dan CodeT5-small, max_length 512
  ▼
parquet per model: model_input, token_count, truncated, dan metadata preprocessing
```

Langkah 1–3 sudah ada sebelum P1.4 (lihat `split_generator.py`). Langkah 4–7 ditambahkan oleh P1.4–P1.7.

## 2. Unit Analisis dan Label / Unit of Analysis and Label

- Unit analisis adalah satu perubahan file (file-level): satu baris = satu pasangan `commit` + `filepath`.
- `label = 1` jika `lines` (nomor baris buggy) tidak kosong, selain itu `label = 0`.
- Baris dengan diff kosong dibuang karena semuanya `label = 0`, sehingga diff kosong akan membocorkan label.

## 3. Input Model: Path + Diff / Model Input

Setiap input berbentuk:

```
<path file>\n<diff>
```

- Path selalu menjadi baris pertama dan tidak pernah dipotong oleh truncation diff. Budget token untuk diff
  dihitung setelah dikurangi token khusus model (jumlahnya diambil dari `num_special_tokens_to_add` milik
  tokenizer, bukan angka tetap) dan token path beserta satu token newline.
- Path yang lebih panjang dari `path_max_tokens` (64 token) dipendekkan dari kiri per direktori dan diawali
  `.../`; nama file selalu dipertahankan. Kolom `path_shortened` mencatat kasus ini dan hasil verifikasi
  melaporkan persentasenya.
- `include_path=False` membuang path. Opsi ini hanya untuk ablasi input (P6.1), bukan untuk eksperimen utama.
- Path ikut menjadi input karena probing P0.6 menunjukkan sinyalnya lemah (lihat
  `path_metadata_signal_auc.csv` dan `path_text_probe.csv` pada run checkpoint 15), sehingga dampaknya terhadap
  hasil dapat diukur lewat ablasi, bukan diasumsikan.

## 4. Pre-cut Diff Ekstrem / Extreme Diff Pre-cut

Diff yang lebih panjang dari `pre_cut_chars` (16,000 karakter, sekitar persentil 99 dari panjang diff) dipotong
terlebih dahulu dengan algoritma hunk-aware yang sama, tetapi dengan biaya berupa jumlah karakter. Tujuannya
hanya kecepatan: tokenisasi dan pemilihan baris tidak perlu memproses diff berukuran jutaan karakter. Baris
yang ditandai `pre_cut = True` selalu juga `truncated = True`, dan `full_tokens` untuk baris tersebut adalah
batas bawah (lower bound), bukan panjang diff aslinya.

## 5. Truncation Hunk-aware / Hunk-aware Truncation

Truncation hanya berjalan jika `path + diff` melebihi 512 token. Diff yang muat tidak diubah sama sekali.
Untuk diff yang tidak muat, `HunkAwareTruncator` (`diff_truncator.py`) bekerja berurutan:

1. **Header hunk dipertahankan.** Setiap hunk yang diterima menyimpan baris `@@ ... @@`. Seluruh header boleh
   memakai paling banyak `header_share` (50%) dari budget. Jika hunk terlalu banyak, hunk dari belakang dibuang
   dan jumlahnya tercatat (`hunks_kept` < `hunks_total`).
2. **Baris +/- diprioritaskan.** Sisa budget dibagi antar hunk dengan max-min fair share: hunk kecil menyimpan
   semua perubahannya, hunk besar tidak bisa menghabiskan budget hunk lain. Di dalam satu hunk, baris perubahan
   diambil menurut urutan asli. Satu baris perubahan yang lebih panjang dari jatah hunk dipotong secara
   proporsional (clipping) agar hunk tidak kosong.
3. **Sisa budget.** Dipakai untuk baris perubahan yang tidak muat di jatah hunknya, lalu baris konteks yang
   paling dekat dengan baris perubahan yang dipertahankan.
4. **Urutan hunk dan urutan baris tidak berubah.** Tidak ada penanda penghilangan yang ditambahkan, jadi
   budget token tidak terpakai untuk penanda.

Biaya sebuah baris adalah jumlah token baris itu ditambah satu token newline. Jumlah token per baris hanya
perkiraan atas token teks utuh (BPE dapat menggabungkan token melewati batas baris), maka sesudah perakitan
teks dihitung ulang dengan tokenizer. Jika masih melebihi batas, budget diperketat sebesar kelebihannya dan
pemilihan diulang sampai `max_refits` (4) kali. Jika tetap gagal, `fallback_truncation = True` dan truncation
bawaan tokenizer menjadi jaring pengaman terakhir; karena path ada di awal, path tetap utuh. Verifikasi
(bagian 7) harus melaporkan nol kasus fallback.

## 6. Kolom Hasil / Output Columns

Parquet per model (`{scenario}/{codebert|codet5}/{scenario}_{train|val|test}.parquet`) menambahkan kolom berikut
pada kolom asli dan `label`:

| Kolom                           | Arti                                                                             |
| ------------------------------- | -------------------------------------------------------------------------------- |
| `model_input`                   | teks `path + diff` yang dimasukkan ke tokenizer (fine-tuning memakai kolom ini)  |
| `token_count`                   | panjang akhir dalam token, token khusus ikut dihitung, tidak pernah melebihi 512 |
| `truncated`                     | `True` jika ada baris atau hunk yang dibuang atau dipotong (termasuk pre-cut)    |
| `pre_cut`                       | `True` jika diff dipotong oleh pre-cut                                           |
| `fallback_truncation`           | `True` jika jaring pengaman tokenizer terpakai (seharusnya selalu `False`)       |
| `path_shortened`                | `True` jika path dipendekkan                                                     |
| `full_tokens`                   | panjang token sebelum truncation token-level (batas bawah jika `pre_cut`)        |
| `hunks_total`, `hunks_kept`     | jumlah hunk pada diff dan yang headernya dipertahankan                           |
| `changed_total`, `changed_kept` | jumlah baris +/- pada diff dan yang dipertahankan                                |

Catatan: sebelum P1.4, kolom `truncated` dihitung dari `token_count > max_length` setelah tokenizer sendiri
memotong ke `max_length`, sehingga nilainya selalu `False` dan laporan pipeline lama selalu menampilkan
Truncated 0%. Angka itu tidak boleh dipakai di skripsi. Persentase terpotong yang benar berasal dari kolom di
atas dan dari laporan `verify-tokenization`.

## 7. Verifikasi / Verification

`verify-tokenization` membaca parquet hasil pipeline, meng-encode ulang `model_input` dengan tokenizer model, dan
memeriksa per scenario, fold, model, dan split:

| Pemeriksaan            | Syarat lulus                                                                           |
| ---------------------- | -------------------------------------------------------------------------------------- |
| `over_limit`           | tidak ada input lebih dari 512 token                                                   |
| `special_token_errors` | setiap input diawali dan diakhiri token khusus model                                   |
| `path_not_intact`      | path (tidak dipendekkan) adalah baris pertama dan token path menjadi awal urutan token |
| `token_count_mismatch` | `token_count` tersimpan sama dengan hasil encode ulang                                 |

Laporan juga memuat persentase `truncated`, `pre_cut`, `fallback`, `path_shortened`, persentase baris +/- yang
dipertahankan (`changed_kept_pct`), dan rata-rata token. Perintah keluar dengan kode 1 jika ada pemeriksaan yang
gagal. Unit test (`tests/test_diff_preprocessing.py`) memakai tokenizer BPE kecil bergaya RoBERTa yang dilatih
secara lokal; test dengan tokenizer asli (`microsoft/codebert-base`, `Salesforce/codet5-small`) otomatis
berjalan jika kedua tokenizer sudah ada di cache lokal dan dilewati jika tidak.

## 8. Keterbatasan / Limitations

- Hunk di luar jatah header (lebih dari separuh budget) dibuang dari belakang, jadi perubahan pada hunk
  terakhir diff yang sangat banyak hunk-nya tidak terlihat model.
- Baris konteks dan baris +/- yang dibuang tidak ditandai pada input, sehingga model tidak tahu bahwa ada
  bagian yang hilang.
- Pre-cut berbasis karakter hanya perkiraan kasar terhadap biaya token; dampaknya terbatas pada sekitar 1% diff.
- Perhitungan token per baris tidak persis sama dengan token teks utuh, maka ada langkah verifikasi ulang
  dan refit; jarang sekali fallback diperlukan, dan jumlahnya dilaporkan.
- Hasil di atas bergantung pada versi tokenizer. Versi `transformers` ada di `uv.lock`.

## 9. Reproduksi / Reproduction

```Shell
uv run python scripts/download_tokenizers.py
uv run python -m software_defects run-pipeline --dataset defectors --scenario Repository_Level_Split
uv run python -m software_defects verify-tokenization --dataset defectors --scenario Repository_Level_Split
```

Ulangi untuk keenam scenario lain (lihat `command/defectors.md`). Laporan tersimpan di
`artifacts/report/defectors/data_pipeline/DefectorsDataPipeline/<timestamp>/` dan
`artifacts/report/defectors/data_pipeline/TokenizationVerifier/<timestamp>/`.
