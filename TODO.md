**Checkpoint 1 — Dataset integrity**
**Yang diperiksa:**
- File Parquet dapat dibaca.
- Tidak ada file yang corrupt atau kosong.
- Ukuran dan jumlah record masuk akal.
**Probe signal:**
- Jumlah record mendekati angka publikasi: **213.419**.
- Terdapat sekitar **24 repository**.
- Kolom utama tersedia: `datetime`, `commit`, `repo`, `filepath`, `content`, `methods`, `induce_bug`.
Angka publikasi merupakan acuan, bukan syarat harus identik sebelum mengetahui varian dan struktur file yang kamu download.&#x20;
**Checkpoint 2 — Label validity**
**Yang diperiksa:**
- Apakah `induce_bug` tersedia dan dapat dibaca?
- Berapa record dengan `induce_bug` kosong dan tidak kosong?
- Apakah label dapat diturunkan secara konsisten?
**Interpretasi label:**
| Kondisi `induce_bug`    | Label           |
| ----------------------- | --------------- |
| List kosong             | 0 — defect-free |
| List berisi nomor baris | 1 — defective   |
**Probe signal:**
- Kedua kelas tersedia.
- Tidak ada label null yang tidak dapat dijelaskan.
- Distribusi label dihitung pada tingkat record dan commit.
Penting: Defectors menggunakan SZZ untuk mengidentifikasi bug-inducing commits. Penulis melakukan beberapa tahap filtering, tetapi tetap mengakui kemungkinan noise pada label.&#x20;
**Checkpoint 3 — JIT input validity**
**Yang diperiksa:**
- Apakah `content` benar-benar berisi Git diff?
- Apakah diff memiliki perubahan kode, bukan sekadar teks kosong?
- Apakah input dapat diproses tokenizer CodeBERT dan CodeT5-small?
- Berapa banyak input yang terlalu panjang?
**Probe signal:**
- Mayoritas record memiliki diff yang valid.
- Diff tidak didominasi record kosong atau malformed.
- Panjang token masih memungkinkan untuk diproses dengan batas GPU yang tersedia.
Catatan: Defectors menggunakan `content` untuk dua tugas berbeda. Untuk JIT-SDP, isinya adalah commit diff; untuk file-level defect prediction, isinya adalah konten file setelah commit. Jangan sampai salah menggunakan representasi file-level untuk eksperimen JIT-SDP.&#x20;
**Checkpoint 4 — Repository dan domain validity**
**Yang diperiksa:**
- Jumlah record per repository.
- Jumlah commit unik per repository.
- Distribusi defective dan defect-free per repository.
- Pengelompokan ML/DL dan non-ML.
**Probe signal:**
- Terdapat beberapa repository yang valid pada masing-masing domain.
- Kedua domain memiliki sampel positif dan negatif.
- Tidak ada domain yang hanya diwakili satu repository.
- Distribusi tidak terlalu terkonsentrasi pada satu repository.
**Contoh kandidat klasifikasi awal:**
| ML/DL                     | Non-ML         |
| ------------------------- | -------------- |
| Lightning                 | Ansible        |
| JAX                       | Django         |
| Hugging Face Transformers | Airflow        |
| Ray                       | Sentry         |
| Scikit-learn              | Celery         |
| YOLOv5                    | Home Assistant |
Ini hanya klasifikasi awal berdasarkan domain yang dicantumkan dalam paper. Kita perlu menetapkan aturan klasifikasi secara konsisten sebelum eksperimen. Misalnya, proyek data science seperti NumPy dan pandas tidak otomatis dikategorikan sebagai ML/DL.
**Checkpoint 5 — Leakage dan unit analisis**
Kritis
Ini checkpoint yang paling penting untuk validitas penelitian.
Defectors memiliki data pada tingkat file-change. Satu commit dapat mengubah beberapa file, sehingga satu commit dapat muncul pada beberapa record.
**Yang diperiksa:**
- Duplikasi `repo + commit + filepath`.
- Jumlah file-change per commit.
- Jumlah commit unik.
- Apakah satu commit memiliki label yang konsisten.
- Apakah repository yang sama masuk train dan test.
**Probe signal:**
- Split train/test berdasarkan repository.
- Tidak ada commit yang bocor antara train dan test.
- Label commit ditentukan dengan aturan agregasi yang jelas.
- Jika menggunakan commit-level prediction, seluruh diff file dalam commit harus diagregasi menjadi satu input.
Contoh agregasi label commit:
\\[ y\_{commit}=\max(y\_{file_1},y\_{file_2},...,y\_{file_n}) \\]
Artinya, commit dianggap defective jika minimal satu file-change terkait memiliki label defective.
Jangan menganggap setiap baris sebagai commit independen.
**Checkpoint 6 — Cross-domain feasibility**
Penentu kelanjutan
**Yang diperiksa:**
- Jumlah repository ML/DL dan non-ML.
- Distribusi kelas pada masing-masing domain.
- Jumlah commit unik setelah agregasi.
- Ketersediaan repository untuk train dan test yang terpisah.
**Probe signal:**
- Kedua domain memiliki beberapa repository.
- Masing-masing domain memiliki cukup sampel positif dan negatif.
- Skenario ML → non-ML dan non-ML → ML dapat dijalankan tanpa repository overlap.
- Within-domain evaluation juga memungkinkan menggunakan repository holdout.
Tidak ada angka minimum universal untuk kelayakan ini. Yang penting, setiap skenario memiliki data train dan test yang independen serta tidak kehilangan salah satu kelas.
## **3. Minimum acceptance criteria**
Ini adalah kriteria operasional yang saya sarankan untuk skripsimu, bukan standar resmi dari pembuat Defectors.
| Aspek             | Minimum agar lanjut                                     |
| ----------------- | ------------------------------------------------------- |
| Dataset integrity | Parquet terbaca dan struktur data terverifikasi         |
| Label             | Ada label 0 dan 1, tanpa null yang tidak terjelaskan    |
| Input             | `content` berisi diff yang dapat diproses               |
| Repository        | Identitas repository tersedia                           |
| Domain            | Ada minimal 3 repository per domain sebagai target awal |
| Sample            | Kedua kelas tersedia di train dan test                  |
| Leakage           | Tidak ada repository overlap antar-split                |
| Unit analisis     | File-change atau commit ditentukan secara eksplisit     |
| Tokenization      | Kedua model berhasil memproses sampel uji               |
| GPU               | Satu eksperimen pilot berhasil tanpa out-of-memory      |
## **4. Probe signal yang perlu kita hasilkan**
Saya ingin audit pertama menghasilkan satu laporan seperti ini:
| Probe                   | Output yang dibutuhkan                         |
| ----------------------- | ---------------------------------------------- |
| Dataset overview        | Total rows, columns, file size                 |
| Repository distribution | Rows dan commits per repository                |
| Label distribution      | Defective vs defect-free                       |
| Domain distribution     | ML/DL vs non-ML                                |
| Commit analysis         | Unique commits dan files per commit            |
| Duplicate analysis      | Duplicate rows dan duplicate diffs             |
| Content quality         | Empty diff, malformed diff, diff length        |
| Token analysis          | Token length P50, P90, P95, P99                |
| Label consistency       | Konsistensi label antar-file dalam satu commit |
| Split feasibility       | Jumlah repository dan sampel pada setiap split |
## **5. Urutan pengerjaan**
Jangan langsung fine-tuning model. Kerjakan bertahap:
1. **Dataset inventory**
   Identifikasi struktur folder, file Parquet, YAML, ukuran, dan metadata.
2. **Data profiling**
   Hitung distribusi record, label, repository, commit, dan kualitas `content`.
3. **Domain mapping**
   Tetapkan daftar repository ML/DL dan non-ML beserta alasan klasifikasinya.
4. **JIT feasibility**
   Pastikan label, diff, dan unit analisis benar.
5. **Split feasibility**
   Simulasikan repository-level split untuk seluruh skenario eksperimen.
6. **Pilot experiment**
   Jalankan baseline sederhana dan satu batch tokenization CodeBERT/CodeT5-small.