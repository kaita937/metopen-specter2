# Catatan Cicilan 3 Literatur Hari 3 (Standar IEEE — Section I, II, III-B, IV-A, & IV-B)

Seluruh 3 referensi Hari 3 memiliki **kurang dari 6 penulis** (`5`, `3`, dan `2` penulis), sehingga wajib ditulis lengkap seluruh nama penulisnya pada daftar pustaka IEEE.

---

## 1. Thakur et al. (2021 — *BEIR Benchmark*, NeurIPS Datasets and Benchmarks)
- **Sitasi Format IEEE (5 penulis — ditulis lengkap):**
  > N. Thakur, N. Reimers, A. Rücklé, A. Srivastava, and I. Gurevych, "BEIR: A heterogeneous benchmark for zero-shot evaluation of information retrieval models," in *Proc. Neural Inf. Process. Syst. Track Datasets Benchmarks (NeurIPS)*, vol. 1, Dec. 2021, pp. 1–16.
- **Poin Sintesis untuk Section I, II, III-B, & IV-A:**
  - Menetapkan protokol baku evaluasi *zero-shot* pada `SciFact` menggunakan `300` kueri uji (`qrels/test.tsv`) terhadap `5.183` dokumen korpus dengan metrik utama **`nDCG@10`**.
  - Menyediakan angka *baseline* leksikal resmi **BM25 = `0.665` (`nDCG@10`)** pada SciFact, serta menunjukkan bahwa banyak model *dense retrieval* umum yang dilatih pada MS MARCO (seperti DPR atau TAS-B umum) justru gagal mengalahkan BM25 di domain ilmiah/biomedis akibat pergeseran distribusi domain (*out-of-distribution shift*).

## 2. Wang et al. (2022 — *Information Processing & Management*)
- **Sitasi Format IEEE (3 penulis — ditulis lengkap):**
  > X. Wang, C. Macdonald, and I. Ounis, "Improving zero-shot retrieval using dense external expansion," *Inf. Process. Manage.*, vol. 59, no. 5, p. 103026, Sep. 2022.
- **Poin Sintesis untuk Section I, II, IV-A, & IV-D:**
  - Mengkaji kelemahan mendasar evaluasi *zero-shot retrieval*: kesenjangan kosakata (*vocabulary mismatch*) pada model leksikal serta kesenjangan representasi domain (*domain/representation shift*) ketika kueri ringkas dicocokkan langsung dengan dokumen target tanpa konteks yang memadai.
  - Mendukung argumen mengapa pemerkayaan konteks dokumen (menggabungkan `Title + [SEP] + Abstract` pada Konfigurasi C bersama adapter asimetris `adhoc_query` / `proximity`) krusial untuk menjembatani celah semantik antara klaim ilmiah pendek dan dokumen bukti.

## 3. Ballester & Penner (2022 — *Journal of Informetrics*)
- **Sitasi Format IEEE (2 penulis — ditulis lengkap):**
  > O. Ballester and O. Penner, "Robustness, replicability and scalability in topic modelling," *J. Informetrics*, vol. 16, no. 1, p. 101224, Feb. 2022.
- **Poin Sintesis untuk Section II & IV-B (*Statistical Significance & Embedding Stability*):**
  - Membuktikan secara statistik bahwa estimasi kemiripan antar-publikasi ilmiah berbasis *neural network embedding* jauh lebih stabil (*statistically robust*) dan konsisten pada ruang topik berskala besar dibandingkan pemodelan topik statistik tradisional.
  - Menjadi pijakan teoretis saat membahas hasil uji **Paired Wilcoxon Signed-Rank** di **Section IV-B**: ketika input dokumen memuat konteks semantik yang utuh (abstrak dan judul), ruang vektor SPECTER2 menghasilkan estimasi kemiripan kosinus yang stabil dan konsisten lintas ratusan kueri klaim.
