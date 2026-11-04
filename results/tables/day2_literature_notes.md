# Catatan Cicilan 4 Literatur Hari 2 (Standar IEEE — Section II & III-C)

Seluruh 4 referensi Hari 2 memiliki **kurang dari 6 penulis**, sehingga sesuai aturan ketat template IEEE, **seluruh nama penulis ditulis lengkap** pada daftar pustaka (sedangkan di dalam kalimat narasi tetap dapat dikutip sebagai *Cohan et al.* `[x]` bila $\ge 3$ penulis).

---

## 1. Cohan et al. (2020 — *SPECTER*, ACL)
- **Sitasi Format IEEE (5 penulis — ditulis lengkap):**
  > A. Cohan, S. Feldman, I. Beltagy, D. Downey, and D. S. Weld, "SPECTER: Document-level representation learning using citation-informed transformers," in *Proc. 58th Annu. Meeting Assoc. Comput. Linguistics (ACL)*, Jul. 2020, pp. 2270–2282.
- **Poin Sintesis untuk Section I, II, & III-C:**
  - Memperkenalkan pembelajaran representasi tingkat dokumen berbasis *triplet loss* yang memanfaatkan sinyal keterkaitan sitasi (*citation graph*) di atas *backbone* **SciBERT** (`SciVocab`).
  - Saat waktu inferensi (*test time*), SPECTER **tidak lagi membutuhkan informasi graf sitasi**, melainkan hanya memetakan teks judul dan abstrak melalui vektor token `[CLS]` berdimensi `768`.
  - Keterbatasan SPECTER generasi pertama yang dijembatani oleh SPECTER2 (Singh et al., 2023): satu ruang vektor tunggal statis untuk semua tugas kurang optimal bagi *asymmetric ad-hoc search* (klaim pendek vs dokumen panjang).

## 2. Dinh et al. (2024 — *Expert Systems with Applications*)
- **Sitasi Format IEEE (4 penulis — ditulis lengkap):**
  > T. N. Dinh, P. Pham, G. L. Nguyen, and B. Vo, "Enhancing local citation recommendation with recurrent highway networks and SciBERT-based embedding," *Expert Syst. Appl.*, vol. 243, p. 122911, Jun. 2024.
- **Poin Sintesis untuk Section II & III-C:**
  - Membuktikan bahwa *embedding* kontekstual berbasis **SciBERT** jauh lebih unggul dalam menangkap terminologi teknis, singkatan domain, dan struktur sintaksis literatur ilmiah dibandingkan model BERT umum maupun pencocokan leksikal.
  - Memperkuat landasan mengapa pada `src/text_configs.py` kita mempertahankan teks asli (*case-preserved, no stemming/stopword removal*) ketika diumpankan ke *backbone* `specter2_base`.

## 3. Wang et al. (2024 — *J. King Saud Univ. - Comput. Inf. Sci.*)
- **Sitasi Format IEEE (5 penulis — ditulis lengkap):**
  > Z. Wang, H. Zhang, H. Chen, Y. Feng, and J.-H. Ding, "Content-based quality evaluation of scientific papers using coarse feature and knowledge entity network," *J. King Saud Univ. - Comput. Inf. Sci.*, vol. 36, no. 6, p. 102119, Jul. 2024.
- **Poin Sintesis untuk Section II:**
  - Menunjukkan efektivitas ekstraksi fitur semantik berbasis konten artikel ilmiah (mengandalkan PLM ilmiah seperti SciBERT/SPECTER beserta entitas pengetahuan di dalam teks) ketika sinyal eksternal belum tersedia atau terbatas.
  - Relevan untuk membandingkan kontribusi informasi *coarse-grained* (judul) terhadap representasi entitas ilmiah yang padat di dalam isi abstrak.

## 4. Jiang et al. (2023 — *Journal of Informetrics*)
- **Sitasi Format IEEE (3 penulis — ditulis lengkap):**
  > Z. Jiang, T. Lin, and C. Huang, "Deep representation learning of scientific paper reveals its potential scholarly impact," *J. Informetrics*, vol. 17, no. 1, p. 101376, Feb. 2023.
- **Poin Sintesis untuk Section I, II, & IV-A (*Cold-Start Scenario*):**
  - Menyoroti masalah klasik ***cold-start*** pada artikel ilmiah baru yang belum memiliki riwayat sitasi (*zero citation history*).
  - Membuktikan bahwa pemetaan *deep representation learning* berbasis Transformer ilmiah terhadap **teks judul dan abstrak** mampu membentuk koordinat ruang semantik yang informatif segera saat artikel terbit, mendukung argumen praktis mengapa *dense retrieval* berbasis konten murni (seperti SPECTER2 pada Konfigurasi A/B/C) krusial bagi mesin pencari akademik modern.
