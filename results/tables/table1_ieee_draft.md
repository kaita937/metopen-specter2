# Draft TABLE I & Catatan Literatur Hari 1 (Standar IEEE)

## 1. Draf TABLE I (Siap Masuk Section III-B)

**TABLE I.** DESCRIPTIVE STATISTICS OF SCIFACT QUERIES AND DOCUMENT INPUT CONFIGURATIONS (`allenai/specter2_base` TOKENIZER)

| Input Component / Configuration | $N$ | Empty | Mean $\pm$ SD | Median | Min–Max | $> 512$ Tokens | Truncated (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Test Queries (Scientific Claims) | 300 | 0 | 19.29 $\pm$ 7.10 | 18.0 | 7–47 | 0 | 0.00 |
| Config A (`Title`) | 5,183 | 0 | 19.49 $\pm$ 6.92 | 19.0 | 3–70 | 0 | 0.00 |
| Config B (`Abstract`) | 5,183 | 0 | 278.70 $\pm$ 124.73 | 258.0 | 49–1,834 | 202 | 3.90 |
| Config C (`Title + [SEP] + Abstract`) | 5,183 | 0 | 297.19 $\pm$ 126.80 | 276.0 | 63–1,858 | 245 | 4.73 |

*Catatan tambahan untuk narasi Section III-B:*
- Total pasangan relevan (`score > 0` pada `qrels/test.tsv`): **339 pasangan** pada **300 kueri uji** (rata-rata $1.13 \pm 0.49$ dokumen relevan per kueri; $92.33\%$ kueri memiliki tepat 1 dokumen relevan, rentang 1–5 dokumen).
- Seluruh panjang sekuens dihitung menggunakan `AutoTokenizer.from_pretrained("allenai/specter2_base")` (kosakata SciVocab `31,116` token, dengan token khusus `[CLS]` dan `[SEP]` disertakan agar persis mencerminkan batas pemotongan `max_length = 512` pada tahap *encoding* GPU).
- Rasio rata-rata panjang token **Config B terhadap Config A** adalah **14.30x** (`278.70` berbanding `19.49` token; atau **15.82x** jika mengeluarkan token khusus `[CLS]` dan `[SEP]`).

---

## 2. Catatan Cicilan 3 Literatur Hari 1 & Verifikasi Data Leakage

### A. Wadden et al. (2020) — Karakteristik Kueri Klaim Ilmiah SciFact
- **Sitasi Format IEEE ($\ge 6$ penulis $\rightarrow$ `et al.`):**
  > D. Wadden *et al.*, "Fact or fiction: Verifying scientific claims," in *Proc. Conf. Empirical Methods Natural Lang. Process. (EMNLP)*, Nov. 2020, pp. 7534–7550.
- **Poin Kunci untuk Section I & III-B:**
  - Kueri pada SciFact bukan kueri kata kunci pendek (*keyword search*), melainkan **klaim ilmiah deklaratif padat** (*atomic scientific claims*, rata-rata `19.29` subword token) yang ditulis ulang oleh pakar dari kalimat sitasi jurnal biomedis/sitologi/kesehatan masyarakat.
  - Karakteristik klaim sering memuat arah kausalitas, negasi, mekanisme molekuler, atau besaran kuantitatif yang sering tidak muncul secara leksikal di judul artikel, melainkan tertanam di bagian *methods/results* pada abstrak.

### B. Singh et al. (2023 — SciRepEval) — Arsitektur Adapter SPECTER2 & Audit Data Leakage
- **Sitasi Format IEEE ($< 6$ penulis $\rightarrow$ tulis seluruh 5 penulis):**
  > A. Singh, M. D'Arcy, A. Cohan, D. Downey, and S. Feldman, "SciRepEval: A multi-format benchmark for scientific document representations," in *Proc. Conf. Empirical Methods Natural Lang. Process. (EMNLP)*, Singapore, Dec. 2023, pp. 5548–5566.
- **Poin Kunci Arsitektur untuk Section II & III-C:**
  - Model tunggal statis seperti SPECTER generasi pertama kurang optimal ketika dipaksa menangani beragam format tugas sekaligus. SPECTER2 mengatasi hal ini dengan melatih ulang *backbone* SciBERT (`allenai/specter2_base`) pada **~6,2 juta triplet sitasi** lintas **23 bidang ilmu**, lalu menambahkan modul *task-format-specific adapters* (Pfeiffer adapter pada tiap lapisan Transformer) di atas representasi token `[CLS]` berdimensi `768`.
  - Untuk temu kembali asimetris (*asymmetric ad-hoc retrieval*), klaim pendek di-*encode* menggunakan adapter `allenai/specter2_adhoc_query`, sedangkan dokumen korpus di-*encode* menggunakan adapter `allenai/specter2` (*proximity*), sehingga proyeksi ruang vektor kueri pendek selaras dengan vektor dokumen ilmiah.
- **Hasil Audit Data Leakage (Wajib Dilaporkan Jujur di Section IV-D *Limitations*):**
  1. **Tidak Ada *Supervised Task Leakage* pada Adapter (`Out-of-Train`):** Berdasarkan Tabel 1 pada Singh et al. (2023), benchmark SciRepEval membagi 24 tugas ke dalam himpunan *In-Train* (8 tugas berskala besar untuk melatih adapter) dan *Out-of-Train* (16 tugas *held-out* untuk evaluasi generalisasi *zero-shot*). **SciFact secara eksplisit berada pada kategori *Out-of-Train* (evaluasi *held-out*)**, sehingga pasangan klaim-abstrak maupun `qrels/test.tsv` SciFact **tidak pernah digunakan** dalam pelatihan `specter2_base` maupun adapter `specter2_adhoc_query` / `specter2`.
  2. **Potensi *Unsupervised Corpus Exposure* pada Pre-training Sitasi:** Meskipun bebas dari kebocoran label relevansi klaim, `specter2_base` dilatih di atas graf sitasi Semantic Scholar (~6,2 juta triplet lintas 23 disiplin ilmu) dan diinisialisasi dari SciBERT. Mengingat korpus SciFact (`5.183` abstrak PubMed/S2ORC) dipublikasikan hingga tahun 2020, sebagian teks dokumen korpus dimungkinkan pernah terpapar secara *unsupervised* melalui relasi sitasi antar-artikel selama tahap *pre-training* dasar, tanpa eksposur terhadap klaim uji.

### C. Kostoff et al. (2004) — Landasan Empiris Ablasi Title vs Abstract di Biomedis
- **Sitasi Format IEEE ($< 6$ penulis $\rightarrow$ tulis seluruh 4 penulis):**
  > R. N. Kostoff, J. A. Block, J. A. Stump, and K. M. Pfeil, "Information content in Medline record fields," *Int. J. Med. Inform.*, vol. 73, no. 6, pp. 515–527, Jun. 2004.
- **Perbandingan dengan Hasil Notebook 01 kita (untuk Section I, II, & IV-A):**
  - Studi klasik Kostoff et al. pada rekaman MEDLINE menunjukkan bahwa *field* **Abstract** mengandung hampir satu orde besaran (**~10x lipat**) frasa dan konsep unik lebih banyak dibandingkan *field* **Title** saja.
  - Temuan EDA kita pada SciFact selaras kuat dan bahkan sedikit lebih tajam pada level subword SciBERT: rata-rata panjang token **Config B (`Abstract`) mencapai `278.70` token vs `19.49` token pada Config A (`Title`) — rasio `14.30x` lipat**! Sebaliknya, kekayaan informasi abstrak membawa konsekuensi pemotongan batas `512` token sebesar **`3.90%` (`202` dokumen)** pada Config B dan **`4.73%` (`245` dokumen)** pada Config C, sedangkan Config A memiliki **`0.00%` pemotongan** (maksimum hanya `70` token).
