"""Pembentuk teks murni untuk studi ablasi komponen dokumen pada SPECTER2.

Tahap 2:
- Tanpa lowercasing, stemming, atau stopword removal manual karena model berbasis
  Transformer ilmiah (SciBERT / SPECTER2) membutuhkan struktur kalimat dan kapitalisasi
  asli (misal singkatan gen/protein atau satuan dosis biomedis).
- Penanganan field kosong:
  Bila `title` atau `text` bernilai kosong setelah `.strip()`, fungsi tetap mengembalikan
  string apa adanya tanpa melemparkan error maupun menghapus dokumen dari urutan korpus.
  Jumlah dokumen dengan field kosong dicatat secara eksplisit melalui `audit_empty_fields()`.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Tuple


def config_A(doc: Mapping[str, str]) -> str:
    """Konfigurasi A (Title-only): mengembalikan judul dokumen setelah strip spasi tepi.

    Jika `title` kosong, mengembalikan string kosong `""` apa adanya.
    """
    title = doc.get("title", "") or ""
    return title.strip()


def config_B(doc: Mapping[str, str]) -> str:
    """Konfigurasi B (Abstract-only): mengembalikan teks abstrak setelah strip spasi tepi.

    Jika `text` kosong, mengembalikan string kosong `""` apa adanya.
    """
    text = doc.get("text", "") or ""
    return text.strip()


def config_C(doc: Mapping[str, str], sep_token: str) -> str:
    """Konfigurasi C (Title + [SEP] + Abstract): menggabungkan judul dan abstrak dengan token pemisah.

    Menggunakan `tokenizer.sep_token` resmi dari `allenai/specter2_base` tanpa spasi tambahan.
    Jika salah satu atau kedua field (`title` / `text`) kosong, tetap mengembalikan hasil
    penggabungan string apa adanya (misal `"{sep_token}{abstract}"` atau `"{title}{sep_token}"`).
    """
    if not sep_token:
        raise ValueError("Parameter `sep_token` tidak boleh kosong (gunakan tokenizer.sep_token).")
    title = (doc.get("title", "") or "").strip()
    text = (doc.get("text", "") or "").strip()
    return f"{title}{sep_token}{text}"


def audit_empty_fields(corpus: Mapping[str, Mapping[str, str]], sep_token: str = "[SEP]") -> Dict[str, int]:
    """Menghitung jumlah dokumen dengan field kosong pada korpus dan tiap konfigurasi."""
    empty_title = 0
    empty_text = 0
    empty_both = 0
    empty_cfg_a = 0
    empty_cfg_b = 0
    empty_cfg_c = 0

    for doc in corpus.values():
        t_clean = config_A(doc)
        a_clean = config_B(doc)
        c_clean = config_C(doc, sep_token=sep_token)

        is_t_empty = len(t_clean) == 0
        is_a_empty = len(a_clean) == 0

        if is_t_empty:
            empty_title += 1
            empty_cfg_a += 1
        if is_a_empty:
            empty_text += 1
            empty_cfg_b += 1
        if is_t_empty and is_a_empty:
            empty_both += 1
        if len(c_clean) == 0 or c_clean == sep_token:
            empty_cfg_c += 1

    return {
        "empty_title_count": empty_title,
        "empty_abstract_count": empty_text,
        "empty_both_count": empty_both,
        "empty_config_A": empty_cfg_a,
        "empty_config_B": empty_cfg_b,
        "empty_config_C_effective": empty_cfg_c,
    }


def build_all_configs(
    corpus: Mapping[str, Mapping[str, str]],
    sep_token: str,
) -> Tuple[List[str], Dict[str, List[str]]]:
    """Membangun urutan `doc_ids` yang identik beserta daftar teks untuk Konfigurasi A, B, dan C."""
    doc_ids: List[str] = list(corpus.keys())
    texts_by_config: Dict[str, List[str]] = {
        "A": [config_A(corpus[did]) for did in doc_ids],
        "B": [config_B(corpus[did]) for did in doc_ids],
        "C": [config_C(corpus[did], sep_token=sep_token) for did in doc_ids],
    }
    return doc_ids, texts_by_config
