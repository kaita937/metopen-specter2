"""Pengelompokan otomatis 4 kategori Error Analysis kualitatif (Tahap 6 — Hari 4).

Memindai hasil evaluasi 300 kueri uji dan mengelompokkannya ke dalam 4 keranjang bukti empiris:
1. Kategori 1: C/B Berhasil (rank <= 10), A Gagal (rank > 10) -> bukti informasi krusial di abstrak.
2. Kategori 2: A Lebih Unggul dari B dan C (rank_A < rank_B dan rank_A < rank_C) -> bukti dilusi/pemotongan abstrak.
3. Kategori 3: Gagal di Ketiga Konfigurasi (rank > 10 di A, B, dan C) -> batas zero-shot SPECTER2.
4. Kategori 4: Berhasil Peringkat Atas di C (rank_C <= 3) meski Lexical Overlap sangat rendah -> bukti semantik.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Set, Tuple

import pandas as pd

TOKEN_RE = re.compile(r"[a-z0-9]+")

# Stopwords bahasa Inggris dasar untuk mengukur tumpang-tindih kata bermakna (content words)
STOPWORDS: Set[str] = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "in", "into", "is", "it", "its", "of", "on", "or", "that", "the", "their",
    "there", "these", "this", "to", "was", "were", "with", "within", "without",
    "can", "may", "than", "between", "among", "during", "after", "before", "both",
}


def tokenize_words(text: str, remove_stopwords: bool = True) -> Set[str]:
    """Mengekstrak himpunan kata unik alfanumerik huruf kecil."""
    tokens = set(TOKEN_RE.findall((text or "").lower()))
    if remove_stopwords:
        tokens = {t for t in tokens if t not in STOPWORDS and len(t) >= 2}
    return tokens


def jaccard_similarity(text_a: str, text_b: str, remove_stopwords: bool = True) -> Tuple[float, List[str]]:
    """Menghitung Jaccard overlap kata unik beserta daftar kata yang beririsan."""
    set_a = tokenize_words(text_a, remove_stopwords=remove_stopwords)
    set_b = tokenize_words(text_b, remove_stopwords=remove_stopwords)
    if not set_a and not set_b:
        return 0.0, []
    inter = sorted(set_a.intersection(set_b))
    union_len = len(set_a.union(set_b))
    return (len(inter) / float(union_len) if union_len > 0 else 0.0), inter


def make_abstract_excerpt(abstract: str, max_chars: int = 280) -> str:
    """Membuat potongan ringkas abstrak yang rapi untuk tabel analisis kualitatif."""
    clean = " ".join((abstract or "").strip().split())
    if len(clean) <= max_chars:
        return clean
    clipped = clean[:max_chars].rsplit(" ", 1)[0]
    return f"{clipped}..."


def run_error_analysis(
    per_query_df: pd.DataFrame,
    corpus: Mapping[str, Mapping[str, str]],
    test_queries: Mapping[str, Mapping[str, str]],
    doc_token_lengths_c: Mapping[str, int] | None = None,
    output_dir: str | Path | None = None,
) -> Dict[str, Any]:
    """Mengklasifikasikan 300 kueri ke 4 kategori dan memilih 1 studi kasus terbaik per kategori."""
    enriched_rows: List[Dict[str, Any]] = []

    for _, row in per_query_df.iterrows():
        qid = str(row["query_id"])
        q_text = test_queries[qid]["text"].strip()

        # Gunakan dokumen relevan utama pada Config C sebagai acuan teks dokumen ground-truth
        target_cid = str(row["best_rel_doc_C"])
        doc = corpus[target_cid]
        doc_title = (doc.get("title") or "").strip()
        doc_abs = (doc.get("text") or "").strip()
        full_doc_text = f"{doc_title} {doc_abs}".strip()

        jacc_content, shared_words = jaccard_similarity(q_text, full_doc_text, remove_stopwords=True)
        jacc_raw, _ = jaccard_similarity(q_text, full_doc_text, remove_stopwords=False)

        rank_a = int(row["best_rank_A"])
        rank_b = int(row["best_rank_B"])
        rank_c = int(row["best_rank_C"])

        score_a = float(row["best_score_A"])
        score_b = float(row["best_score_B"])
        score_c = float(row["best_score_C"])

        ndcg_a = float(row["ndcg10_A"])
        ndcg_b = float(row["ndcg10_B"])
        ndcg_c = float(row["ndcg10_C"])

        cat1_cb_ok_a_fail = (rank_c <= 10 or rank_b <= 10) and (rank_a > 10)
        cat2_a_beats_bc = (rank_a < rank_b) and (rank_a < rank_c)
        cat3_all_fail = (rank_a > 10) and (rank_b > 10) and (rank_c > 10)
        cat4_low_lex_top3 = (rank_c <= 3) and (jacc_content <= 0.12)

        tok_c = int(doc_token_lengths_c.get(target_cid, 0)) if doc_token_lengths_c else 0

        enriched_rows.append({
            "query_id": qid,
            "query_text": q_text,
            "rel_doc_id": target_cid,
            "doc_title": doc_title,
            "doc_abstract_excerpt": make_abstract_excerpt(doc_abs, max_chars=280),
            "doc_tokens_C": tok_c,
            "rank_A": rank_a,
            "rank_B": rank_b,
            "rank_C": rank_c,
            "score_A": f"{score_a:.4f}",
            "score_B": f"{score_b:.4f}",
            "score_C": f"{score_c:.4f}",
            "ndcg10_A": f"{ndcg_a:.4f}",
            "ndcg10_B": f"{ndcg_b:.4f}",
            "ndcg10_C": f"{ndcg_c:.4f}",
            "jaccard_content": round(jacc_content, 4),
            "jaccard_raw": round(jacc_raw, 4),
            "shared_content_words": ", ".join(shared_words) if shared_words else "(none)",
            "is_cat1_cb_success_a_fail": cat1_cb_ok_a_fail,
            "is_cat2_a_beats_bc": cat2_a_beats_bc,
            "is_cat3_all_fail": cat3_all_fail,
            "is_cat4_semantic_low_overlap": cat4_low_lex_top3,
        })

    df_all = pd.DataFrame(enriched_rows)

    # Ringkasan populasi tiap keranjang dari 300 kueri uji
    n_total = len(df_all)
    cat_summary_df = pd.DataFrame([
        {
            "category_id": "Category 1",
            "category_label": "C/B Succeeds (rank <= 10), A Fails (rank > 10)",
            "query_count": int(df_all["is_cat1_cb_success_a_fail"].sum()),
            "percentage_of_300": f"{df_all['is_cat1_cb_success_a_fail'].mean() * 100.0:.2f}",
        },
        {
            "category_id": "Category 2",
            "category_label": "A Outperforms B & C (rank_A < rank_B and rank_A < rank_C)",
            "query_count": int(df_all["is_cat2_a_beats_bc"].sum()),
            "percentage_of_300": f"{df_all['is_cat2_a_beats_bc'].mean() * 100.0:.2f}",
        },
        {
            "category_id": "Category 3",
            "category_label": "All Configurations Fail (rank > 10 in A, B, and C)",
            "query_count": int(df_all["is_cat3_all_fail"].sum()),
            "percentage_of_300": f"{df_all['is_cat3_all_fail'].mean() * 100.0:.2f}",
        },
        {
            "category_id": "Category 4",
            "category_label": "Top-3 in C (rank_C <= 3) Despite Minimal Lexical Overlap (Jaccard <= 0.12)",
            "query_count": int(df_all["is_cat4_semantic_low_overlap"].sum()),
            "percentage_of_300": f"{df_all['is_cat4_semantic_low_overlap'].mean() * 100.0:.2f}",
        },
    ])

    used_qids: Set[str] = set()
    selected_records: List[Dict[str, Any]] = []

    # 1. Pilih studi kasus terbaik Kategori 1: rank_C == 1, rank_B <= 3, rank_A jatuh jauh tapi bukan outlier ekstrem tak relevan
    c1_pool = df_all[df_all["is_cat1_cb_success_a_fail"]].copy()
    c1_pool = c1_pool.sort_values(by=["rank_C", "rank_B", "rank_A"], ascending=[True, True, False])
    c1_best = c1_pool.iloc[0].to_dict()
    used_qids.add(str(c1_best["query_id"]))
    selected_records.append({
        "category_id": "Cat. 1",
        "category_phenomenon": "C/B Succeeds (<=10), A Fails (>10): Crucial Evidence in Abstract",
        **c1_best,
    })

    # 2. Pilih studi kasus terbaik Kategori 2: rank_A == 1 (atau terkecil) sementara rank_B & rank_C turun jauh
    c2_pool = df_all[df_all["is_cat2_a_beats_bc"] & (~df_all["query_id"].isin(used_qids))].copy()
    c2_pool["rank_gap_c_minus_a"] = c2_pool["rank_C"] - c2_pool["rank_A"]
    c2_pool = c2_pool.sort_values(by=["rank_A", "rank_gap_c_minus_a", "doc_tokens_C"], ascending=[True, False, False])
    c2_best = c2_pool.iloc[0].to_dict()
    c2_best.pop("rank_gap_c_minus_a", None)
    used_qids.add(str(c2_best["query_id"]))
    selected_records.append({
        "category_id": "Cat. 2",
        "category_phenomenon": "A Outperforms B & C: Focused Title vs. Long Abstract Dilution",
        **c2_best,
    })

    # 3. Pilih studi kasus terbaik Kategori 3: Gagal di ketiga konfigurasi (rank > 10 di A, B, C)
    c3_pool = df_all[df_all["is_cat3_all_fail"] & (~df_all["query_id"].isin(used_qids))].copy()
    # Pilih kasus kegagalan moderat-berat (rank_C antara 25 s.d. 300) yang representatif untuk dibedah
    c3_mod = c3_pool[(c3_pool["rank_C"] >= 25) & (c3_pool["rank_C"] <= 350)]
    if len(c3_mod) == 0:
        c3_mod = c3_pool
    c3_mod = c3_mod.sort_values(by=["rank_C", "rank_A"], ascending=[False, False])
    c3_best = c3_mod.iloc[0].to_dict()
    used_qids.add(str(c3_best["query_id"]))
    selected_records.append({
        "category_id": "Cat. 3",
        "category_phenomenon": "All Configs Fail (>10): Fine-Grained Claim / Zero-Shot Limitation",
        **c3_best,
    })

    # 4. Pilih studi kasus terbaik Kategori 4: rank_C == 1 (atau <= 3) dengan Jaccard overlap kata bermakna terendah
    c4_pool = df_all[(df_all["rank_C"] <= 3) & (~df_all["query_id"].isin(used_qids))].copy()
    c4_pool = c4_pool.sort_values(by=["jaccard_content", "rank_C", "jaccard_raw"], ascending=[True, True, True])
    c4_best = c4_pool.iloc[0].to_dict()
    used_qids.add(str(c4_best["query_id"]))
    selected_records.append({
        "category_id": "Cat. 4",
        "category_phenomenon": "Top-3 in C Despite Minimal Lexical Overlap: Semantic Bridging",
        **c4_best,
    })

    selected_df = pd.DataFrame(selected_records)
    # Format kolom jaccard dengan nol di depan desimal sesuai standar IEEE
    selected_df["jaccard_content"] = selected_df["jaccard_content"].map(lambda v: f"{float(v):.4f}")
    selected_df["jaccard_raw"] = selected_df["jaccard_raw"].map(lambda v: f"{float(v):.4f}")

    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        df_export = df_all.copy()
        df_export["jaccard_content"] = df_export["jaccard_content"].map(lambda v: f"{float(v):.4f}")
        df_export["jaccard_raw"] = df_export["jaccard_raw"].map(lambda v: f"{float(v):.4f}")
        df_export.to_csv(out_path / "all_queries_categorized.csv", index=False)
        cat_summary_df.to_csv(out_path / "category_summary.csv", index=False)
        selected_df.to_csv(out_path / "selected_cases.csv", index=False)

    return {
        "all_categorized_df": df_all,
        "category_summary_df": cat_summary_df,
        "selected_cases_df": selected_df,
    }
