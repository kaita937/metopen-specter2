"""Modul pencari kemiripan vektor (Cosine Similarity) dan pemeringkatan dokumen (Tahap 4 — Hari 3).

Mengimplementasikan:
- Normalisasi L2 baris matriks kueri dan dokumen:
  Q_norm = Q / np.clip(np.linalg.norm(Q, axis=1, keepdims=True), 1e-12, None)
- Perkalian matriks kemiripan kosinus: scores = Q_norm @ D_norm.T -> (300, 5183)
- Ekstraksi kandidat top-k beserta posisi peringkat eksak (1-based) dokumen relevan.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Tuple

import numpy as np


def l2_normalize(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Melakukan normalisasi L2 per baris vektor secara stabil numerik (float64)."""
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Matriks input harus 2D (N, D), diterima shape={arr.shape}")
    norms = np.clip(np.linalg.norm(arr, axis=1, keepdims=True), eps, None)
    return arr / norms


def compute_cosine_scores(query_emb: np.ndarray, doc_emb: np.ndarray) -> np.ndarray:
    """Menghitung matriks Cosine Similarity berukuran (num_queries, num_docs)."""
    q_norm = l2_normalize(query_emb)
    d_norm = l2_normalize(doc_emb)
    return q_norm @ d_norm.T


def rank_all_queries(scores: np.ndarray) -> np.ndarray:
    """Mengurutkan indeks dokumen dari skor tertinggi ke terendah per kueri.

    Menggunakan mergesort stabil agar penentuan peringkat deterministik.
    """
    return np.argsort(-scores, axis=1, kind="mergesort")


def extract_top_k(
    scores: np.ndarray,
    doc_ids: List[str],
    top_k: int = 10,
) -> List[List[Tuple[str, float, int]]]:
    """Mengekstrak daftar top-k (doc_id, cosine_score, rank_1based) untuk setiap kueri."""
    ranked_indices = rank_all_queries(scores)[:, :top_k]
    results: List[List[Tuple[str, float, int]]] = []
    for q_idx in range(scores.shape[0]):
        row_items: List[Tuple[str, float, int]] = []
        for pos, d_idx in enumerate(ranked_indices[q_idx], start=1):
            row_items.append((doc_ids[int(d_idx)], float(scores[q_idx, int(d_idx)]), pos))
        results.append(row_items)
    return results


def get_relevant_ranks_per_query(
    scores: np.ndarray,
    query_ids: List[str],
    doc_ids: List[str],
    qrels: Mapping[str, Mapping[str, int]],
) -> Dict[str, Dict[str, Dict[str, float | int]]]:
    """Mencatat peringkat eksak (1..5183) dan skor kosinus tiap dokumen relevan per kueri.

    Sangat berguna untuk evaluasi metrik sekaligus pengelompokan otomatis Error Analysis (Tahap 6).
    """
    ranked_indices = rank_all_queries(scores)
    doc_id_to_col = {did: idx for idx, did in enumerate(doc_ids)}

    rel_info: Dict[str, Dict[str, Dict[str, float | int]]] = {}
    for q_row, qid in enumerate(query_ids):
        order_row = ranked_indices[q_row]
        # Peta kolom dokumen -> peringkat 1-based
        col_to_rank = np.empty(len(doc_ids), dtype=np.int32)
        col_to_rank[order_row] = np.arange(1, len(doc_ids) + 1, dtype=np.int32)

        q_rels = qrels.get(qid, {})
        q_dict: Dict[str, Dict[str, float | int]] = {}
        for cid, rel_grade in q_rels.items():
            if rel_grade <= 0:
                continue
            col_idx = doc_id_to_col[cid]
            q_dict[cid] = {
                "rank": int(col_to_rank[col_idx]),
                "score": float(scores[q_row, col_idx]),
                "rel": int(rel_grade),
            }
        rel_info[qid] = q_dict
    return rel_info


def build_pytrec_run_dict(
    scores: np.ndarray,
    query_ids: List[str],
    doc_ids: List[str],
    top_k: int,
) -> Dict[str, Dict[str, float]]:
    """Membangun dictionary `run` dengan tepat `top_k` kandidat teratas per kueri untuk pytrec_eval."""
    ranked_indices = rank_all_queries(scores)[:, :top_k]
    run_dict: Dict[str, Dict[str, float]] = {}
    for q_row, qid in enumerate(query_ids):
        q_run: Dict[str, float] = {}
        for d_idx in ranked_indices[q_row]:
            col = int(d_idx)
            q_run[doc_ids[col]] = float(scores[q_row, col])
        run_dict[qid] = q_run
    return run_dict
