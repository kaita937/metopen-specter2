"""Implementasi mandiri metrik evaluasi IR & validasi silang pytrec_eval (Tahap 4 — Hari 3).

Rumus matematika yang diimplementasikan di sini identik dengan Persamaan (2)–(4) pada Section III-D:
- Persamaan (2): nDCG@10 (Normalized Discounted Cumulative Gain pada pemotongan k = 10)
- Persamaan (3): MRR@10  (Mean Reciprocal Rank dengan batas peringkat r <= 10)
- Persamaan (4): Recall@5 (Proporsi dokumen relevan yang ditemukan dalam 5 urutan teratas)

Seluruh skor per kueri divalidasi silang terhadap `pytrec_eval` dengan toleransi mutlak <= 1e-6.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping

import numpy as np
import pytrec_eval

from src.retrieval import build_pytrec_run_dict, rank_all_queries


def compute_ndcg_at_k_single(
    ranked_doc_ids: List[str],
    qrel_for_query: Mapping[str, int],
    k: int = 10,
) -> float:
    """Menghitung nDCG@k untuk satu kueri sesuai definisi standar trec_eval."""
    positive_rels = sorted([rel for rel in qrel_for_query.values() if rel > 0], reverse=True)
    if not positive_rels:
        return 0.0

    dcg = 0.0
    for idx, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        rel = qrel_for_query.get(doc_id, 0)
        if rel > 0:
            dcg += float(rel) / math.log2(idx + 1.0)

    idcg = 0.0
    for idx, rel in enumerate(positive_rels[:k], start=1):
        idcg += float(rel) / math.log2(idx + 1.0)

    return dcg / idcg if idcg > 0.0 else 0.0


def compute_rr_at_k_single(
    ranked_doc_ids: List[str],
    qrel_for_query: Mapping[str, int],
    k: int = 10,
) -> float:
    """Menghitung Reciprocal Rank@k (RR@10) untuk satu kueri (0.0 bila tidak ada di top-k)."""
    for idx, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        if qrel_for_query.get(doc_id, 0) > 0:
            return 1.0 / float(idx)
    return 0.0


def compute_recall_at_k_single(
    ranked_doc_ids: List[str],
    qrel_for_query: Mapping[str, int],
    k: int = 5,
) -> float:
    """Menghitung Recall@k (Recall@5) untuk satu kueri."""
    total_relevant = sum(1 for rel in qrel_for_query.values() if rel > 0)
    if total_relevant == 0:
        return 0.0
    hits = sum(1 for doc_id in ranked_doc_ids[:k] if qrel_for_query.get(doc_id, 0) > 0)
    return float(hits) / float(total_relevant)


def evaluate_manual_per_query(
    scores: np.ndarray,
    query_ids: List[str],
    doc_ids: List[str],
    qrels: Mapping[str, Mapping[str, int]],
) -> Dict[str, Dict[str, float]]:
    """Menghitung nDCG@10, MRR@10, dan Recall@5 secara mandiri untuk setiap kueri."""
    max_cutoff = 10
    ranked_indices = rank_all_queries(scores)[:, :max_cutoff]

    per_query: Dict[str, Dict[str, float]] = {}
    for q_row, qid in enumerate(query_ids):
        top_doc_ids = [doc_ids[int(col)] for col in ranked_indices[q_row]]
        q_rel = qrels.get(qid, {})
        per_query[qid] = {
            "ndcg@10": compute_ndcg_at_k_single(top_doc_ids, q_rel, k=10),
            "mrr@10": compute_rr_at_k_single(top_doc_ids, q_rel, k=10),
            "recall@5": compute_recall_at_k_single(top_doc_ids, q_rel, k=5),
        }
    return per_query


def evaluate_and_validate_with_pytrec(
    scores: np.ndarray,
    query_ids: List[str],
    doc_ids: List[str],
    qrels: Mapping[str, Mapping[str, int]],
    tol: float = 1e-6,
) -> Dict[str, Any]:
    """Menghitung metrik manual sekaligus memvalidasi silang per kueri melawan pytrec_eval.

    Catatan teknis krusial:
    - Untuk `ndcg_cut.10` dan `recall.5`, digunakan kandidat top-100.
    - Untuk `recip_rank` (MRR@10), dictionary `run` dibatasi ketat tepat pada **top-10** dokumen
      teratas per kueri karena `recip_rank` di trec_eval membaca seluruh daftar pada `run`.
    """
    manual_per_query = evaluate_manual_per_query(scores, query_ids, doc_ids, qrels)

    qrels_clean: Dict[str, Dict[str, int]] = {
        str(qid): {str(cid): int(score) for cid, score in doc_map.items()}
        for qid, doc_map in qrels.items()
    }

    run_top100 = build_pytrec_run_dict(scores, query_ids, doc_ids, top_k=100)
    run_top10 = build_pytrec_run_dict(scores, query_ids, doc_ids, top_k=10)

    eval_ndcg_recall = pytrec_eval.RelevanceEvaluator(qrels_clean, {"ndcg_cut.10", "recall.5"})
    pytrec_res_100 = eval_ndcg_recall.evaluate(run_top100)

    eval_mrr10 = pytrec_eval.RelevanceEvaluator(qrels_clean, {"recip_rank"})
    pytrec_res_10 = eval_mrr10.evaluate(run_top10)

    max_diffs = {"ndcg@10": 0.0, "mrr@10": 0.0, "recall@5": 0.0}
    for qid in query_ids:
        m_ndcg = manual_per_query[qid]["ndcg@10"]
        p_ndcg = float(pytrec_res_100[qid]["ndcg_cut_10"])
        diff_ndcg = abs(m_ndcg - p_ndcg)

        m_mrr = manual_per_query[qid]["mrr@10"]
        p_mrr = float(pytrec_res_10[qid]["recip_rank"])
        diff_mrr = abs(m_mrr - p_mrr)

        m_rec = manual_per_query[qid]["recall@5"]
        p_rec = float(pytrec_res_100[qid]["recall_5"])
        diff_rec = abs(m_rec - p_rec)

        max_diffs["ndcg@10"] = max(max_diffs["ndcg@10"], diff_ndcg)
        max_diffs["mrr@10"] = max(max_diffs["mrr@10"], diff_mrr)
        max_diffs["recall@5"] = max(max_diffs["recall@5"], diff_rec)

        assert diff_ndcg <= tol, (
            f"Selisih nDCG@10 melebihi toleransi pada qid={qid}: manual={m_ndcg} vs pytrec={p_ndcg}"
        )
        assert diff_mrr <= tol, (
            f"Selisih MRR@10 melebihi toleransi pada qid={qid}: manual={m_mrr} vs pytrec={p_mrr}"
        )
        assert diff_rec <= tol, (
            f"Selisih Recall@5 melebihi toleransi pada qid={qid}: manual={m_rec} vs pytrec={p_rec}"
        )

    ndcg_arr = np.array([manual_per_query[qid]["ndcg@10"] for qid in query_ids], dtype=np.float64)
    mrr_arr = np.array([manual_per_query[qid]["mrr@10"] for qid in query_ids], dtype=np.float64)
    rec_arr = np.array([manual_per_query[qid]["recall@5"] for qid in query_ids], dtype=np.float64)

    summary = {
        "num_queries": len(query_ids),
        "ndcg@10_mean": float(ndcg_arr.mean()),
        "ndcg@10_sd": float(ndcg_arr.std(ddof=1)),
        "mrr@10_mean": float(mrr_arr.mean()),
        "mrr@10_sd": float(mrr_arr.std(ddof=1)),
        "recall@5_mean": float(rec_arr.mean()),
        "recall@5_sd": float(rec_arr.std(ddof=1)),
        "max_abs_diff_ndcg@10": max_diffs["ndcg@10"],
        "max_abs_diff_mrr@10": max_diffs["mrr@10"],
        "max_abs_diff_recall@5": max_diffs["recall@5"],
    }

    return {
        "summary": summary,
        "per_query": manual_per_query,
        "arrays": {
            "ndcg@10": ndcg_arr,
            "mrr@10": mrr_arr,
            "recall@5": rec_arr,
        },
    }
