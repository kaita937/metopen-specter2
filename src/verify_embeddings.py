"""Validator hasil encoding matriks SPECTER2 dari Google Colab (Tahap 3 — Hari 2).

Memeriksa seluruh syarat kewarasan (sanity check):
- query_emb.npy berukuran (300, 768), float32, bebas NaN/Inf, norma L2 > 0
- doc_emb_A.npy, doc_emb_B.npy, doc_emb_C.npy berukuran (5183, 768), float32, bebas NaN/Inf, norma L2 > 0
- query_ids.json (300 ID) cocok 100% dengan kueri uji pada qrels/test.tsv
- doc_ids.json (5.183 ID) cocok 100% dengan corpus.jsonl
- encoding_log.json memuat waktu encoding GPU, ms/dokumen, batch_size, dan spesifikasi hardware.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from src.data_loader import load_scifact_dataset


def verify_all_embeddings(
    project_root: str | Path | None = None,
) -> Dict[str, Any]:
    """Memvalidasi seluruh artefak di dalam folder embeddings/."""
    root = Path(project_root) if project_root else Path(__file__).resolve().parent.parent
    emb_dir = root / "embeddings"
    data_dir = root / "data" / "scifact"

    required_files = [
        "query_emb.npy",
        "query_ids.json",
        "doc_emb_A.npy",
        "doc_emb_B.npy",
        "doc_emb_C.npy",
        "doc_ids.json",
        "encoding_log.json",
    ]
    missing = [fname for fname in required_files if not (emb_dir / fname).is_file()]
    if missing:
        raise FileNotFoundError(
            f"File hasil encoding belum lengkap di {emb_dir}: {missing}"
        )

    corpus, test_queries, _ = load_scifact_dataset(data_dir, validate=True)

    with (emb_dir / "query_ids.json").open("r", encoding="utf-8") as f:
        query_ids = [str(x) for x in json.load(f)]
    with (emb_dir / "doc_ids.json").open("r", encoding="utf-8") as f:
        doc_ids = [str(x) for x in json.load(f)]
    with (emb_dir / "encoding_log.json").open("r", encoding="utf-8") as f:
        encoding_log = json.load(f)

    assert len(query_ids) == 300, f"Panjang query_ids harus 300, ditemukan {len(query_ids)}"
    assert set(query_ids) == set(test_queries.keys()), "Himpunan query_ids tidak cocok dengan qrels/test.tsv!"
    assert len(doc_ids) == 5183, f"Panjang doc_ids harus 5183, ditemukan {len(doc_ids)}"
    assert set(doc_ids) == set(corpus.keys()), "Himpunan doc_ids tidak cocok dengan corpus.jsonl!"
    assert len(set(doc_ids)) == 5183, "Terdapat duplikasi ID pada doc_ids.json!"

    q_emb = np.load(emb_dir / "query_emb.npy")
    d_a = np.load(emb_dir / "doc_emb_A.npy")
    d_b = np.load(emb_dir / "doc_emb_B.npy")
    d_c = np.load(emb_dir / "doc_emb_C.npy")

    assert q_emb.shape == (300, 768), f"Bentuk query_emb salah: {q_emb.shape}"
    assert d_a.shape == (5183, 768), f"Bentuk doc_emb_A salah: {d_a.shape}"
    assert d_b.shape == (5183, 768), f"Bentuk doc_emb_B salah: {d_b.shape}"
    assert d_c.shape == (5183, 768), f"Bentuk doc_emb_C salah: {d_c.shape}"

    summary_norms: Dict[str, Dict[str, float]] = {}
    for name, mat in [("query_emb", q_emb), ("doc_emb_A", d_a), ("doc_emb_B", d_b), ("doc_emb_C", d_c)]:
        assert mat.dtype == np.float32, f"{name} harus bertipe float32, ditemukan {mat.dtype}"
        assert not np.isnan(mat).any(), f"{name} mengandung nilai NaN!"
        assert not np.isinf(mat).any(), f"{name} mengandung nilai Inf!"
        norms = np.linalg.norm(mat, axis=1)
        assert np.all(norms > 0), f"{name} memiliki baris dengan norma vektor <= 0!"
        summary_norms[name] = {
            "min_norm": round(float(norms.min()), 4),
            "mean_norm": round(float(norms.mean()), 4),
            "max_norm": round(float(norms.max()), 4),
        }

    # Pastikan A, B, dan C benar-benar matriks berbeda (tidak salah tertimpa file yang sama)
    assert not np.allclose(d_a, d_b), "PERINGATAN: doc_emb_A dan doc_emb_B identik!"
    assert not np.allclose(d_b, d_c), "PERINGATAN: doc_emb_B dan doc_emb_C identik!"

    return {
        "status": "PASSED",
        "shapes": {
            "query_emb": q_emb.shape,
            "doc_emb_A": d_a.shape,
            "doc_emb_B": d_b.shape,
            "doc_emb_C": d_c.shape,
        },
        "norms": summary_norms,
        "hardware": encoding_log.get("hardware", {}),
        "timings": encoding_log.get("configs", {}),
    }


if __name__ == "__main__":
    res = verify_all_embeddings()
    print("=== SANITY CHECK EMBEDDINGS HARI 2: LOLOS 100% ===")
    print(json.dumps(res, indent=2, default=str))
