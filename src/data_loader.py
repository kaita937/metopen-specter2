"""Pemuat data dan validator integritas dataset SciFact (BEIR).

Tahap 1:
- Membaca corpus.jsonl (_id, title, text)
- Membaca queries.jsonl (_id, text)
- Membaca qrels/test.tsv (query-id, corpus-id, score > 0)
- Menyaring daftar kueri sehingga hanya menyisakan kueri uji yang muncul di qrels/test.tsv
- Menjalankan assertion integritas data (5.183 dokumen, 300 kueri uji, ±339 pasangan relevan,
  dan 100% cakupan corpus-id pada korpus).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Tuple

CorpusDict = Dict[str, Dict[str, str]]
QueryDict = Dict[str, Dict[str, str]]
QrelsDict = Dict[str, Dict[str, int]]


def load_corpus(corpus_path: str | Path) -> CorpusDict:
    """Membaca corpus.jsonl dan menyimpan atribut _id, title, serta text."""
    path = Path(corpus_path)
    if not path.is_file():
        raise FileNotFoundError(f"File korpus tidak ditemukan: {path}")

    corpus: CorpusDict = {}
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "_id" not in row:
                raise KeyError(f"Atribut '_id' hilang pada {path} baris {line_num}")
            doc_id = str(row["_id"])
            corpus[doc_id] = {
                "_id": doc_id,
                "title": "" if row.get("title") is None else str(row["title"]),
                "text": "" if row.get("text") is None else str(row["text"]),
            }
    return corpus


def load_queries(queries_path: str | Path) -> QueryDict:
    """Membaca queries.jsonl mentah dan menyimpan atribut _id serta text."""
    path = Path(queries_path)
    if not path.is_file():
        raise FileNotFoundError(f"File kueri tidak ditemukan: {path}")

    queries: QueryDict = {}
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "_id" not in row:
                raise KeyError(f"Atribut '_id' hilang pada {path} baris {line_num}")
            query_id = str(row["_id"])
            queries[query_id] = {
                "_id": query_id,
                "text": "" if row.get("text") is None else str(row["text"]),
            }
    return queries


def load_qrels(qrels_path: str | Path) -> QrelsDict:
    """Membaca qrels/test.tsv dan hanya menyimpan pasangan dengan score > 0.

    Struktur dictionary keluaran {query_id: {corpus_id: int_score}} langsung
    kompatibel dengan pytrec_eval.RelevanceEvaluator pada Tahap 4.
    """
    path = Path(qrels_path)
    if not path.is_file():
        raise FileNotFoundError(f"File qrels tidak ditemukan: {path}")

    qrels: QrelsDict = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required_cols = {"query-id", "corpus-id", "score"}
        if reader.fieldnames is None or not required_cols.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"Header TSV pada {path} tidak valid: {reader.fieldnames}. "
                f"Wajib memuat {sorted(required_cols)}."
            )
        for row in reader:
            score = int(float(row["score"]))
            if score > 0:
                qid = str(row["query-id"]).strip()
                cid = str(row["corpus-id"]).strip()
                if qid not in qrels:
                    qrels[qid] = {}
                qrels[qid][cid] = score
    return qrels


def validate_scifact_integrity(
    corpus: CorpusDict,
    test_queries: QueryDict,
    qrels: QrelsDict,
) -> Dict[str, int]:
    """Menjalankan assertion integritas dataset SciFact sesuai spesifikasi protokol BEIR."""
    num_docs = len(corpus)
    num_queries = len(test_queries)
    num_rel_pairs = sum(len(doc_scores) for doc_scores in qrels.values())

    missing_queries = [qid for qid in qrels if qid not in test_queries]
    missing_corpus_ids = {
        cid
        for doc_scores in qrels.values()
        for cid in doc_scores
        if cid not in corpus
    }

    assert num_docs == 5183, (
        f"Integritas gagal: jumlah dokumen seharusnya 5183, ditemukan {num_docs}."
    )
    assert num_queries == 300, (
        f"Integritas gagal: jumlah kueri uji seharusnya 300, ditemukan {num_queries}."
    )
    assert len(missing_queries) == 0, (
        f"Integritas gagal: {len(missing_queries)} query-id di qrels tidak ada di queries.jsonl."
    )
    assert len(missing_corpus_ids) == 0, (
        f"Integritas gagal: {len(missing_corpus_ids)} corpus-id di qrels tidak ditemukan di corpus.jsonl."
    )
    assert 330 <= num_rel_pairs <= 350, (
        f"Integritas gagal: jumlah pasangan relevan di luar kisaran ±339 (ditemukan {num_rel_pairs})."
    )

    return {
        "num_docs": num_docs,
        "num_test_queries": num_queries,
        "num_relevant_pairs": num_rel_pairs,
        "missing_corpus_ids": len(missing_corpus_ids),
    }


def load_scifact_dataset(
    data_dir: str | Path | None = None,
    validate: bool = True,
) -> Tuple[CorpusDict, QueryDict, QrelsDict]:
    """Memuat korpus, kueri uji tersaring (300 kueri), dan qrels/test.tsv sekaligus."""
    if data_dir is None:
        project_root = Path(__file__).resolve().parent.parent
        base_dir = project_root / "data" / "scifact"
    else:
        base_dir = Path(data_dir)

    corpus = load_corpus(base_dir / "corpus.jsonl")
    all_queries = load_queries(base_dir / "queries.jsonl")
    qrels = load_qrels(base_dir / "qrels" / "test.tsv")

    # Saring kueri agar hanya menyisakan kueri yang dievaluasi pada qrels/test.tsv
    test_queries: QueryDict = {
        qid: qdata for qid, qdata in all_queries.items() if qid in qrels
    }

    if validate:
        validate_scifact_integrity(corpus, test_queries, qrels)

    return corpus, test_queries, qrels


if __name__ == "__main__":
    corpus_data, queries_data, qrels_data = load_scifact_dataset(validate=True)
    stats = validate_scifact_integrity(corpus_data, queries_data, qrels_data)
    print("=== Validasi Integritas Dataset SciFact (BEIR) BERHASIL ===")
    print(f"Jumlah dokumen korpus       : {stats['num_docs']:,}")
    print(f"Jumlah kueri uji (test)     : {stats['num_test_queries']:,}")
    print(f"Jumlah pasangan relevan (>0): {stats['num_relevant_pairs']:,}")
    print(f"Corpus-ID hilang dari korpus: {stats['missing_corpus_ids']}")
