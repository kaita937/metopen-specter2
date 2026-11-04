"""Aplikasi Demo Interaktif Komparasi Konfigurasi SPECTER2 pada SciFact (Tahap 7 — Hari 4).

Cara menjalankan dari root folder `metopen-specter2/`:
    .venv\\Scripts\\streamlit run demo/app.py

Fitur utama:
1. Instan di laptop tanpa GPU (@st.cache_resource memuat matriks .npy hasil encoding Tesla T4).
2. Pemilih 300 kueri uji SciFact + tombol pintasan 4 Studi Kasus Terbaik (TABLE IV Error Analysis).
3. Komparasi 3 kolom berdampingan:
   - Konfigurasi A (Title)
   - Konfigurasi B (Abstract)
   - Konfigurasi C (Title + [SEP] + Abstract)
   Menampilkan Top-5 dokumen, skor Cosine Similarity (0.xxxx), dan penanda hijau `✓ Relevan (qrels)`.
4. Tab ke-2 (Opsional Live CPU): Uji kalimat klaim ilmiah bebas menggunakan model `specter2_adhoc_query`.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import streamlit as st

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_scifact_dataset
from src.metrics import (
    compute_ndcg_at_k_single,
    compute_recall_at_k_single,
    compute_rr_at_k_single,
)
from src.retrieval import l2_normalize

st.set_page_config(
    page_title="Demo SPECTER2 pada SciFact (BEIR)",
    page_icon="🔬",
    layout="wide",
)


@st.cache_resource(show_spinner="Memuat indeks vektor SPECTER2 & korpus SciFact ke memori...")
def load_demo_resources() -> Dict[str, Any]:
    """Memuat seluruh matriks vektor ter-normalisasi L2 dan metadata SciFact satu kali ke RAM."""
    data_dir = PROJECT_ROOT / "data" / "scifact"
    emb_dir = PROJECT_ROOT / "embeddings"
    tables_dir = PROJECT_ROOT / "results" / "tables"
    ea_dir = PROJECT_ROOT / "results" / "error_analysis"

    corpus, test_queries, qrels = load_scifact_dataset(data_dir, validate=True)

    with (emb_dir / "query_ids.json").open("r", encoding="utf-8") as f:
        query_ids: List[str] = [str(x) for x in json.load(f)]
    with (emb_dir / "doc_ids.json").open("r", encoding="utf-8") as f:
        doc_ids: List[str] = [str(x) for x in json.load(f)]
    with (emb_dir / "encoding_log.json").open("r", encoding="utf-8") as f:
        encoding_log = json.load(f)

    q_norm = l2_normalize(np.load(emb_dir / "query_emb.npy"))
    d_norm = {
        "A": l2_normalize(np.load(emb_dir / "doc_emb_A.npy")),
        "B": l2_normalize(np.load(emb_dir / "doc_emb_B.npy")),
        "C": l2_normalize(np.load(emb_dir / "doc_emb_C.npy")),
    }

    qid_to_row = {qid: idx for idx, qid in enumerate(query_ids)}
    did_to_col = {did: idx for idx, did in enumerate(doc_ids)}

    selected_cases_path = ea_dir / "selected_cases.csv"
    selected_cases_df = pd.read_csv(selected_cases_path) if selected_cases_path.is_file() else pd.DataFrame()

    metrics_df = pd.read_csv(tables_dir / "metrics_summary.csv") if (tables_dir / "metrics_summary.csv").is_file() else pd.DataFrame()
    wilcoxon_df = pd.read_csv(tables_dir / "wilcoxon.csv") if (tables_dir / "wilcoxon.csv").is_file() else pd.DataFrame()

    return {
        "corpus": corpus,
        "test_queries": test_queries,
        "qrels": qrels,
        "query_ids": query_ids,
        "doc_ids": doc_ids,
        "qid_to_row": qid_to_row,
        "did_to_col": did_to_col,
        "q_norm": q_norm,
        "d_norm": d_norm,
        "encoding_log": encoding_log,
        "selected_cases_df": selected_cases_df,
        "metrics_df": metrics_df,
        "wilcoxon_df": wilcoxon_df,
    }


@st.cache_resource(show_spinner="Memuat model SPECTER2 + adapter adhoc_query di CPU (hanya 1x)...")
def load_live_cpu_query_encoder():
    """Memuat model allenai/specter2_base + adhoc_query di CPU khusus Tab 2 (opsional)."""
    import torch
    from adapters import AutoAdapterModel
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
    model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
    try:
        model.load_adapter("allenai/specter2_adhoc_query", source="hf", load_as="adhoc_query", set_active=True)
    except TypeError:
        model.load_adapter("allenai/specter2_adhoc_query", load_as="adhoc_query", set_active=True)
    model.set_active_adapters("adhoc_query")
    model.to("cpu")
    model.eval()
    return tokenizer, model, torch


def retrieve_for_query_vector(
    q_vec_norm_1d: np.ndarray,
    d_norm_mat: np.ndarray,
    doc_ids: List[str],
    did_to_col: Dict[str, int],
    qrel_map: Dict[str, int],
    top_k: int = 5,
) -> Dict[str, Any]:
    """Menghitung skor kosinus satu vektor kueri terhadap satu konfigurasi dokumen."""
    scores = d_norm_mat @ q_vec_norm_1d  # (5183,)
    order = np.argsort(-scores, kind="mergesort")

    top_10_ids = [doc_ids[int(i)] for i in order[:10]]
    top_k_items = []
    for rank_idx, col_idx in enumerate(order[:top_k], start=1):
        did = doc_ids[int(col_idx)]
        top_k_items.append({
            "rank": rank_idx,
            "doc_id": did,
            "score": float(scores[int(col_idx)]),
            "is_relevant": qrel_map.get(did, 0) > 0,
        })

    # Peringkat eksak setiap dokumen relevan di qrels
    col_to_rank = np.empty(len(doc_ids), dtype=np.int32)
    col_to_rank[order] = np.arange(1, len(doc_ids) + 1, dtype=np.int32)
    gt_ranks = {}
    for rel_did, grade in qrel_map.items():
        if grade > 0 and rel_did in did_to_col:
            c_idx = did_to_col[rel_did]
            gt_ranks[rel_did] = {
                "rank": int(col_to_rank[c_idx]),
                "score": float(scores[c_idx]),
            }

    metrics = {}
    if qrel_map:
        metrics = {
            "ndcg@10": compute_ndcg_at_k_single(top_10_ids, qrel_map, k=10),
            "mrr@10": compute_rr_at_k_single(top_10_ids, qrel_map, k=10),
            "recall@5": compute_recall_at_k_single(top_10_ids, qrel_map, k=5),
        }

    return {
        "top_k": top_k_items,
        "gt_ranks": gt_ranks,
        "metrics": metrics,
    }


def render_doc_card(item: Dict[str, Any], doc_meta: Dict[str, str]) -> None:
    """Merender kartu kandidat dokumen dengan penanda hijau bila relevan (qrels)."""
    is_rel = item["is_relevant"]
    border_color = "#2E7D32" if is_rel else "rgba(128, 128, 128, 0.32)"
    bg_color = "rgba(46, 125, 50, 0.10)" if is_rel else "rgba(128, 128, 128, 0.04)"
    badge_html = (
        '<span style="background-color:#2E7D32;color:#FFFFFF;padding:2px 8px;'
        'border-radius:10px;font-size:0.78rem;font-weight:600;">✓ Relevan (qrels)</span>'
        if is_rel
        else '<span style="color:#777777;font-size:0.76rem;">Non-qrels</span>'
    )

    title_txt = (doc_meta.get("title") or "(Tanpa Judul)").strip()
    abs_txt = (doc_meta.get("text") or "").strip()

    st.markdown(
        f"""
        <div style="border:1.5px solid {border_color}; background:{bg_color};
                    border-radius:8px; padding:10px 12px; margin-bottom:8px;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                <span style="font-weight:700; font-size:0.9rem;">
                    #{item['rank']} &nbsp;|&nbsp; Doc <code>{item['doc_id']}</code>
                </span>
                {badge_html}
            </div>
            <div style="font-size:0.82rem; margin-bottom:5px;">
                Cosine Similarity: <b><code>{item['score']:.4f}</code></b>
            </div>
            <div style="font-size:0.88rem; font-weight:600; line-height:1.35;">
                {title_txt}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander(f"Lihat Abstrak Doc {item['doc_id']}", expanded=False):
        st.caption(abs_txt)


res = load_demo_resources()

st.title("🔬 Evaluasi Ablasi SPECTER2 pada Klaim Ilmiah SciFact (BEIR)")
st.caption(
    "Perbandingan asimetris **Konfigurasi A (`Title`)**, **Konfigurasi B (`Abstract`)**, dan "
    "**Konfigurasi C (`Title + [SEP] + Abstract`)** menggunakan `allenai/specter2_adhoc_query` & `allenai/specter2` (`proximity`). "
    "Berjalan instan di CPU tanpa GPU."
)

tab_eval, tab_live, tab_summary = st.tabs([
    "🔍 Eksplorasi 300 Kueri Uji SciFact",
    "✍️ Uji Kueri Ilmiah Bebas (Live CPU)",
    "📊 Ringkasan Hasil & Statistik IEEE",
])

CONFIG_HEADERS = {
    "A": ("Konfigurasi A — Title Only", "12.24 s (13.12× lebih cepat)"),
    "B": ("Konfigurasi B — Abstract Only", "153.17 s (1.05× lebih cepat)"),
    "C": ("Konfigurasi C — Title + [SEP] + Abstract", "160.66 s (Akurasi Terbaik)"),
}

# ==============================================================================
# TAB 1: EKSPLORASI 300 KUERI UJI SCIFACT (INSTAN TANPA GPU)
# ==============================================================================
with tab_eval:
    query_ids = res["query_ids"]
    test_queries = res["test_queries"]
    selected_cases_df = res["selected_cases_df"]

    if "active_qid" not in st.session_state:
        if not selected_cases_df.empty:
            st.session_state["active_qid"] = str(selected_cases_df.iloc[0]["query_id"])
        else:
            st.session_state["active_qid"] = query_ids[0]

    # Pintasan 4 Studi Kasus Terbaik (TABLE IV Error Analysis) untuk Presentasi Live
    if not selected_cases_df.empty:
        st.markdown("##### ⚡ Pintasan Cepat 4 Studi Kasus `TABLE IV` (Error Analysis):")
        preset_cols = st.columns(4)
        short_labels = [
            "1️⃣ A Gagal → C Peringkat #1",
            "2️⃣ Judul (A) Unggul vs Abstrak",
            "3️⃣ Batas Zero-Shot (Gagal Semua)",
            "4️⃣ Minim Kata Sama tapi C Top-3",
        ]
        for idx, (_, crow) in enumerate(selected_cases_df.iterrows()):
            if idx < 4:
                cqid = str(crow["query_id"])
                btn_lbl = f"{short_labels[idx]} (ID {cqid})"
                if preset_cols[idx].button(btn_lbl, width="stretch"):
                    st.session_state["active_qid"] = cqid

    options_map = {
        qid: f"[ID {qid}] {test_queries[qid]['text'][:115]}{'...' if len(test_queries[qid]['text']) > 115 else ''}"
        for qid in query_ids
    }
    current_idx = query_ids.index(st.session_state["active_qid"]) if st.session_state["active_qid"] in query_ids else 0

    chosen_qid = st.selectbox(
        "Pilih dari 300 Kueri Klaim Ilmiah Uji (`qrels/test.tsv`):",
        options=query_ids,
        index=current_idx,
        format_func=lambda x: options_map[x],
    )
    st.session_state["active_qid"] = chosen_qid

    claim_text = test_queries[chosen_qid]["text"]
    qrel_for_q = res["qrels"].get(chosen_qid, {})
    q_row_idx = res["qid_to_row"][chosen_qid]
    q_vec = res["q_norm"][q_row_idx]

    cfg_outputs = {
        cfg: retrieve_for_query_vector(
            q_vec_norm_1d=q_vec,
            d_norm_mat=res["d_norm"][cfg],
            doc_ids=res["doc_ids"],
            did_to_col=res["did_to_col"],
            qrel_map=qrel_for_q,
            top_k=5,
        )
        for cfg in ["A", "B", "C"]
    }

    # Kotak Informasi Klaim & Ground-Truth (qrels)
    st.info(f"**Klaim Ilmiah Uji (Query ID `{chosen_qid}`):**\n\n> {claim_text}")

    with st.expander(f"📌 Kunci Jawaban Ground-Truth `qrels/test.tsv` ({len(qrel_for_q)} dokumen relevan)", expanded=True):
        for rel_did in qrel_for_q:
            gt_doc = res["corpus"][rel_did]
            r_a = cfg_outputs["A"]["gt_ranks"][rel_did]["rank"]
            r_b = cfg_outputs["B"]["gt_ranks"][rel_did]["rank"]
            r_c = cfg_outputs["C"]["gt_ranks"][rel_did]["rank"]
            s_a = cfg_outputs["A"]["gt_ranks"][rel_did]["score"]
            s_b = cfg_outputs["B"]["gt_ranks"][rel_did]["score"]
            s_c = cfg_outputs["C"]["gt_ranks"][rel_did]["score"]

            st.markdown(
                f"- **Doc `{rel_did}`** — *{gt_doc['title']}*  \n"
                f"  Peringkat pada **Config A**: `#{r_a}` (`{s_a:.4f}`) &nbsp;|&nbsp; "
                f"**Config B**: `#{r_b}` (`{s_b:.4f}`) &nbsp;|&nbsp; "
                f"**Config C**: `#{r_c}` (`{s_c:.4f}`)"
            )
            st.caption(gt_doc["text"])

    st.divider()

    # Tampilan 3 Kolom Berdampingan (A, B, C)
    cols = st.columns(3)
    for col_ui, cfg in zip(cols, ["A", "B", "C"]):
        with col_ui:
            header_title, gpu_note = CONFIG_HEADERS[cfg]
            st.subheader(header_title)
            m = cfg_outputs[cfg]["metrics"]
            st.caption(
                f"⏱️ Indexing T4: `{gpu_note}`  \n"
                f"**nDCG@10:** `{m['ndcg@10']:.4f}` | **MRR@10:** `{m['mrr@10']:.4f}` | **R@5:** `{m['recall@5']:.4f}`"
            )
            for cand in cfg_outputs[cfg]["top_k"]:
                render_doc_card(cand, res["corpus"][cand["doc_id"]])

# ==============================================================================
# TAB 2: UJI KUERI ILMIAH BEBAS (OPSIONAL LIVE CPU ENCODER)
# ==============================================================================
with tab_live:
    st.markdown(
        "Ketik klaim biomedis/ilmiah bebas dalam bahasa Inggris. Aplikasi akan memanggil adapter "
        "**`allenai/specter2_adhoc_query`** di CPU hanya untuk **1 kalimat input Anda**, lalu mencocokkannya "
        "secara *real-time* terhadap `5.183` vektor dokumen Konfigurasi A, B, dan C."
    )
    default_free_claim = "Vitamin D supplementation reduces the risk of acute respiratory tract infection."
    free_claim = st.text_input("Masukkan kalimat klaim ilmiah (Bahasa Inggris):", value=default_free_claim)

    if st.button("🚀 Encode & Cari Top-5 Dokumen di A, B, dan C", type="primary"):
        if not free_claim.strip():
            st.warning("Silakan masukkan teks klaim terlebih dahulu.")
        else:
            tokenizer_live, model_live, torch_mod = load_live_cpu_query_encoder()
            with torch_mod.no_grad():
                enc = tokenizer_live(
                    [free_claim.strip()],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                out = model_live(**enc)
                cls_vec = out.last_hidden_state[:, 0, :].cpu().numpy()
                cls_norm_1d = l2_normalize(cls_vec)[0]

            live_cols = st.columns(3)
            for col_ui, cfg in zip(live_cols, ["A", "B", "C"]):
                with col_ui:
                    header_title, _ = CONFIG_HEADERS[cfg]
                    st.subheader(header_title)
                    live_res = retrieve_for_query_vector(
                        q_vec_norm_1d=cls_norm_1d,
                        d_norm_mat=res["d_norm"][cfg],
                        doc_ids=res["doc_ids"],
                        did_to_col=res["did_to_col"],
                        qrel_map={},
                        top_k=5,
                    )
                    for cand in live_res["top_k"]:
                        render_doc_card(cand, res["corpus"][cand["doc_id"]])

# ==============================================================================
# TAB 3: RINGKASAN METRIK, UJI WILCOXON, & GRAFIK IEEE
# ==============================================================================
with tab_summary:
    st.subheader("TABLE II — Efektivitas Retrieval Zero-Shot pada SciFact (N = 300)")
    if not res["metrics_df"].empty:
        st.dataframe(res["metrics_df"], width="stretch", hide_index=True)

    st.subheader("TABLE III — Uji Signifikansi Paired Wilcoxon (Bonferroni α = 0.0167) & Efisiensi GPU")
    if not res["wilcoxon_df"].empty:
        st.dataframe(res["wilcoxon_df"], width="stretch", hide_index=True)

    fig_cols = st.columns(2)
    fig1_p = PROJECT_ROOT / "results" / "figures" / "fig1_system_architecture.png"
    fig2_p = PROJECT_ROOT / "results" / "figures" / "fig2_retrieval_performance.png"
    if fig1_p.is_file():
        fig_cols[0].image(str(fig1_p), caption="Fig. 1. Asymmetric SPECTER2 Retrieval Architecture.")
    if fig2_p.is_file():
        fig_cols[1].image(str(fig2_p), caption="Fig. 2. Retrieval Performance Across Input Configurations.")
