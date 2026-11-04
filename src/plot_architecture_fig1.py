"""Generator Diagram Arsitektur Eksperimen Asimetris SPECTER2 (Fig. 1 Standar IEEE — Hari 4).

Spesifikasi publikasi IEEE:
- Lebar 2 kolom IEEE (double-column span): figsize = (7.16, 2.75) inci
- Resolusi tajam 300 DPI (.png) dan format vektor murni (.pdf)
- Font Times New Roman 8 pt, tanpa plt.title() di atas gambar (judul berada di caption Fig. 1)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def generate_fig1_architecture(figures_dir: str | Path) -> Path:
    """Menggambar dan menyimpan Fig. 1 (diagram alur metodologi asimetris SPECTER2)."""
    out_dir = Path(figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8,
    })

    fig, ax = plt.subplots(figsize=(7.16, 2.90), dpi=300)
    ax.set_xlim(0.0, 10.0)
    ax.set_ylim(0.0, 4.06)
    ax.axis("off")

    # Latar belakang swimlane: Atas = Offline Indexing, Bawah = Online Query Retrieval
    # Dipisahkan oleh kanal horizontal bersih (y = 1.83 s.d. 2.15) untuk jalur panah penghubung
    lane_offline = FancyBboxPatch(
        (0.10, 2.21),
        9.80,
        1.68,
        boxstyle="round,pad=0.05,rounding_size=0.08",
        facecolor="#F4F7FB",
        edgecolor="#B0C4DE",
        linewidth=0.8,
        linestyle="--",
    )
    lane_online = FancyBboxPatch(
        (0.10, 0.12),
        9.80,
        1.66,
        boxstyle="round,pad=0.05,rounding_size=0.08",
        facecolor="#F6FBF6",
        edgecolor="#A8D0A8",
        linewidth=0.8,
        linestyle="--",
    )
    ax.add_patch(lane_offline)
    ax.add_patch(lane_online)

    ax.text(
        0.22,
        3.74,
        "Offline Document Indexing Pipeline (GPU: NVIDIA Tesla T4)",
        fontsize=7.8,
        fontweight="bold",
        color="#1F4E79",
        va="center",
    )
    ax.text(
        0.22,
        1.64,
        "Online Asymmetric Query Retrieval & Evaluation Pipeline",
        fontsize=7.8,
        fontweight="bold",
        color="#275E2B",
        va="center",
    )

    def draw_box(
        x: float,
        y: float,
        w: float,
        h: float,
        title: str,
        body: str,
        fc: str = "#FFFFFF",
        ec: str = "#333333",
        title_color: str = "#111111",
    ) -> None:
        box = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.04,rounding_size=0.06",
            facecolor=fc,
            edgecolor=ec,
            linewidth=0.85,
        )
        ax.add_patch(box)
        ax.text(
            x + w / 2.0,
            y + h * 0.72,
            title,
            ha="center",
            va="center",
            fontsize=7.6,
            fontweight="bold",
            color=title_color,
        )
        ax.text(
            x + w / 2.0,
            y + h * 0.30,
            body,
            ha="center",
            va="center",
            fontsize=6.8,
            color="#222222",
            linespacing=1.18,
        )

    def draw_arrow(x1: float, y1: float, x2: float, y2: float, color: str = "#333333") -> None:
        arr = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>,head_length=4.2,head_width=2.6",
            mutation_scale=1.0,
            linewidth=0.9,
            color=color,
        )
        ax.add_patch(arr)

    # --- Kotak Jalur Atas (Offline Indexing) ---
    y_top = 2.34
    h_box = 1.18

    draw_box(
        0.24, y_top, 1.48, h_box,
        "SciFact Corpus",
        "5,183 Biomedical\nPapers (BEIR)\ncorpus.jsonl",
        fc="#FFFFFF", ec="#4C72B0", title_color="#1F4E79",
    )
    draw_box(
        2.02, y_top, 2.06, h_box,
        "Text Configurations",
        "A: Title\nB: Abstract\nC: Title + [SEP] + Abstract",
        fc="#EEF3FA", ec="#4C72B0", title_color="#1F4E79",
    )
    draw_box(
        4.38, y_top, 2.02, h_box,
        "Document Encoder",
        "allenai/specter2_base\n+ proximity adapter\n(max_length = 512)",
        fc="#E2ECF9", ec="#2F5597", title_color="#16365C",
    )
    draw_box(
        6.70, y_top, 1.38, h_box,
        "Pooling & Norm",
        "Last-Layer [CLS]\n+ L2 Norm\n(dim = 768)",
        fc="#FFFFFF", ec="#4C72B0", title_color="#1F4E79",
    )
    draw_box(
        8.36, y_top, 1.40, h_box,
        "Doc Embeddings",
        "Matrices D_A, D_B, D_C\n(5,183 × 768)\nfloat32 (.npy)",
        fc="#FFF8E7", ec="#B8860B", title_color="#7F5A00",
    )

    draw_arrow(1.74, y_top + h_box / 2, 2.00, y_top + h_box / 2, "#2F5597")
    draw_arrow(4.10, y_top + h_box / 2, 4.36, y_top + h_box / 2, "#2F5597")
    draw_arrow(6.42, y_top + h_box / 2, 6.68, y_top + h_box / 2, "#2F5597")
    draw_arrow(8.10, y_top + h_box / 2, 8.34, y_top + h_box / 2, "#2F5597")

    # --- Kotak Jalur Bawah (Online Retrieval & Evaluation) ---
    y_bot = 0.24

    draw_box(
        0.24, y_bot, 1.48, h_box,
        "Test Queries",
        "300 Scientific\nClaims (Filtered)\nqueries.jsonl",
        fc="#FFFFFF", ec="#55A868", title_color="#275E2B",
    )
    draw_box(
        2.02, y_bot, 2.06, h_box,
        "Asymmetric Query Enc.",
        "allenai/specter2_base\n+ adhoc_query adapter\n(Encoded 1× Only)",
        fc="#E6F4EA", ec="#38761D", title_color="#1E4620",
    )
    draw_box(
        4.38, y_bot, 1.56, h_box,
        "Query Matrix",
        "Last-Layer [CLS]\n+ L2 Norm: Q_norm\n(300 × 768)",
        fc="#FFFFFF", ec="#55A868", title_color="#275E2B",
    )
    draw_box(
        6.24, y_bot, 1.62, h_box,
        "Cosine Retrieval",
        "Matrix Dot Product\nS = Q_norm @ D_norm^T\nTop-k Ranking (300×5,183)",
        fc="#FFF2CC", ec="#BF9000", title_color="#7F5A00",
    )
    draw_box(
        8.14, y_bot, 1.62, h_box,
        "Benchmark & Stat.",
        "qrels/test.tsv (339 pairs)\nnDCG@10, MRR@10, R@5\nPaired Wilcoxon (Bonf.)",
        fc="#FBE9E7", ec="#A61C00", title_color="#7B1FA2",
    )

    draw_arrow(1.74, y_bot + h_box / 2, 2.00, y_bot + h_box / 2, "#38761D")
    draw_arrow(4.10, y_bot + h_box / 2, 4.36, y_bot + h_box / 2, "#38761D")
    draw_arrow(5.96, y_bot + h_box / 2, 6.22, y_bot + h_box / 2, "#38761D")
    draw_arrow(7.88, y_bot + h_box / 2, 8.12, y_bot + h_box / 2, "#A61C00")

    # Panah penghubung ortogonal (2 belokan di kanal tengah y = 1.98) dari Doc Embeddings ke Cosine Retrieval
    x_src = 9.06
    x_dst = 7.05
    y_src = y_top - 0.04
    y_mid = 1.98
    y_dst = y_bot + h_box + 0.05

    ax.plot(
        [x_src, x_src, x_dst],
        [y_src, y_mid, y_mid],
        color="#B8860B",
        linewidth=1.05,
        solid_capstyle="round",
        solid_joinstyle="round",
    )
    connector_down = FancyArrowPatch(
        (x_dst, y_mid + 0.01),
        (x_dst, y_dst),
        arrowstyle="-|>,head_length=4.5,head_width=2.8",
        mutation_scale=1.0,
        linewidth=1.05,
        color="#B8860B",
    )
    ax.add_patch(connector_down)
    ax.text(
        8.08,
        y_mid,
        "Evaluate A, B, C",
        fontsize=6.6,
        fontstyle="italic",
        color="#7F5A00",
        ha="center",
        va="center",
        bbox={"boxstyle": "round,pad=0.14", "facecolor": "#FFFFFF", "edgecolor": "#E0C97F", "linewidth": 0.5},
    )

    fig.tight_layout(pad=0.15)
    png_path = out_dir / "fig1_system_architecture.png"
    pdf_path = out_dir / "fig1_system_architecture.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    out = generate_fig1_architecture(root / "results" / "figures")
    print("Fig. 1 tersimpan di:", out)
