#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Manifest to LaTeX Converter (Fallback Engine Mapping)
Path: scripts/export_manifest_to_latex.py
Author: Maksym Nenashev (Systems Engineer)
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MANIFEST_FILE = BASE_DIR / "artifacts" / "PAPER_ARTIFACT_MANIFEST.json"
OUTPUT_TEX = BASE_DIR / "artifacts" / "latex_tables_and_snippets.tex"

EXPECTED_ENGINES = [
    "v1", "v3", "v3.1", "v3.2", "v4", "v4.1", "v4.2",
    "v5", "v5.1", "v5.2", "v6", "v6.1", "v6.2",
    "v7", "v7.1", "v7.2", "v8", "v8.1", "v8.2", "v9", "v9.1", "v9.2"
]


def escape_latex(text: str) -> str:
    return str(text).replace("&", "\\&").replace("_", "\\_").replace("%", "\\%")


def format_dataset_table(protocol_data):
    tex = [
        "% --- DATASET PROTOCOL TABLE ---",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Dataset provenance, deduplication yields, and 80/20 stratified split protocol.}",
        "\\label{tab:dataset_protocol}",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "\\textbf{Regional Cluster} & \\textbf{Engines} & \\textbf{Clean N} & \\textbf{Train Set (80\\%)} & \\textbf{Hold-out Test (20\\%)} & \\textbf{SHA-256 (16-char)} \\\\",
        "\\midrule"
    ]

    items = protocol_data.values() if isinstance(protocol_data, dict) else protocol_data

    for v in items:
        if not isinstance(v, dict):
            continue
        cluster = escape_latex(v.get("cluster", "N/A"))
        engines = escape_latex(v.get("engines") or v.get("engine_base") or v.get("engine") or "N/A")
        clean_n = f"{v.get('clean_n', 0):,}"
        train_str = f"{v.get('train_n', 0):,} [{v.get('train_pos',0)}/{v.get('train_neg',0)}]"
        test_str = f"{v.get('test_n', 0):,} [{v.get('test_pos',0)}/{v.get('test_neg',0)}]"
        sha = str(v.get("sha256", ""))[:16]
        tex.append(f"{cluster} & \\texttt{{{engines}}} & {clean_n} & {train_str} & {test_str} & \\texttt{{{sha}}} \\\\")

    tex.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}\n\n"])
    return "\n".join(tex)


def format_rq1_table(rq1_data):
    tex = [
        "% --- RQ1 CLASSIFICATION TABLE ---",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Hold-out classification evaluation across 22 specialized engines (review threshold $\\tau$).}",
        "\\label{tab:rq1_classification}",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "\\textbf{Engine} & $\\tau$ & \\textbf{TP} & \\textbf{FP} & \\textbf{FN} & \\textbf{Recall (95\\% CI)} & \\textbf{FPR (95\\% CI)} \\\\",
        "\\midrule"
    ]

    items = []
    if isinstance(rq1_data, dict):
        for k, v in rq1_data.items():
            if isinstance(v, dict):
                item = v.copy()
                item["engine"] = k
                items.append(item)
    elif isinstance(rq1_data, list):
        for idx, entry in enumerate(rq1_data):
            if isinstance(entry, dict):
                item = entry.copy()
                if not item.get("engine") or item.get("engine") == "v?":
                    item["engine"] = EXPECTED_ENGINES[idx] if idx < len(EXPECTED_ENGINES) else f"v_{idx}"
                items.append(item)

    for m in items:
        if not isinstance(m, dict):
            continue
        eng = escape_latex(m.get("engine") or "v?")
        t_val = m.get("threshold", 0.85)
        tp, fp, fn = m.get("tp", 0), m.get("fp", 0), m.get("fn", 0)
        rec = m.get("recall", 0.0)
        rec_ci = m.get("recall_95ci") or m.get("recall_ci") or [0, 0]
        fpr = m.get("fpr", 0.0)
        fpr_ci = m.get("fpr_95ci") or m.get("fpr_ci") or [0, 0]

        rec_str = f"{rec:.4f} [{rec_ci[0]:.3f}, {rec_ci[1]:.3f}]"
        fpr_str = f"{fpr:.4f} [{fpr_ci[0]:.3f}, {fpr_ci[1]:.3f}]"
        tex.append(f"\\texttt{{{eng}}} & {t_val:.2f} & {tp} & {fp} & {fn} & {rec_str} & {fpr_str} \\\\")

    tex.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}\n\n"])
    return "\n".join(tex)


def format_rq4_table(rq4_data):
    tex = [
        "% --- RQ4 CONCURRENCY SWEEP TABLE ---",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{High-resolution concurrency sweep ($C=50..100$, $N=1000$) mapping event-loop tail latency transition.}",
        "\\label{tab:rq4_scalability}",
        "\\begin{tabular}{ccccccc}",
        "\\toprule",
        "\\textbf{C} & \\textbf{N} & \\textbf{Total $P_{50}$ (ms)} & \\textbf{Total $P_{99}$ (ms)} & \\textbf{Inference $P_{99}$ (ms)} & \\textbf{Queue $P_{99}$ (ms)} & \\textbf{Mean Queue Share} \\\\",
        "\\midrule"
    ]

    items = rq4_data if isinstance(rq4_data, list) else []
    for r in items:
        if not isinstance(r, dict):
            continue
        c_val = r.get("c")
        n_val = r.get("n")
        p50 = r.get("p50_total", 0.0)
        p99 = r.get("p99_total", 0.0)
        inf_p99 = r.get("p99_cpu", 0.0)
        q_p99 = r.get("p99_queue", 0.0)
        q_share = r.get("mean_queue_ratio_pct", 0.0)
        tex.append(f"\\textbf{{{c_val}}} & {n_val} & {p50:.2f} & {p99:.2f} & {inf_p99:.2f} & {q_p99:.2f} & \\textbf{{{q_share:.1f}\\%}} \\\\")

    tex.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}\n\n"])
    return "\n".join(tex)


def main():
    if not MANIFEST_FILE.exists():
        print(f"❌ Error: Manifest file not found at {MANIFEST_FILE}")
        return

    with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    content = [
        "% ==================================================",
        "% AUTOMATICALLY GENERATED LATEX SNIPPETS FROM MANIFEST",
        "% ==================================================\n"
    ]

    if "dataset_protocol" in manifest and manifest["dataset_protocol"]:
        content.append(format_dataset_table(manifest["dataset_protocol"]))

    if "rq1_classification_quality" in manifest and manifest["rq1_classification_quality"]:
        content.append(format_rq1_table(manifest["rq1_classification_quality"]))

    if "rq4_scalability_concurrency" in manifest and manifest["rq4_scalability_concurrency"]:
        content.append(format_rq4_table(manifest["rq4_scalability_concurrency"]))

    with open(OUTPUT_TEX, "w", encoding="utf-8") as f:
        f.write("\n".join(content))

    print(f"✅ Regenerated valid LaTeX snippet: {OUTPUT_TEX.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()