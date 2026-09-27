#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dataset Protocol Inspector
Generates exact sample counts, deduplication metrics, class balances (Pos/Neg),
train/test splits (80/20, seed=42), and LaTeX/Markdown tables for Section 4.2.
Author: Maksym Nenashev (Systems Engineer)
"""

import hashlib
import json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "artifacts"

DATASET_MAPPING = [
    {"cluster": "Human Trafficking", "engine": "v1", "file": "trafficking_v1.csv", "langs": "Multilingual / Specialized"},
    {"cluster": "Western European", "engine": "v3 / v3.1 / v3.2", "file": "west_v3_base.csv", "langs": "EN, FR, DE, ES, IT, PT, NL"},
    {"cluster": "Central & Eastern Europe", "engine": "v4 / v4.1 / v4.2", "file": "cee_v4_base.csv", "langs": "PL, CS, SK, HU, RO"},
    {"cluster": "Baltic", "engine": "v5 / v5.1 / v5.2", "file": "baltic_v5_base.csv", "langs": "ET, LV, LT"},
    {"cluster": "CIS", "engine": "v6 / v6.1 / v6.2", "file": "cis_v6_base.csv", "langs": "RU, UK, BE, KK"},
    {"cluster": "Nordic", "engine": "v7 / v7.1 / v7.2", "file": "nordic_v7_base.csv", "langs": "SV, NO, DA, FI"},
    {"cluster": "Balkan", "engine": "v8 / v8.1 / v8.2", "file": "balkan_v8_base.csv", "langs": "BG, HR, SR, SL, EL, TR"},
    {"cluster": "Caucasus", "engine": "v9 / v9.1 / v9.2", "file": "caucasus_v9_base.csv", "langs": "KA, HY, AZ"}
]

def calculate_sha256(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()[:12]

def analyze_datasets():
    results = []
    total_raw_global = 0
    total_clean_global = 0

    print("==================================================================")
    print("📊 GENERATING DATASET PROTOCOL METRICS (80/20 Split, Seed=42)")
    print("==================================================================")

    for item in DATASET_MAPPING:
        file_path = DATA_DIR / item["file"]
        if not file_path.exists():
            print(f"⚠️ Warning: File {item['file']} not found in {DATA_DIR}")
            continue

        file_sha = calculate_sha256(file_path)
        df_raw = pd.read_csv(file_path, on_bad_lines="skip", engine="python")
        raw_count = len(df_raw)
        total_raw_global += raw_count

        # 1. Очистка и полная строковая дедупликация
        df_clean = df_raw.dropna(subset=["text", "label"]).copy()
        df_clean["text"] = df_clean["text"].astype(str).str.strip()
        df_clean = df_clean.drop_duplicates(subset=["text"])
        clean_count = len(df_clean)
        total_clean_global += clean_count
        duplicates_removed = raw_count - clean_count

        # 2. Баланс классов (0 = Neg / Clean, 1 = Pos / Toxic)
        y = df_clean["label"].values.astype(int)
        pos_count = int((y == 1).sum())
        neg_count = int((y == 0).sum())

        # 3. Стратифицированный сплит 80/20 (Seed = 42)
        train_df, test_df = train_test_split(
            df_clean, test_size=0.20, random_state=42, stratify=y
        )

        train_pos = int((train_df["label"] == 1).sum())
        train_neg = int((train_df["label"] == 0).sum())
        test_pos = int((test_df["label"] == 1).sum())
        test_neg = int((test_df["label"] == 0).sum())

        results.append({
            "cluster": item["cluster"],
            "engines": item["engine"],
            "file": item["file"],
            "sha256": file_sha,
            "languages": item["langs"],
            "raw_n": raw_count,
            "clean_n": clean_count,
            "duplicates_removed": duplicates_removed,
            "pos_n": pos_count,
            "neg_n": neg_count,
            "train_n": len(train_df),
            "train_pos": train_pos,
            "train_neg": train_neg,
            "test_n": len(test_df),
            "test_pos": test_pos,
            "test_neg": test_neg
        })

    # Вывод Markdown таблицы
    print("\n--- DATASET PROTOCOL TABLE (MARKDOWN FOR PAPER) ---")
    print("| Regional Cluster | Engines | Total Clean N | Train (80%) [Pos/Neg] | Hold-out Test (20%) [Pos/Neg] | Class Ratio (Pos:Neg) | SHA-256 |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in results:
        ratio = f"{r['pos_n']/r['clean_n']*100:.1f}% : {r['neg_n']/r['clean_n']*100:.1f}%"
        train_str = f"{r['train_n']} [{r['train_pos']}/{r['train_neg']}]"
        test_str = f"{r['test_n']} [{r['test_pos']}/{r['test_neg']}]"
        print(f"| **{r['cluster']}** | `{r['engines']}` | {r['clean_n']:,} | {train_str} | {test_str} | {ratio} | `{r['sha256']}` |")

    print(f"\n📈 GLOBAL SUMMARY: Raw Total = {total_raw_global:,} | Clean Deduplicated Total = {total_clean_global:,}")

    # Сохранение лога протокола в JSON
    output_file = OUTPUT_DIR / "dataset_protocol_summary.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✅ Saved full protocol metadata to {output_file.relative_to(BASE_DIR)}")

if __name__ == "__main__":
    analyze_datasets()