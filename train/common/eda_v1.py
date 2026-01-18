# python train/eda_v1.py 
# Exploratory Data Analysis for dataset_v1.csv

import pandas as pd
import numpy as np

DATA_PATH = "data/processed/dataset_v1.csv"

df = pd.read_csv(DATA_PATH)

labels = ["toxicity", "hate", "sexual", "violence", "scam", "spam"]

print("=== BASIC INFO ===")
print(df.info())

print("\n=== DATASET SHAPE ===")
print(df.shape)

print("\n=== LABEL MEANS (class prevalence) ===")
print(df[labels].mean().sort_values(ascending=False))

print("\n=== LABEL COUNTS ===")
print(df[labels].sum().sort_values(ascending=False))

print("\n=== MULTI-LABEL ANALYSIS ===")
df["label_sum"] = df[labels].sum(axis=1)
print(df["label_sum"].value_counts().sort_index())

print("\n=== TEXT LENGTH ANALYSIS ===")
df["text_len"] = df["text"].astype(str).apply(len)

print(df["text_len"].describe())

print("\n=== TEXT LENGTH QUANTILES ===")
print(df["text_len"].quantile([0.5, 0.75, 0.9, 0.95, 0.99]))
