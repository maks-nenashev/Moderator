# train/prepare_dataset_v4_cee.py

import pandas as pd
from pathlib import Path


# -------- CONFIG --------

BASE_DIR = Path("data/raw/v4_sexual/CEE")
OUTPUT_PATH = Path("data/processed/dataset_v4_cee.csv")

LANG_FILES = [
    "PL.csv",
    "CZ.csv",
    "SK.csv",
    "HU.csv",
    "RO.csv",
]

EXPECTED_COLUMNS = {
    "text",
    "label",
    "language",
    "country_group",
    "source",
    "confidence_hint",
    "notes",
}


# -------- LOAD & VALIDATE --------

def load_language_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)

    # column validation
    if set(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            f"Invalid columns in {path.name}\n"
            f"Expected: {EXPECTED_COLUMNS}\n"
            f"Found: {set(df.columns)}"
        )

    # country group validation
    if df["country_group"].nunique() != 1 or df["country_group"].iloc[0] != "WEST":
        raise ValueError(
            f"Invalid country_group in {path.name}. "
            f"Expected only 'WEST'."
        )

    # basic sanity checks
    if df["text"].isna().any():
        raise ValueError(f"NaN text values found in {path.name}")

    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError(f"Invalid label values in {path.name}")

    return df


# -------- MAIN PIPELINE --------

def main():
    dfs = []

    print("Loading v3 WEST sexual intent datasets...\n")

    for filename in LANG_FILES:
        path = BASE_DIR / filename
        print(f"→ loading {filename}")
        df = load_language_csv(path)
        print(f"  rows: {len(df)} | language: {df['language'].iloc[0]}")
        dfs.append(df)

    print("\nConcatenating datasets...")
    full_df = pd.concat(dfs, ignore_index=True)

    print("Shuffling dataset...")
    full_df = full_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    print("\nFinal dataset stats:")
    print(f"Total rows: {len(full_df)}")
    print("Label distribution:")
    print(full_df["label"].value_counts(normalize=True))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved dataset to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
