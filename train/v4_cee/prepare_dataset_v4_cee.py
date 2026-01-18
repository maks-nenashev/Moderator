import pandas as pd
from pathlib import Path
import argparse

# -------- CONFIG --------
#  Note: This script prepares datasets for versions 4.0 and 4.1
BASE_DIR_V4 = Path("data/raw/v4_sexual/CEE")
BASE_DIR_V4_1 = Path("data/raw/v4.1_sexual/CEE")

# Output directory
OUTPUT_BASE = Path("data/processed")

# Files for base v4
LANG_FILES = ["PL.csv", "CZ.csv", "SK.csv", "HU.csv", "RO.csv"]

# Files for v4.1 slang augmentation
SLANG_FILES = ["PL_slang.csv", "CZ_slang.csv", "SK_slang.csv", "HU_slang.csv", "RO_slang.csv"]

EXPECTED_COLUMNS = {
    "text", "label", "language", "country_group", 
    "source", "confidence_hint", "notes"
}

def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f" Warning: Missing file {path.name}, skipping...")
        return pd.DataFrame()

    df = pd.read_csv(path)
    
    # If columns are missing, add them with default values
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = "slang_augmentation" if col == "source" else "CEE"
            
    return df[list(EXPECTED_COLUMNS)]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default="4.0", help="Version to prepare: 4.0 or 4.1")
    args = parser.parse_args()

    dfs = []
    print(f" Starting dataset preparation for version {args.version}...")

    # 1. Load base v4 datasets
    print("\n--- Loading Base CEE (v4) ---")
    for f in LANG_FILES:
        path = BASE_DIR_V4 / f
        df = load_csv(path)
        if not df.empty:
            print(f" {f}: {len(df)} rows")
            dfs.append(df)

    # 2. If version is 4.1, load slang augmentation datasets
    if args.version == "4.1":
        print("\n--- Loading Hardcore Slang (v4.1) ---")
        for f in SLANG_FILES:
            path = BASE_DIR_V4_1 / f
            df = load_csv(path)
            if not df.empty:
                print(f" {f}: {len(df)} rows")
                dfs.append(df)

    if not dfs:
        print("No data loaded!")
        return

    full_df = pd.concat(dfs, ignore_index=True)
    full_df = full_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    #  3. Save the prepared dataset
    suffix = "4_1" if args.version == "4.1" else "4"
    out_path = OUTPUT_BASE / f"dataset_v4_{suffix}_cee.csv"

    print(f"\nFinal dataset stats:")
    print(f"Total rows: {len(full_df)}")
    print(full_df["label"].value_counts(normalize=True))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(out_path, index=False)
    print(f"\nSaved to: {out_path.resolve()}")

if __name__ == "__main__":
    main()