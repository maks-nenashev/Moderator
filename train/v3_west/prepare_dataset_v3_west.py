import pandas as pd
from pathlib import Path
import argparse

# -------- CONFIG --------
# Western European block (v3/v3.1)
BASE_DIR_V3 = Path("data/raw/v3_sexual/WEST")
BASE_DIR_V3_1 = Path("data/raw/v3.1_sexual/WEST")

# Output directory
OUTPUT_BASE = Path("data/processed")

# Files for base v3 
LANG_FILES = ["EN.csv", "DE.csv", "FR.csv", "ES.csv", "NL.csv", "IT.csv", "PT.csv"]

# Files for v3.1 slang augmentation 
SLANG_FILES = ["EN_slang.csv", "DE_slang.csv", "FR_slang.csv", "ES_slang.csv", "NL_slang.csv", "IT_slang.csv", "PT_slang.csv"]

EXPECTED_COLUMNS = {
    "text", "label", "language", "country_group", 
    "source", "confidence_hint", "notes"
}

def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f" Warning: Missing file {path.name}, skipping...")
        return pd.DataFrame()

    df = pd.read_csv(path)
    
    # Cleaning header artifacts if they exist
    if 'label' in df.columns:
        df = df[df['label'].astype(str) != 'label']
        df['label'] = df['label'].astype(int)
    
    # If columns are missing, add them with default values for WEST
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = "slang_augmentation" if col == "source" else "WEST"
            
    return df[list(EXPECTED_COLUMNS)]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default="3.0", help="Version to prepare: 3.0 or 3.1")
    args = parser.parse_args()

    dfs = []
    print(f" Starting dataset preparation for version {args.version}...")

    # 1. Load base v3 datasets (Symmetric to Phase 1 in CEE)
    print("\n--- Loading Base WEST (v3) ---")
    for f in LANG_FILES:
        path = BASE_DIR_V3 / f
        df = load_csv(path)
        if not df.empty:
            print(f" {f}: {len(df)} rows")
            dfs.append(df)

    # 2. If version is 3.1, load slang augmentation datasets (Symmetric to Phase 2 in CEE)
    if args.version == "3.1":
        print("\n--- Loading Hardcore Slang (v3.1) ---")
        for f in SLANG_FILES:
            path = BASE_DIR_V3_1 / f
            df = load_csv(path)
            if not df.empty:
                print(f" {f}: {len(df)} rows")
                dfs.append(df)

    if not dfs:
        print("No data loaded!")
        return

    full_df = pd.concat(dfs, ignore_index=True)
    full_df = full_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # 3. Save the prepared dataset (Symmetric naming convention)
    suffix = "3_1" if args.version == "3.1" else "3"
    out_path = OUTPUT_BASE / f"dataset_v3_{suffix}_west.csv"

    print(f"\nFinal dataset stats:")
    print(f"Total rows: {len(full_df)}")
    print(full_df["label"].value_counts(normalize=True))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(out_path, index=False)
    print(f"\nSaved to: {out_path.resolve()}")

if __name__ == "__main__":
    main()