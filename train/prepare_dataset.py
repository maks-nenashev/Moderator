import pandas as pd
import os
import glob
from pathlib import Path

def prepare(version):
    root_dir = Path(__file__).parent.parent
    raw_dir = root_dir / "data" / "raw"
    
    # Логика выбора папок в зависимости от версии
    if version == "3":
        paths = glob.glob(str(raw_dir / "v3_sexual/WEST/*.csv"))
        out_name = "dataset_v3_west.csv"
    elif version == "3.1":
        # Склеиваем базу + сленг для v3.1
        paths = glob.glob(str(raw_dir / "v3_sexual/WEST/*.csv")) + \
                glob.glob(str(raw_dir / "v3.1_sexual/WEST/*.csv"))
        out_name = "dataset_v3_1_west.csv"
    elif version == "4":
        paths = glob.glob(str(raw_dir / "v4_sexual/CEE/*.csv"))
        out_name = "dataset_v4_cee.csv"
    else:
        print(f"❌ Версия {version} не поддерживается.")
        return

    dfs = []
    for p in paths:
        df = pd.read_csv(p)
        print(f"🔹 Загружено {os.path.basename(p)}: {len(df)} строк")
        dfs.append(df)
    
    final_df = pd.concat(dfs, ignore_index=True)
    out_path = root_dir / "data" / "processed" / out_name
    final_df.to_csv(out_path, index=False)
    
    print(f"\n✅ Итоговый датасет {version} сохранен: {out_path}")
    print(f"📊 Всего строк: {len(final_df)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True, help="3, 3.1 or 4")
    args = parser.parse_args()
    prepare(args.version)
