import argparse
import pandas as pd
import joblib

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--vectorizer", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.data, on_bad_lines='skip')

    model = joblib.load(args.model)
    vectorizer = joblib.load(args.vectorizer)

    X = vectorizer.transform(df["text"])
    scores = model.predict_proba(X)[:, 1]

    out_df = df[["text", "label", "language"]].copy()
    out_df["score"] = scores

    out_df.to_csv(args.out, index=False)
    print(f"Saved scores to {args.out}")
    print(out_df.groupby("language")["score"].describe())

if __name__ == "__main__":
    main()

