import argparse, os, time, json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from joblib import dump
from src.utils.io import seed_everything, ensure_dir, file_size_mb
from src.data.text_clean import clean_text

def get_args():
    parser = argparse.ArgumentParser(description="Train TF-IDF + LogisticRegression")
    parser.add_argument("--train", type=str, required=True, help="Path to train.parquet")
    parser.add_argument("--valid", type=str, required=True, help="Path to valid.parquet")
    parser.add_argument("--out",   type=str, required=True, help="Output dir, e.g. models/tfidf")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--ngram_min", type=int, default=1, help="Lower bound for n-grams")
    parser.add_argument("--ngram_max", type=int, default=2, help="Upper bound for n-grams")
    parser.add_argument("--min_df",    type=int,   default=5,   help="Min doc freq")
    parser.add_argument("--max_df",    type=float, default=0.9, help="Max doc freq (fraction)")
    parser.add_argument("--sublinear_tf", action="store_true",
                        help="Use sublinear TF scaling if flag is present (default False)")
    parser.add_argument("--analyzer", choices=["word", "char_wb"], default="word",
                        help='Feature analyzer: "word" or "char_wb"')
    parser.add_argument("--C", type=float, default=1.0, help="Inverse regularization strength")
    parser.add_argument("--class_weight", choices=["balanced", "none"], default="balanced",
                        help='"balanced" or "none"')
    args = parser.parse_args()
    args.ngram_range = (args.ngram_min, args.ngram_max)
    args.class_weight = None if args.class_weight == "none" else "balanced"

    return args

def load_data(args) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(args.train, columns=["text", "label"])
    valid = pd.read_parquet(args.valid, columns=["text", "label"])

    if len(train) == 0:
        raise ValueError(f"Empty train dataset: {args.train}")
    if len(valid) == 0:
        raise ValueError(f"Empty valid dataset: {args.valid}")

    train["text"] = train["text"].astype("string")
    valid["text"] = valid["text"].astype("string")
    train["label"] = train["label"].astype("int8")
    valid["label"] = valid["label"].astype("int8")

    return train, valid

def main():
    args = get_args()
    seed_everything(args.seed)
    train, valid = load_data(args)

    train["text"] = train["text"].map(lambda s: clean_text(s, "tfidf"))
    valid["text"] = valid["text"].map(lambda s: clean_text(s, "tfidf"))

    vectorizer = TfidfVectorizer(
        ngram_range=(args.ngram_min, args.ngram_max),
        min_df=args.min_df,
        max_df=args.max_df,
        sublinear_tf=args.sublinear_tf,
        dtype=np.float32,
        analyzer=args.analyzer,
    )
    x_train = vectorizer.fit_transform(train["text"])
    x_valid = vectorizer.transform(valid["text"])

    y_train = train["label"].astype(int).to_numpy()
    y_valid = valid["label"].astype(int).to_numpy()

    model = LogisticRegression(
        max_iter=200,
        solver="lbfgs",
        C=args.C,
        class_weight=args.class_weight,  # already normalized in get_args()
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    proba_valid = model.predict_proba(x_valid)[:, 1]
    pred_valid = (proba_valid >= 0.5).astype(int)

    acc = accuracy_score(y_valid, pred_valid)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_valid, pred_valid, average="binary", zero_division=0
    )
    auc = roc_auc_score(y_valid, proba_valid)
    cm = confusion_matrix(y_valid, pred_valid).tolist()

    t0 = time.perf_counter()
    _ = model.predict_proba(x_valid)
    t1 = time.perf_counter()
    docs_sec = len(valid) / (t1 - t0) if t1 > t0 else float("nan")
    print(f"[step8] Validation throughput: {docs_sec:.1f} docs/sec")

    ensure_dir(args.out)

    vectorizer_path = os.path.join(args.out, "vectorizer.joblib")
    classifier_path = os.path.join(args.out, "classifier.joblib")
    meta_path = os.path.join(args.out, "metadata.json")

    dump(vectorizer, vectorizer_path)
    dump(model, classifier_path)

    metadata = {
        "vectorizer": {
            "ngram_range": (args.ngram_min, args.ngram_max),
            "min_df": args.min_df,
            "max_df": args.max_df,
            "sublinear_tf": args.sublinear_tf,
            "analyzer": args.analyzer,
        },
        "classifier": {
            "C": args.C,
            "class_weight": args.class_weight,
            "max_iter": 200,
            "solver": "lbfgs",
        },
        "metrics": {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "roc_auc": float(auc),
            "confusion_matrix": cm,
        },
        "n_features": int(x_train.shape[1]),
        "train_size": int(len(train)),
        "valid_size": int(len(valid)),
        "docs_per_sec": float(docs_sec),
        "seed": args.seed,
    }

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n[step10] === TF-IDF + LogisticRegression Report ===")
    print(f"Train size: {len(train)}, Valid size: {len(valid)}")
    print(f"n_features: {x_train.shape[1]}")
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {rec:.3f}")
    print(f"F1 score : {f1:.3f}")
    print(f"ROC AUC  : {auc:.3f}")
    print(f"Confusion matrix: {cm}")
    print(f"Throughput: {docs_sec:.1f} docs/sec")
    print(f"Artifacts saved to {args.out}")

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())