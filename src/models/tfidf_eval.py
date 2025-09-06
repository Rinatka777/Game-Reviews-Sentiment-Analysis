import argparse, json, os
import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from src.data.text_clean import clean_text

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test", required=True)
    p.add_argument("--model_dir", required=True, help="e.g. models/tfidf_word_1_2")
    return p.parse_args()

def main():
    args = get_args()
    test = pd.read_parquet(args.test, columns=["text","label"])
    test["text"] = test["text"].astype("string").map(lambda s: clean_text(s, "tfidf"))
    y_true = test["label"].astype(int).to_numpy()

    vect = load(os.path.join(args.model_dir, "vectorizer.joblib"))
    clf  = load(os.path.join(args.model_dir, "classifier.joblib"))

    X = vect.transform(test["text"])
    proba = clf.predict_proba(X)[:, 1]
    pred  = (proba >= 0.5).astype(int)

    acc = accuracy_score(y_true, pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
    auc = roc_auc_score(y_true, proba)
    cm  = confusion_matrix(y_true, pred).tolist()

    print("\n=== TF-IDF Test Metrics ===")
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {rec:.3f}")
    print(f"F1 score : {f1:.3f}")
    print(f"ROC AUC  : {auc:.3f}")
    print(f"Confusion matrix: {cm}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
