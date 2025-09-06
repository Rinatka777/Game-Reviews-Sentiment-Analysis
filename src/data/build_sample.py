import argparse
import os
from typing import Iterable, Optional, Tuple, List
from sklearn.model_selection import train_test_split

import pandas as pd
from src.utils.io import (
    seed_everything,
    iter_paths,
    stream_read, ensure_dir, save_parquet,
)
from src.utils.io import seed_everything, iter_paths, stream_read, save_parquet, ensure_dir, file_size_mb

TEXT_CANDS: List[str] = ["review", "review_text", "text", "content", "body"]
LABEL_CANDS: List[str] = [
    "recommended", "voted_up", "review_score",
    "label", "sentiment", "rating", "score", "target"
]



def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_pattern", required=True)
    p.add_argument("--out", dest="out_dir", required=True)
    p.add_argument("--n", dest="n_total", type=int, required=True)
    p.add_argument("--seed", dest="seed", type=int, default=42)
    p.add_argument("--chunksize", dest="chunksize", type=int, default=100_000)
    return p.parse_args()


def detect_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    col_map = {c.strip(): c for c in df.columns}
    lower_index = {c.strip().lower(): c for c in df.columns}

    for name in candidates:
        if name.lower() in lower_index:
            return lower_index[name.lower()]
    for name in candidates:
        for lc, original in lower_index.items():
            if name.lower() in lc:
                return original
    return None



def find_columns_from_first_chunk(files: List[str], chunksize: int = 10_000) -> Tuple[str, str]:
    chunk_iter = stream_read(files[0], usecols=None, chunksize=chunksize)
    first_chunk = next(chunk_iter, None)
    if first_chunk is None or first_chunk.empty:
        raise RuntimeError("First file yielded no data")
    text_col = detect_column(first_chunk, TEXT_CANDS)
    label_col = detect_column(first_chunk, LABEL_CANDS)
    if text_col is None or label_col is None:
        raise ValueError(f"Could not find text/label columns. Available: {list(first_chunk.columns)}")
    return text_col, label_col


def filter_lengths(df: pd.DataFrame, text_col: str, min_len: int = 5, max_len: int = 2000) -> pd.DataFrame:
    lengths = df[text_col].astype(str).str.len()
    mask = (lengths >= min_len) & (lengths <= max_len)
    return df.loc[mask]


def remaining(got_0: int, got_1: int, per_class_target: int) -> Tuple[int, int]:
    need0 = max(0, per_class_target - got_0)
    need1 = max(0, per_class_target - got_1)
    return need0, need1


def clip_to_need(df_class: pd.DataFrame, need: int, seed: int) -> pd.DataFrame:
    if need <= 0:
        return df_class.iloc[0:0]
    if len(df_class) <= need:
        return df_class
    return df_class.sample(n=need, random_state=seed)

def coerce_label_to_binary(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.astype(int)

    if pd.api.types.is_numeric_dtype(s):
        return (s.astype(float) > 0).astype("int8")

    lowered = s.astype(str).str.strip().str.lower()
    pos = {"recommended", "positive", "pos", "true", "yes", "y", "1"}
    neg = {"not recommended", "negative", "neg", "false", "no", "n", "0"}
    out = pd.Series(pd.NA, index=s.index, dtype="Int8")
    out[lowered.isin(pos)] = 1
    out[lowered.isin(neg)] = 0

    return out.astype("Int8")


def normalize_chunk(chunk: pd.DataFrame, text_col: str, label_col: str) -> pd.DataFrame:
    chunk = chunk.dropna(subset=[text_col, label_col]).copy()
    chunk[label_col] = coerce_label_to_binary(chunk[label_col])
    chunk = chunk.dropna(subset=[label_col])
    chunk[label_col] = chunk[label_col].astype("int8")
    chunk = filter_lengths(chunk, text_col)
    return chunk[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})



def main() -> None:
    args = get_args()
    seed_everything(args.seed)

    files: List[str] = iter_paths(args.in_pattern)
    if not files:
        raise FileNotFoundError(f"No input files match: {args.in_pattern}")
    print(f"Found {len(files)} file(s)")

    text_col, label_col = find_columns_from_first_chunk(files)
    print(f"Using text_col='{text_col}', label_col='{label_col}'")

    per_class_target = args.n_total // 2
    got_0 = 0
    got_1 = 0
    buf_0: List[pd.DataFrame] = []
    buf_1: List[pd.DataFrame] = []
    early_stop = False

    for path in files:
        for chunk in stream_read(path, usecols=[text_col, label_col], chunksize=args.chunksize):
            chunk = normalize_chunk(chunk, text_col, label_col)
            c0 = chunk[chunk["label"] == 0]
            c1 = chunk[chunk["label"] == 1]

            need0, need1 = remaining(got_0, got_1, per_class_target)

            if need0 == 0 and need1 == 0:
                early_stop = True
                break

            take0 = clip_to_need(c0, need0, seed=args.seed)
            take1 = clip_to_need(c1, need1, seed=args.seed)

            if len(take0) > 0:
                buf_0.append(take0)
                got_0 += len(take0)
            if len(take1) > 0:
                buf_1.append(take1)
                got_1 += len(take1)

        if early_stop:
            break

    print(f"Collected -> class0={got_0}, class1={got_1}, target_each={per_class_target}")

    df0 = pd.concat(buf_0, ignore_index=True) if buf_0 else pd.DataFrame(columns=["text", "label"])
    df1 = pd.concat(buf_1, ignore_index=True) if buf_1 else pd.DataFrame(columns=["text", "label"])

    min_class = min(len(df0), len(df1))
    if min_class == 0:
        raise ValueError(f"One class is empty. class0={len(df0)}, class1={len(df1)}")

    if len(df0) > min_class:
        df0 = df0.sample(n=min_class, random_state=args.seed)
    if len(df1) > min_class:
        df1 = df1.sample(n=min_class, random_state=args.seed)

    sample = pd.concat([df0, df1], ignore_index=True)
    sample = sample.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    print(f"[step10] combined={len(sample)} | class0={len(df0)} | class1={len(df1)}")
    print(f"[step10] pos_rate={float(sample['label'].mean()):.3f}")
    print(f"[step10] avg_len={float(sample['text'].astype(str).str.len().mean()):.1f}")

    if list(sample.columns) != ["text", "label"]:
        raise ValueError(f"Unexpected columns: {list(sample.columns)}")
    if not sample["label"].dropna().isin([0, 1]).all():
        raise ValueError("Labels must be strictly 0 or 1")

    n0 = int((sample["label"] == 0).sum())
    n1 = int((sample["label"] == 1).sum())
    if n0 != n1:
        raise ValueError(f"Classes not balanced: class0={n0}, class1={n1}")

    print(f"[step10] Contract OK → total={len(sample)}, per_class={n0}")

    x = sample["text"]
    y = sample["label"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        x, y,
        test_size=0.20,
        stratify=y,
        random_state=args.seed,
    )

    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=args.seed,
    )

    train = pd.DataFrame({"text": X_train, "label": y_train})
    valid = pd.DataFrame({"text": X_valid, "label": y_valid})
    test = pd.DataFrame({"text": X_test, "label": y_test})

    def pos_rate(df):
        return float(df["label"].mean())

    print(f"train={len(train)} valid={len(valid)} test={len(test)}")
    print(f"pos_rate train={pos_rate(train):.3f} valid={pos_rate(valid):.3f} test={pos_rate(test):.3f}")

    train["text"] = train["text"].astype("string")
    valid["text"] = valid["text"].astype("string")
    test["text"] = test["text"].astype("string")

    train["label"] = train["label"].astype("int8")
    valid["label"] = valid["label"].astype("int8")
    test["label"] = test["label"].astype("int8")

    out_dir = args.out_dir
    ensure_dir(out_dir)

    saved = []
    for name, df in [("train", train), ("valid", valid), ("test", test)]:
        path = os.path.join(out_dir, f"{name}.parquet")
        tmp = path + ".tmp"
        save_parquet(df, tmp)
        os.replace(tmp, path)
        saved.append((name, path))

    for name, path in saved:
        print(f"Saved: {path} ({len(pd.read_parquet(path))} rows)  |  size ~ {file_size_mb(path)} MB")

    def pos_rate(df: pd.DataFrame) -> float:
        return float(df["label"].mean())

    def avg_len(df: pd.DataFrame) -> float:
        return float(df["text"].astype(str).str.len().mean())

    train_path = os.path.join(args.out_dir, "train.parquet")
    valid_path = os.path.join(args.out_dir, "valid.parquet")
    test_path = os.path.join(args.out_dir, "test.parquet")

    train_chk = pd.read_parquet(train_path, columns=["text", "label"])
    valid_chk = pd.read_parquet(valid_path, columns=["text", "label"])
    test_chk = pd.read_parquet(test_path, columns=["text", "label"])

    print(f"Train: {len(train_chk):>7} | pos_rate={pos_rate(train_chk):.3f} | avg_len={avg_len(train_chk):.1f}")
    print(f"Valid: {len(valid_chk):>7} | pos_rate={pos_rate(valid_chk):.3f} | avg_len={avg_len(valid_chk):.1f}")
    print(f"Test : {len(test_chk):>7} | pos_rate={pos_rate(test_chk):.3f} | avg_len={avg_len(test_chk):.1f}")
    print("Saved →")
    print(" ", train_path)
    print(" ", valid_path)
    print(" ", test_path)


if __name__ == "__main__":
    main()
