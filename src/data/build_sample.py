import argparse
import os
from typing import Iterable, Optional, Tuple, List

import pandas as pd
from src.utils.io import (
    seed_everything,
    iter_paths,
    stream_read,
)

TEXT_CANDS: List[str] = ["review", "review_text", "text", "content", "body"]
LABEL_CANDS: List[str] = ["recommended", "label", "sentiment", "rating", "score", "target"]


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_pattern", required=True)
    p.add_argument("--out", dest="out_dir", required=True)
    p.add_argument("--n", dest="n_total", type=int, required=True)
    p.add_argument("--seed", dest="seed", type=int, default=42)
    p.add_argument("--chunksize", dest="chunksize", type=int, default=100_000)
    return p.parse_args()


def detect_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for name in candidates:
        if name in df.columns:
            return name
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


def normalize_chunk(chunk: pd.DataFrame, text_col: str, label_col: str) -> pd.DataFrame:
    chunk = chunk.dropna(subset=[text_col, label_col]).copy()
    chunk[label_col] = chunk[label_col].astype(int)
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


if __name__ == "__main__":
    main()
