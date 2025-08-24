import argparse
import os, math, random
from typing import List, Iterable, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.io import seed_everything, iter_paths, stream_read, save_parquet, ensure_dir
from src.data.text_clean import clean_text

TEXT_CANDS  = ["review", "review_text", "text", "content", "body"]
LABEL_CANDS = ["recommended", "label", "sentiment", "rating", "score", "target"]


def get_args():
    parser = argparse.ArgumentParser(description="Process some data")
    parser.add_argument("--in", type=str, required=True,
                        help="Input path or glob, e.g. data/raw/*.jsonl")
    parser.add_argument("--out", type=str, required=True,
                        help="Output directory, e.g. data/samples/steam_200k")
    parser.add_argument("--n", type=int, required=True,
                        help="Target total rows, e.g. 200000")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--chunksize", type=int, default=100_000,
                        help="Rows per chunk (default: 100000)")

    args = parser.parse_args()

    return args

def detect_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for name in candidates:
        if name in df.columns:
            return name
    return None

def find_columns_from_first_chunk(files: list[str], chunksize: int = 10_000) -> tuple[str, str]:
    chunk_iter = stream_read(files[0], usecols=None, chunksize=chunksize)
    try:
        first_chunk = next(chunk_iter)
    except StopIteration:
        raise RuntimeError("First file yielded no data; check path/format.")

    text_col = detect_column(first_chunk, TEXT_CANDS)
    label_col = detect_column(first_chunk, LABEL_CANDS)
    if text_col is None or label_col is None:
        raise ValueError(
            f"Could not find text/label columns. "
            f"Seen columns: {list(first_chunk.columns)}"
        )
    return text_col, label_col


def filter_lengths(df: pd.DataFrame, text_col: str, min_len: int = 5, max_len: int = 2000) -> pd.DataFrame:
    lengths = df[text_col].astype(str).str.len()
    mask = (lengths >= min_len) & (lengths <= max_len)
    return df.loc[mask]

def per_class_quota(total:int, pos_value) -> dict:
    per_class = total // 2
    remainder = total % 2
    quotas = {
        pos_value: per_class + remainder,
        f"not_{pos_value}": per_class
    }
    return quotas
def remaining(got_0, got_1, per_class_quota):

def main():
    args = get_args()
    seed_everything(args.seed)
    files = iter_paths(args.in)
    text_col, label_col = find_columns_from_first_chunk(files)
    print(f"Detected text column = {text_col}, label column = {label_col}")

    er_class_target = args.n // 2
    got_0 = 0
    got_1 = 0
    buf_0: list[pd.DataFrame] = []
    buf_1: list[pd.DataFrame] = []
    early_stop = False

