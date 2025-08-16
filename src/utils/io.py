import os, glob, random, math, pathlib
from typing import Iterable, Literal, Iterator, Optional, List
import numpy as np
import pandas as pd

def seed_everything(seed:int) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def sniff_format(path:str) -> Literal['csv', 'parquet', 'jsonl']:
    ext = pathlib.Path(path).suffix.lower()
    if ext == ".csv":
        return "csv"
    elif ext == ".parquet":
        return "parquet"
    elif ext == ".jsonl":
        return "jsonl"
    else:
        raise ValueError(f"Unknown file format: {ext}")

def stream_read(path:str,usecols: Optional[List[str]], chunksize:int) -> Iterator[pd.DataFrame]:
    fmt = sniff_format(path)
    if chunksize <= 0:
        raise ValueError("Chunksize must be a positive integer")

    if fmt == "csv":
        reader = pd.read_csv(path, usecols= usecols, chunksize=chunksize, low_memory = True, encoding_errors= "ignore", lineterminator = "\n")
        for chunk in reader:
            yield chunk
    elif fmt == "jsonl":
        reader = pd.read_json(path, lines = True, chunksize=chunksize)
        for chunk in reader:
            yield chunk
    elif fmt == "parquet":
        raise ValueError("Parquet format is not supported:convert to CSV/JSONL or load fully.")

def iter_paths(pattern_or_path:str) -> List[str]:
    if os.path.exists(pattern_or_path) and os.path.isfile(pattern_or_path):
        return [pattern_or_path]
    else:
        files = sorted(glob.glob(pattern_or_path))
    if not files:
        raise FileNotFoundError(f"No files found matching pattern: {pattern_or_path}")

    return files

def save_parquet(df:pd.DataFrame, path:str) -> None:
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

def file_size_mb(path:str) -> float:
    size = os.path.getsize(path)
    return round(size / (1024 ** 2), 3)

def ensure_dir(path:str) -> None:
    pathlib.Path(path).mkdir(parents = True, exist_ok = True)

if __name__ == "__main__":
    seed_everything(42)
    print(sniff_format("x.csv"))
    print(sniff_format("x.jsonl"))
    print(sniff_format("x.parquet"))
    try:
        for i, chunk in enumerate(stream_read("data/raw/sample.csv", ["review", "recommended"], 10000)):
          print("chunk", i, chunk.shape)
          if i == 1: break
    except Exception as e:
      print("stream error:", e)