import os, glob, random, math, pathlib
from typing import Iterable, Literal, Iterator, Optional, List
import numpy as np
import pandas as pd

def seed_everything(seed:int) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def sniff_format(path:str) -> Literal['csv', 'parquet', 'jsonl']:
