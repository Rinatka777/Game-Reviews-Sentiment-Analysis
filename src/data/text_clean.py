import re
from typing import Literal

_URL_RE  = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_USER_RE = re.compile(r"@[A-Za-z0-9_]+")
_WS_RE   = re.compile(r"\s+")

def normalize_spaces(s:str) -> str:
    return _WS_RE.sub(" ", s).strip()

def replace_urls_and_users(s:str) -> str:
    s1 = _URL_RE.sub("<url", s)
    s2 = _USER_RE.sub("<user", s1)
    return s2

def clean_text(s:str, mode:Literal["tfidf", "transformer"] = "tfidf") -> str:
    if not isinstance(s, str):
        s = ""
    if mode == "tfidf":
        s = s.lower()
        s = replace_urls_and_users(s)
        s = normalize_spaces(s)
        return s
    elif mode == "transformer":
        s = normalize_spaces(s)
        return s
    else:
        raise ValueError(f"Unknown mode: {mode}")

    #continue with other small functions

