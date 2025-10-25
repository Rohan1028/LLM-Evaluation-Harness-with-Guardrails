from __future__ import annotations

import hashlib
import json
import math
import random
import string
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, TypeVar

import yaml

T = TypeVar("T")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def save_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def save_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False))
            fh.write("\n")


def save_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    import pandas as pd

    ensure_dir(path.parent)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def timestamp_token() -> str:
    import datetime as _dt

    return _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def git_sha() -> Optional[str]:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return None


def hash_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def batched(iterable: Sequence[T], batch_size: int) -> Iterator[Sequence[T]]:
    for idx in range(0, len(iterable), batch_size):
        yield iterable[idx : idx + batch_size]


def optional_import(module_name: str) -> Any:
    try:
        module = __import__(module_name)
        return module
    except ImportError:
        return None


def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b, strict=False))
    norm_a = math.sqrt(sum(a * a for a in vec_a)) or 1.0
    norm_b = math.sqrt(sum(b * b for b in vec_b)) or 1.0
    return dot / (norm_a * norm_b)


def seeded_shuffle(items: List[T], seed: int) -> List[T]:
    rng = random.Random(seed)
    copy = list(items)
    rng.shuffle(copy)
    return copy


def slugify(value: str, sep: str = "-") -> str:
    allowed = string.ascii_letters + string.digits
    cleaned = "".join(ch if ch in allowed else sep for ch in value)
    while sep * 2 in cleaned:
        cleaned = cleaned.replace(sep * 2, sep)
    return cleaned.strip(sep).lower()
