
from __future__ import annotations
import json, os, tempfile
from pathlib import Path
from .research_config import DATA_DIR, OBS_FILE, OUTCOME_FILE, HYPOTHESIS_FILE, ERROR_FILE, ANALYSIS_FILE, STATE_FILE, MEMORY_FILE

ROOT = DATA_DIR

def _resolve_path(name_or_path):
    p = Path(name_or_path)
    if p.is_absolute():
        return p
    root = Path(ROOT)
    try:
        p.relative_to(root)
        return p
    except ValueError:
        return root / p

def _append(path: Path, record: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, separators=(",", ":"), default=str)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())

def append_jsonl(name_or_path, record):
    _append(_resolve_path(name_or_path), record)

def read_jsonl(name_or_path, *, strict=True):
    p = _resolve_path(name_or_path)
    if not p.exists():
        return []

    out = []
    bad = 0
    with p.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                bad += 1
                if strict:
                    raise ValueError(
                        f"Corrupt JSONL: {p} line {line_no}: {exc}"
                    ) from exc

    if bad:
        raise ValueError(f"{bad} malformed JSONL records found in {p}")
    return out

def read_state():
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Corrupt research state: {STATE_FILE}") from exc

def write_state(state):
    ROOT.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".state-", dir=ROOT, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, STATE_FILE)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

def load_memory():
    if not MEMORY_FILE.exists():
        return {"version": 1, "validated": [], "rejected": []}
    try:
        return json.loads(MEMORY_FILE.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Corrupt research memory: {MEMORY_FILE}") from exc

def write_memory(memory):
    ROOT.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".memory-", dir=ROOT, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(memory, f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, MEMORY_FILE)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
