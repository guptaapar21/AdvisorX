from __future__ import annotations
import json, os, tempfile
from pathlib import Path
from .research_config import DATA_DIR, OBS_FILE, OUTCOME_FILE, HYPOTHESIS_FILE, ERROR_FILE, ANALYSIS_FILE, STATE_FILE, MEMORY_FILE

ROOT=DATA_DIR

def _append(path: Path, record: dict):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf-8') as f:f.write(json.dumps(record,separators=(',',':'),default=str)+'\n')

def append_jsonl(name_or_path, record):
    p=Path(name_or_path)
    if not p.is_absolute(): p=ROOT / p
    _append(p,record)

def read_jsonl(name_or_path):
    p=Path(name_or_path)
    if not p.is_absolute(): p=ROOT / p
    if not p.exists(): return []
    out=[]
    with p.open(encoding='utf-8') as f:
        for line in f:
            try:
                if line.strip(): out.append(json.loads(line))
            except json.JSONDecodeError: continue
    return out

def read_state():
    return json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}

def write_state(state):
    ROOT.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.state-',dir=ROOT,text=True)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f: json.dump(state,f,indent=2,default=str); f.flush(); os.fsync(f.fileno())
        os.replace(tmp,STATE_FILE)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def load_memory():
    if not MEMORY_FILE.exists(): return {"version":1,"validated":[],"rejected":[]}
    try: return json.loads(MEMORY_FILE.read_text())
    except Exception: return {"version":1,"validated":[],"rejected":[]}

def write_memory(memory):
    ROOT.mkdir(parents=True,exist_ok=True)
    tmp=MEMORY_FILE.with_suffix('.tmp'); tmp.write_text(json.dumps(memory,indent=2,default=str)); os.replace(tmp,MEMORY_FILE)
