#!/usr/bin/env python3
"""Persist the research engine's discovery/validation diagnostics for inspection.

This file is deliberately separate from research_formulations.json. The latter is
live-facing and contains only holdout-passed formulations; this report keeps the
ranked discovery leads and recent rejection reasons visible for debugging without
feeding unvalidated research into Gemini.
"""
from __future__ import annotations

import json
from pathlib import Path

import research_analyst

ROOT = Path("market_behavior_research")
OUTPUT = Path("research_diagnostics.json")
FEE_PCT = 0.15
TOP_N = 30
MAX_REJECTED = 100


def main() -> int:
    rows = research_analyst.load_rows(ROOT)
    result = research_analyst.analyse(rows, FEE_PCT)
    payload = {
        "schema_version": "research_diagnostics_v1",
        "generated_at": research_analyst.datetime.now(research_analyst.timezone.utc).isoformat(),
        "research_only": True,
        "status": result.get("status"),
        "rows": result.get("rows", 0),
        "validation_rows": result.get("validation_rows", 0),
        "holdout_rows": result.get("holdout_rows", 0),
        "fee_pct": result.get("fee_pct", FEE_PCT),
        "candidate_count": len(research_analyst.generate_candidates(rows)) if rows else 0,
        "discovery_leads": result.get("discovery_leads", [])[:TOP_N],
        "validated": result.get("validated", []),
        "rejected": result.get("rejected", [])[:MAX_REJECTED],
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "rows": payload["rows"],
        "candidate_count": payload["candidate_count"],
        "discovery_leads": len(payload["discovery_leads"]),
        "validated_formulations": len(payload["validated"]),
        "rejected": len(payload["rejected"]),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
