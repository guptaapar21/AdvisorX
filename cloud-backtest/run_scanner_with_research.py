#!/usr/bin/env python3
"""Run the production scanner with validated research context injected read-only."""
from __future__ import annotations

import json
from pathlib import Path

import gemini_advisor

FORMULATION_FILE = Path("research_formulations.json")
MAX_CONTEXT_CHARS = 12000


def load_formulations() -> list[dict]:
    if not FORMULATION_FILE.exists():
        return []
    try:
        payload = json.loads(FORMULATION_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return []
    if payload.get("research_only") is not True:
        return []
    values = payload.get("validated_formulations")
    return values if isinstance(values, list) else []


def build_context(formulations: list[dict]) -> str:
    if not formulations:
        return """\n\nRESEARCH FEEDBACK:\nNo formulation has currently passed the research validation + chronological holdout + cost-stress gates. Do not infer an edge from research discovery leads.\n"""
    lines = [
        "",
        "RESEARCH FEEDBACK — HOLDOUT-PASSED EVIDENCE ONLY:",
        "The following formulations passed the research engine's development validation, chronological holdout, multi-coin/multi-block robustness checks, and cost-stress gate.",
        "Use them as empirical supporting evidence, never as an unconditional trade rule. They do not override entry quality, portfolio risk, geometry, fees, or invalidation.",
    ]
    for f in formulations:
        lines.append(json.dumps({
            "name": f.get("name"),
            "event_family": f.get("event_family"),
            "direction": f.get("direction"),
            "horizon_min": f.get("horizon_min"),
            "conditions": f.get("conditions", {}),
            "validation_avg_net_return": (f.get("validation") or {}).get("avg_net_return"),
            "validation_profit_factor": (f.get("validation") or {}).get("profit_factor"),
            "holdout_avg_net_return": (f.get("holdout") or {}).get("avg_net_return"),
            "holdout_profit_factor": (f.get("holdout") or {}).get("profit_factor"),
        }, sort_keys=True, separators=(",", ":")))
    return "\n" + "\n".join(lines)[:MAX_CONTEXT_CHARS] + "\n"


formulations = load_formulations()
import trend_alignment_scanner_live  # noqa: E402

# The production launcher finalizes the V4 prompt during import; append the
# validated research context only after that, so the final Gemini prompt sees it.
gemini_advisor.SYSTEM_PROMPT += build_context(formulations)
print(f"Research feedback loaded: {len(formulations)} holdout-passed formulation(s).")

if __name__ == "__main__":
    trend_alignment_scanner_live.main()
