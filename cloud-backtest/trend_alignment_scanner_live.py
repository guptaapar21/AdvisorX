"""Production launcher that adds the deterministic entry-quality gate."""
import gemini_advisor
from entry_quality_gate import apply_entry_quality_gate

_original = gemini_advisor.get_trade_suggestions_batch

def _wrapped(signals, scorecard=None, open_positions=None):
    ok, flagged, updates = _original(signals, scorecard, open_positions)
    if not ok:
        return ok, flagged, updates
    rejected = apply_entry_quality_gate(flagged, signals)
    if rejected:
        print(f'  Entry-quality gate: rejected {rejected} Gemini TAKE proposal(s)')
    return ok, flagged, updates

gemini_advisor.get_trade_suggestions_batch = _wrapped
from trend_alignment_scanner import main  # noqa: E402

if __name__ == '__main__':
    main()
