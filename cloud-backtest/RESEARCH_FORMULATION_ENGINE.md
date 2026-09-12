# AdvisorX Research Formulation Engine

AdvisorX now has a controlled research-to-live feedback loop.

## Pipeline

`market_behavior_recorder.py` captures closed-1m market events and waits for complete forward outcomes. `research_analyst.py` joins each outcome to its original observation, generates a bounded set of falsifiable formulations, and evaluates them chronologically.

The analyst uses a 70/30 chronological development/holdout split. Candidate formulations are based on event family + direction + forward horizon, with a small bounded set of contextual conditions such as high activity, extreme activity, strong delta-proxy magnitude, strong wick and strong event score.

A formulation can reach `HOLDOUT_PASSED` only when it has enough observations in development and holdout, positive average net return after round-trip fees, acceptable profit factor, positive block/coin breadth, a positive block-level 95% lower confidence bound, and positive performance under a higher cost-stress assumption.

## Live feedback boundary

`research_formulations.json` contains only formulations that passed every research gate. `run_scanner_with_research.py` injects those formulations into the Gemini prompt as empirical evidence.

The feedback is advisory evidence only. It cannot:

- create a TAKE on its own;
- lower or bypass the V4 entry-quality gate;
- bypass portfolio risk checks;
- override geometry or fee safeguards;
- change V4 thresholds;
- mutate trade-management policy.

The normal production sequence remains: research context -> Gemini directional/setup judgment -> deterministic V4 entry/risk gates -> deterministic position management/resolution.

## No-edge state is valid

When no formulation passes the holdout and cost-stress gates, the analyst deliberately publishes an empty validated set. Discovery leads remain research evidence and are not promoted merely to increase trade frequency.

This is intentional: the system should prefer `no validated edge yet` over converting an in-sample pattern into a live rule.
