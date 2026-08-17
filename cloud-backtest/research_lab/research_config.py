
from __future__ import annotations
import os
from pathlib import Path

DATA_DIR = Path(os.getenv("RESEARCH_DATA_DIR", "cloud-backtest/research_lab_data"))
CANDLE_DIR = DATA_DIR / "candles_1m"  # transient only; workflow does not persist this folder
OBS_FILE = DATA_DIR / "observations.jsonl"
OUTCOME_FILE = DATA_DIR / "outcomes.jsonl"
HYPOTHESIS_FILE = DATA_DIR / "hypotheses.jsonl"
ERROR_FILE = DATA_DIR / "errors.jsonl"
ANALYSIS_FILE = DATA_DIR / "analysis_log.jsonl"
STATE_FILE = DATA_DIR / "state.json"
MEMORY_FILE = DATA_DIR / "research_memory.json"

FEATURE_SCHEMA_VERSION = "2026-08-17-r4"
HORIZONS_MIN = (5, 10, 15, 30, 60)
MAX_HORIZON_MIN = max(HORIZONS_MIN)
PURGE_MIN = MAX_HORIZON_MIN

# 120m fetch gives enough lookback for 60m features and recovery from delayed jobs.
OBSERVATION_LOOKBACK_MIN = int(os.getenv("RESEARCH_OBSERVATION_LOOKBACK_MIN", "120"))
CANDLE_RETENTION_DAYS = int(os.getenv("RESEARCH_CANDLE_RETENTION_DAYS", "2"))

# Research every hour after enough NEW matured outcomes exist.
ANALYSIS_UTC_HOUR = int(os.getenv("RESEARCH_ANALYSIS_UTC_HOUR", "0"))
ANALYSIS_INTERVAL_MIN = int(os.getenv("RESEARCH_ANALYSIS_INTERVAL_MIN", "60"))
MIN_NEW_OUTCOMES_FOR_ANALYSIS = int(
    os.getenv("RESEARCH_MIN_NEW_OUTCOMES_FOR_ANALYSIS", "1000")
)

TAKER_FEE_RATE = float(os.getenv("RESEARCH_TAKER_FEE", "0.00075"))
GEMINI_MODEL = os.getenv("RESEARCH_GEMINI_MODEL", "gemini-3.5-flash-lite")
GEMINI_MAX_PASSES = int(os.getenv("RESEARCH_GEMINI_MAX_PASSES", "2"))

COINS = tuple(
    x.strip().upper()
    for x in os.getenv(
        "RESEARCH_COINS",
        "BTC,ETH,BNB,SOL,XRP,DOGE,LTC,LINK,TRX,AVAX,HYPE,ZEC,ADA,ACE,PAXG",
    ).split(",")
    if x.strip()
)

# Research candidate search controls.
MIN_MINER_RULE_N = int(os.getenv("RESEARCH_MINER_RULE_N", "75"))
MIN_MINER_COIN_COUNT = int(os.getenv("RESEARCH_MINER_COIN_COUNT", "4"))
MIN_MINER_AVG_RETURN = float(os.getenv("RESEARCH_MINER_MIN_AVG", "0.00005"))
MIN_MINER_PF = float(os.getenv("RESEARCH_MINER_MIN_PF", "1.02"))
MIN_MINER_UNIVARIATE_KEEP = int(os.getenv("RESEARCH_MINER_UNIVARIATE_KEEP", "20"))
MIN_MINER_PAIR_KEEP = int(os.getenv("RESEARCH_MINER_PAIR_KEEP", "40"))
MIN_MINER_TRIPLE_KEEP = int(os.getenv("RESEARCH_MINER_TRIPLE_KEEP", "15"))

DISCOVERY_PASSES = int(os.getenv("RESEARCH_DISCOVERY_PASSES", "2"))
DISCOVERY_PASS_CANDIDATES = int(os.getenv("RESEARCH_DISCOVERY_PASS_CANDIDATES", "120"))
DISCOVERY_INPUT_CHAR_BUDGET = int(os.getenv("RESEARCH_DISCOVERY_INPUT_CHAR_BUDGET", "100000"))
MAX_HYPOTHESES_PER_PASS = int(os.getenv("RESEARCH_MAX_HYPOTHESES_PER_PASS", "10"))
MAX_TOTAL_HYPOTHESES = int(os.getenv("RESEARCH_MAX_TOTAL_HYPOTHESES", "20"))
REQUEST_TIMEOUT_SECONDS = int(os.getenv("RESEARCH_GEMINI_TIMEOUT_SECONDS", "45"))

MIN_DISCOVERY_OBSERVATIONS = int(os.getenv("RESEARCH_MIN_DISCOVERY_OBSERVATIONS", "300"))
MIN_VALIDATION_OBSERVATIONS = int(os.getenv("RESEARCH_MIN_VALIDATION_OBSERVATIONS", "100"))
MIN_HOLDOUT_OBSERVATIONS = int(os.getenv("RESEARCH_MIN_HOLDOUT_OBSERVATIONS", "100"))

# Finalist/robustness controls.
HOLDOUT_FINALISTS = int(os.getenv("RESEARCH_HOLDOUT_FINALISTS", "10"))
RESEARCH_BLOCK_MINUTES = int(os.getenv("RESEARCH_BLOCK_MINUTES", "30"))
MIN_VALIDATION_TIME_BLOCKS = int(os.getenv("RESEARCH_MIN_VALIDATION_TIME_BLOCKS", "4"))
MIN_HOLDOUT_TIME_BLOCKS = int(os.getenv("RESEARCH_MIN_HOLDOUT_TIME_BLOCKS", "4"))
MIN_POSITIVE_BLOCK_FRACTION = float(os.getenv("RESEARCH_MIN_POSITIVE_BLOCK_FRACTION", "0.60"))
MIN_POSITIVE_COIN_FRACTION = float(os.getenv("RESEARCH_MIN_POSITIVE_COIN_FRACTION", "0.60"))
MIN_XSEC_COVERAGE = float(os.getenv("RESEARCH_MIN_XSEC_COVERAGE", "0.70"))

# Explicit stress is diagnostic; base outcome already contains configured taker costs.
STRESS_COST_MULTIPLIERS = tuple(
    float(x.strip())
    for x in os.getenv("RESEARCH_STRESS_COST_MULTIPLIERS", "1.0,1.5,2.0").split(",")
    if x.strip()
)

# Research-only promotion gates. Never connected to AdvisorX decisions.
PROMOTE_MIN_VALID_N = int(os.getenv("RESEARCH_PROMOTE_MIN_VALID_N", "50"))
PROMOTE_MIN_HOLDOUT_N = int(os.getenv("RESEARCH_PROMOTE_MIN_HOLDOUT_N", "50"))
PROMOTE_MIN_VALID_PF = float(os.getenv("RESEARCH_PROMOTE_MIN_VALID_PF", "1.15"))
PROMOTE_MIN_HOLDOUT_PF = float(os.getenv("RESEARCH_PROMOTE_MIN_HOLDOUT_PF", "1.10"))
PROMOTE_MIN_VALID_AVG = float(os.getenv("RESEARCH_PROMOTE_MIN_VALID_AVG", "0.0005"))
PROMOTE_MIN_HOLDOUT_AVG = float(os.getenv("RESEARCH_PROMOTE_MIN_HOLDOUT_AVG", "0.0002"))
PROMOTE_MIN_HOLDOUT_WINRATE = float(os.getenv("RESEARCH_PROMOTE_MIN_HOLDOUT_WINRATE", "0.52"))
PROMOTE_MAX_HOLDOUT_DD = float(os.getenv("RESEARCH_PROMOTE_MAX_HOLDOUT_DD", "0.08"))

# The result is a fixed-horizon predictive rule, not an executable trade strategy.
RESEARCH_RULE_TYPE = "fixed_horizon_predictive_rule"

HOLDOUT_TEST_BUDGET_PER_EPOCH = int(os.getenv("RESEARCH_HOLDOUT_TEST_BUDGET_PER_EPOCH", "10"))

PERSIST_EVERY_MINUTES = int(os.getenv("RESEARCH_PERSIST_EVERY_MINUTES", "5"))
