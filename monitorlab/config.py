"""Demo-wide configuration.

Ollama URL and timeouts live here (and via ``DEMO_OLLAMA_*`` env vars). The
vendored settings module does not load a ``.env``, so a stale parent-project
``.env`` cannot redirect the demo mid-tutorial.
"""

from __future__ import annotations

import os
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = DEMO_DIR / "data"
RUNS_DIR = DATA_DIR / "runs"
SAMPLES_FILE = DATA_DIR / "samples.json"

OLLAMA_URL = os.environ.get("DEMO_OLLAMA_URL", "http://localhost:11434")
OLLAMA_TIMEOUT = int(os.environ.get("DEMO_OLLAMA_TIMEOUT", "180"))

# Models that answer fast enough to loop live. Used only to sort the picker so
# the safe choices float to the top; every installed model stays selectable.
FAST_MODELS = ("llama3.2:3b", "phi3:3.8b", "qwen2.5-coder:7b", "llama3:8b", "llama3.1:8b")

# Models large enough that a live loop would stall the room.
SLOW_MODELS = ("llama3.3:70b", "gpt-oss:120b", "codestral:22b", "gpt-oss:20b")

# Name substrings for models that emit a separate `thinking` trace which
# shares the num_predict budget with the visible answer (gpt-oss, gemma4, …).
REASONING_MODEL_TAGS = ("gpt-oss", "gemma4")

# Below this floor, default thinking often exhausts the budget and leaves
# response empty (done_reason=length). Measured ~4096 on gpt-oss:20b / gemma4:e4b
# with SYSTEM_PROMPT. DEMO_GPT_OSS_NUM_PREDICT is kept as a legacy alias.
REASONING_MIN_NUM_PREDICT = int(
    os.environ.get(
        "DEMO_REASONING_NUM_PREDICT",
        os.environ.get("DEMO_GPT_OSS_NUM_PREDICT", "4096"),
    )
)
# Back-compat for imports / docs that still say GPT_OSS_*.
GPT_OSS_MIN_NUM_PREDICT = REASONING_MIN_NUM_PREDICT

# ----------------------------------------------------------------- the metrics
# key -> (label, description, direction)
# direction is "higher_is_better" or "lower_is_better" and drives whether an
# alarm fires on an upward or downward excursion.
METRICS: dict[str, dict[str, str]] = {
    "hallucination": {
        "label": "Grounding (hallucination)",
        "short": "Grounding",
        "description": (
            "mean of rule_definition_score and ner_consistency_score, the "
            "project's own hallucination metric (chart_utils.py:79-82). Falls "
            "when the model asserts entities that DBpedia and spaCy will not "
            "corroborate."
        ),
        "direction": "higher_is_better",
        "color": "#22d3ee",
    },
    "robustness": {
        "label": "Structural robustness",
        "short": "Robustness",
        "description": (
            "mean of ICR, IPR and CI (chart_utils.py:87-92). Measures how well "
            "the emitted triples conform to the DBpedia schema it was asked for."
        ),
        "direction": "higher_is_better",
        "color": "#a78bfa",
    },
    "drift": {
        "label": "Semantic drift (self-inconsistency)",
        "short": "Drift",
        "description": (
            "resample the same prompt k times and measure how much the answers "
            "disagree: mean pairwise Jaccard distance over triple sets, blended "
            "with TF-IDF cosine dispersion. 0 means every resample agreed."
        ),
        "direction": "lower_is_better",
        "color": "#fb923c",
    },
    "bias": {
        "label": "Biased framing (sentiment spread)",
        "short": "Bias",
        "description": (
            "how far apart the model's sentiment lands across demographic "
            "groups asked the same question about the same text, averaged over "
            "prompts. Rises when the model tells materially different stories "
            "to different groups. The parent project's silhouette-based "
            "bias_score is computed alongside it as a diagnostic, but is not "
            "the monitored signal: a silhouette measures how cleanly the groups "
            "cluster rather than how far apart they are, and on contested vs "
            "neutral presets it can score both nearly the same."
        ),
        "direction": "lower_is_better",
        "color": "#f472b6",
    },
    "latency": {
        "label": "Latency (s)",
        "short": "Latency",
        "description": "wall-clock seconds for the triple-extraction call.",
        "direction": "lower_is_better",
        "color": "#94a3b8",
    },
}

# Metrics on by default. Bias is off because it costs six extra LLM calls.
DEFAULT_METRICS = ("hallucination", "robustness", "drift", "latency")


def classify_model(name: str) -> str:
    """Rough speed bucket for a model name, used for UI hints."""
    if name in FAST_MODELS:
        return "fast"
    if name in SLOW_MODELS:
        return "slow"
    return "unknown"
