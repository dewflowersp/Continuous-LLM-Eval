"""monitorlab -- continuous evaluation and monitoring of a local LLM.

Supporting code for the tutorial "Continuous Evaluation and Monitoring of
Foundation Models". Streams text through an Ollama model repeatedly, scores each
response with the vendored production metrics, and runs online drift and
anomaly detection over the resulting time series.

Layout:
    compat      loads TripleParser / MetricsCalculator / NER / prompts from vendor/
    vendor/     self-contained production ML slice (no Reflex, torch, Postgres)
    ollama_client  minimal Ollama HTTP client
    metrics/    the four monitored signals
    detectors   online adaptive-threshold and change-point detection
    stream      the time-indexed buffer and the background stream runner
    scenarios   sample texts and mid-stream perturbations
    replay      record runs to JSONL and play them back offline
"""

__all__ = [
    "compat",
    "config",
    "detectors",
    "metrics",
    "ollama_client",
    "replay",
    "scenarios",
    "stream",
]
