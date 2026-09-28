"""Load the vendored production ML modules used by the demo.

The grounding and structure metrics run the same TripleParser,
MetricsCalculator, NERConsistencyChecker and KGClient that MonitorLLM uses in
production. Those modules live under ``monitorlab.vendor/`` so this folder runs
without the rest of the repository (no Reflex, torch, or SQLAlchemy).
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

# Kept for selfcheck / UI messaging: everything now lives inside demo/.
REPO_ROOT = Path(__file__).resolve().parents[1]

_cache: dict[str, Any] = {}


def _once(key: str, factory) -> Any:
    if key not in _cache:
        _cache[key] = factory()
    return _cache[key]


def load(dotted: str) -> Any:
    """Import a module by dotted name (vendor package or absolute)."""
    return import_module(dotted)


def triple_parser():
    """The production RDF triple parser (``TripleParser``)."""
    return _once(
        "triple_parser",
        lambda: load("monitorlab.vendor.triple_parser").TripleParser,
    )


def metrics_calculator():
    """The production ICR/IPR/CI calculator (``MetricsCalculator``)."""
    return _once(
        "metrics_calculator",
        lambda: load("monitorlab.vendor.metrics_calculator").MetricsCalculator,
    )


def settings():
    """Demo settings module: prompts and the shared spaCy model."""
    return _once("settings", lambda: load("monitorlab.vendor.settings"))


def kg_client_class():
    """The production ``KGClient`` class."""
    return _once("kg_client", lambda: load("monitorlab.vendor.kg_client").KGClient)


def ner_checker():
    """A shared ``NERConsistencyChecker`` instance.

    Construction loads the spaCy pipeline and attaches an EntityRuler, so it is
    built once and reused. The instance is *not* thread-safe for concurrent
    calls into spaCy, so callers hold ``NER_LOCK`` while scoring.
    """

    def build():
        module = load("monitorlab.vendor.ner_consistency")
        cfg = settings()
        if getattr(cfg, "NLP_MODEL", None) is None:
            raise RuntimeError(
                "spaCy's en_core_web_sm model is not available, so the grounding "
                "metric cannot run. Install it with:\n"
                "    .venv/bin/python -m spacy download en_core_web_sm"
            )
        return module.NERConsistencyChecker()

    return _once("ner_checker", build)


def provenance() -> list[dict[str, str]]:
    """Where each scoring component lives, for display in the UI."""
    return [
        {"component": "TripleParser", "module": "monitorlab/vendor/triple_parser.py"},
        {
            "component": "MetricsCalculator",
            "module": "monitorlab/vendor/metrics_calculator.py",
        },
        {
            "component": "NERConsistencyChecker",
            "module": "monitorlab/vendor/ner_consistency.py",
        },
        {"component": "KGClient", "module": "monitorlab/vendor/kg_client.py"},
        {
            "component": "SYSTEM_PROMPT / SENSITIVITY_*",
            "module": "monitorlab/vendor/settings.py",
        },
    ]
