#!/usr/bin/env python3
"""Verify the demo's vendored ML modules work offline.

Checks, in order:
  1. the vendored modules import at all;
  2. they import *without* dragging in torch, SQLAlchemy or Reflex;
  3. they actually compute, on a fixed triple set, with no network and no Ollama.

Run by setup.sh. Exits non-zero with an actionable message on failure.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from monitorlab import compat  # noqa: E402

GREEN, RED, YELLOW, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[0m"

# A hand-written triple set in the exact format SYSTEM_PROMPT asks for, so the
# check does not depend on an LLM being available.
FIXTURE_TEXT = "Elon Musk announced that Tesla would build a new factory in Berlin."
FIXTURE_TRIPLES = """<http://dbpedia.org/resource/Elon_Musk, rdf:type, http://dbpedia.org/ontology/Person>
<http://dbpedia.org/resource/Tesla_Inc, rdf:type, http://dbpedia.org/ontology/Company>
<http://dbpedia.org/resource/Berlin, rdf:type, http://dbpedia.org/ontology/City>
<http://dbpedia.org/resource/Elon_Musk, foaf:name, "Elon Musk"@en>
<http://dbpedia.org/resource/Tesla_Inc, dbo:location, http://dbpedia.org/resource/Berlin>"""


def ok(msg: str) -> None:
    print(f"    {GREEN}ok{RESET}   {msg}")


def fail(msg: str, hint: str = "") -> None:
    print(f"    {RED}fail{RESET} {msg}")
    if hint:
        print(f"         {hint}")
    sys.exit(1)


def main() -> None:
    # 1. Imports.
    try:
        parser = compat.triple_parser()
        calc = compat.metrics_calculator()
        cfg = compat.settings()
    except Exception as exc:  # noqa: BLE001
        fail(f"could not import vendored modules: {exc}")
    ok("vendored TripleParser, MetricsCalculator and settings import")

    # 2. Nothing heavy came along for the ride.
    for unwanted, why in [
        ("torch", "would indicate a heavy optional dependency leaked in"),
        ("sqlalchemy", "would indicate a heavy optional dependency leaked in"),
        ("reflex", "would indicate a heavy optional dependency leaked in"),
    ]:
        if unwanted in sys.modules:
            fail(f"{unwanted!r} was imported ({why})")
    ok("no torch, sqlalchemy or reflex in sys.modules")

    # 3. The vendored code computes.
    triples = parser.parse_triples_from_text(FIXTURE_TRIPLES)
    if len(triples) != 5:
        fail(f"expected 5 parsed triples, got {len(triples)}")
    icr, _, _ = calc.calculate_icr_metric(triples)
    ipr, _, _ = calc.calculate_ipr_metric(triples)
    ci = calc.calculate_ci(FIXTURE_TRIPLES)
    ok(f"structural metrics compute: icr={icr:.3f} ipr={ipr:.3f} ci={ci:.3f}")

    # 4. spaCy and the grounding metric. Network is not required: KGClient fails
    #    open to an empty type list, which only lowers the score.
    if getattr(cfg, "NLP_MODEL", None) is None:
        fail(
            "spaCy's en_core_web_sm is not installed",
            "fix: .venv/bin/python -m spacy download en_core_web_sm",
        )
    ok("spaCy en_core_web_sm is loaded")

    try:
        checker = compat.ner_checker()
        analysis = checker.consistency_scorer.get_detailed_consistency_analysis(
            FIXTURE_TEXT, triples
        )
    except Exception as exc:  # noqa: BLE001
        fail(f"NERConsistencyChecker failed: {exc}")
    ok(
        "NER consistency computes: score="
        f"{analysis['consistency_score']:.3f}, "
        f"{len(analysis['rule_based_entities'])} triple entities vs "
        f"{len(analysis['spacy_entities'])} spaCy entities"
    )

    cache = Path(__file__).resolve().parents[1] / ".kg_cache"
    n = len(list(cache.glob("*.json"))) if cache.is_dir() else 0
    if n:
        ok(f"DBpedia cache has {n} entries")
    else:
        print(f"    {YELLOW}warn{RESET} DBpedia cache is empty; first run will be slower")

    print(f"\n    {GREEN}All checks passed.{RESET}")


if __name__ == "__main__":
    main()
