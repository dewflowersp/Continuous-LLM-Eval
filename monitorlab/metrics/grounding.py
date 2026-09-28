"""Grounding: will an external authority corroborate what the model asserted?

    hallucination = (rule_definition_score + ner_consistency_score) / 2

The parent project's own formula, from
``reflex_app/utils/chart_utils.py:79-82``. Despite the name it reads as a
*health* score: high is good, and it falls when grounding fails.

Two independent checks, which is why averaging them is more informative than
either alone:

``ner_consistency_score``
    F1 between the entities implied by the model's triples and the entities
    spaCy finds in the source text. Catches entities the model invented, and
    entities it ignored. Local and fast.

``rule_definition_score``
    ``0.7 * DBpedia coverage + 0.3 * EntityRuler coverage``. Asks a real
    knowledge graph whether each entity exists with a compatible type. This is
    the check that catches a confidently-stated entity that simply is not real.

Network behaviour: the DBpedia leg goes over SPARQL and is cached on disk.
``KGClient`` fails open -- a lookup that times out returns no types, which lowers
the score rather than raising. That is the right default for a monitor but it
means an offline run reads as slightly worse grounding, so the UI reports cache
hits and lets the room see the difference.

A measured caveat, and the most useful thing in this module to put on a slide.
The F1 is computed over the entities the checker can pull *out of the triples*,
and that extraction is lossy: on a 33-triple response about Scottish football it
recovered between one and three entities. With a denominator that small the score
is quantised, and on five consecutive steps with byte-identical input this metric
produced 0.450, 0.200, 0.266, 0.000, 0.450 -- a noise width of 61% of its own
level, driven by whether the string "Edinburgh" happened to survive extraction.
On entity-dense prose where extraction recovers more (the ``grounded`` preset) the
same metric sits at 0.459 with a noise width of 9.5%.

Nothing here corrects that; correcting it would mean no longer demonstrating the
production metric. Instead ``support`` reports how many entities the score was
computed from, and the result carries a note when that number is too small for
the value to be worth reading. Publishing an estimate without its support is how
a dashboard ends up being confidently wrong, and this is a live example of it.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from .. import compat

# spaCy pipelines are not safe to call concurrently, and RuleDefinitionScorer
# mutates the shared pipeline by adding an EntityRuler. One lock guards every
# entry into the borrowed NER code.
NER_LOCK = threading.Lock()

# Below this many entities the F1 can only take a handful of values, so a change
# in it says more about extraction luck than about grounding.
MIN_SUPPORT = 3


@dataclass
class GroundingResult:
    ner_consistency: float = 0.0
    rule_definition: float = 0.0
    hallucination: float = 0.0
    support: int = 0  # entities the F1 was actually computed over
    triple_entities: list[dict[str, str]] = field(default_factory=list)
    spacy_entities: list[dict[str, str]] = field(default_factory=list)
    inconsistencies: list[dict[str, str]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    note: str = ""

    @property
    def thin_support(self) -> bool:
        return self.support < MIN_SUPPORT

    def as_row(self) -> dict[str, Any]:
        return {
            "ner_consistency_score": round(self.ner_consistency, 5),
            "rule_definition_score": round(self.rule_definition, 5),
            "hallucination": round(self.hallucination, 5),
            "entities_compared": self.support,
        }


def score(source_text: str, triples: list[tuple[str, str, str]]) -> GroundingResult:
    """Score the grounding of one parsed triple set against its source text.

    Calls ``get_detailed_consistency_analysis`` and
    ``calculate_rule_definitions_score`` separately rather than using
    ``generate_ner_consistency_report``, which would compute the rule score
    twice and so make two full passes of DBpedia lookups per step. The
    arithmetic is identical -- ``analysis_service`` reads exactly these two
    values (``analysis_service.py:209-235``) -- but a live stream cannot afford
    the duplicate network round trips.
    """
    if not triples:
        return GroundingResult(note="no triples to ground")

    checker = compat.ner_checker()

    with NER_LOCK:
        analysis = checker.consistency_scorer.get_detailed_consistency_analysis(
            source_text, triples
        )
        rule_definition = checker.calculate_rule_definitions_score(source_text, triples)

    # The scorers return -1 to mean "could not compute" (too few entities to
    # compare, for instance). Treating that as 0.0 would look like a grounding
    # failure, so it is surfaced as a note instead and clamped out of the way.
    ner_consistency = float(analysis.get("consistency_score", 0.0) or 0.0)
    notes = []
    if ner_consistency < 0:
        notes.append("NER consistency was not computable")
        ner_consistency = 0.0
    if rule_definition is None or rule_definition < 0:
        notes.append("rule definition score was not computable")
        rule_definition = 0.0

    stats = dict(analysis.get("statistics") or {})
    support = int(stats.get("total_rule_based") or 0)
    if support < MIN_SUPPORT:
        notes.append(
            f"scored from {support} extracted entit{'y' if support == 1 else 'ies'} "
            f"out of {len(triples)} triples, so this value is quantised -- read the "
            f"trend, not the step"
        )

    return GroundingResult(
        ner_consistency=ner_consistency,
        rule_definition=float(rule_definition),
        hallucination=round((float(rule_definition) + ner_consistency) / 2, 5),
        support=support,
        triple_entities=list(analysis.get("rule_based_entities") or []),
        spacy_entities=list(analysis.get("spacy_entities") or []),
        inconsistencies=list(analysis.get("inconsistencies") or []),
        stats=stats,
        note="; ".join(notes),
    )


def ungrounded_entities(result: GroundingResult) -> list[dict[str, str]]:
    """Entities the model asserted that spaCy could not find in the source.

    The drill-down the audience actually wants during the hallucination beat:
    not "the score dropped" but "it dropped because it claimed these."
    """
    return [
        inconsistency
        for inconsistency in result.inconsistencies
        if inconsistency.get("type") == "missing_in_spacy"
    ]
