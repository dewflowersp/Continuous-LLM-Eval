"""Structural robustness: does the model honour the schema it was asked for?

    robustness = (ICR + IPR + CI) / 3

That is the parent project's own definition, from
``reflex_app/utils/chart_utils.py:87-92``. The three components come straight
from the production ``MetricsCalculator``:

ICR (instantiated class ratio)
    Of the DBpedia classes mentioned, how many were actually typed with
    ``rdf:type``? Penalises naming an entity without saying what it is.
IPR (instantiated property ratio)
    What share of predicates come from standard vocabularies (rdf, rdfs, owl,
    foaf, dc) rather than being invented on the spot?
CI (class instantiation)
    Distinct ``rdf:type`` objects divided by the number of triple lines.

Reading these together is the point: a model can score well on one by accident.
The composite moves when output quality genuinely degrades -- which is exactly
what makes it worth putting on a control chart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .. import compat


@dataclass
class StructureResult:
    icr: float = 0.0
    ipr: float = 0.0
    ci: float = 0.0
    robustness: float = 0.0
    triples: list[tuple[str, str, str]] = field(default_factory=list)
    triple_count: int = 0
    parse_failed: bool = False
    note: str = ""

    def as_row(self) -> dict[str, Any]:
        return {
            "icr": round(self.icr, 5),
            "ipr": round(self.ipr, 5),
            "ci": round(self.ci, 5),
            "robustness": round(self.robustness, 5),
            "triple_count": self.triple_count,
        }


def score(raw_text: str) -> StructureResult:
    """Score one raw LLM response.

    Takes the unparsed text because ``calculate_ci`` works on the raw string
    (it divides by line count), while ICR and IPR work on parsed tuples. The
    call sequence here mirrors ``analysis_service.analyze_single_model``
    (``analysis_service.py:283-290``) so the numbers match the pipeline's.
    """
    parser = compat.triple_parser()
    calculator = compat.metrics_calculator()

    triples = parser.parse_triples_from_text(raw_text or "")

    if not triples:
        # An empty parse is a real monitoring event, not an error to swallow:
        # the model was asked for triples and produced nothing usable. Scoring
        # it zero lets the signal register the failure instead of skipping a
        # step and hiding it.
        return StructureResult(
            parse_failed=True,
            note="no parsable triples in the response",
        )

    icr, _, _ = calculator.calculate_icr_metric(triples)
    ipr, _, _ = calculator.calculate_ipr_metric(triples)
    ci = calculator.calculate_ci(raw_text)

    return StructureResult(
        icr=icr,
        ipr=ipr,
        ci=ci,
        robustness=round((icr + ipr + ci) / 3, 5),
        triples=triples,
        triple_count=len(triples),
    )
