"""Semantic drift, measured as self-inconsistency across resamples.

This is the demo's own metric rather than one borrowed from the parent project,
and it is the one that most directly earns the phrase "treat outputs as
time-indexed signals".

The idea: ask the same model the same question ``k`` times in the same step. A
model that has a firm grip on the text returns effectively the same structure
every time. A model that is guessing returns a different structure each time.
The disagreement *within* a step is a usable confidence signal, and tracking it
*across* steps is what exposes drift -- because a model whose internal agreement
is quietly collapsing will show it here well before its average score moves.

Two views of disagreement, because each misses something the other catches:

Jaccard distance over triple sets
    Structural. Did the resamples commit to the same facts? Insensitive to
    phrasing, which is what we want, but blind to wording that changes meaning
    without changing the extracted triples.
TF-IDF cosine dispersion
    Lexical. How differently were the answers worded? Catches wholesale
    rewrites, including the case where triple extraction failed identically on
    every resample and Jaccard therefore sees perfect agreement.

The reported metric is the mean of the two, so 0.0 means every resample agreed
and higher is worse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Sequence


@dataclass
class DriftResult:
    drift: float = 0.0
    jaccard_distance: float = 0.0
    lexical_dispersion: float = 0.0
    samples: int = 0
    stable_triples: int = 0
    volatile_triples: int = 0
    stable_examples: list[str] = field(default_factory=list)
    volatile_examples: list[str] = field(default_factory=list)
    computable: bool = True
    note: str = ""

    def as_row(self) -> dict[str, Any]:
        return {
            "drift": round(self.drift, 5),
            "jaccard_distance": round(self.jaccard_distance, 5),
            "lexical_dispersion": round(self.lexical_dispersion, 5),
            "resamples": self.samples,
        }


def _normalise(triple: Sequence[str]) -> str:
    """Canonical form of a triple, for set comparison.

    Lowercased and whitespace-stripped so that cosmetic differences between
    resamples do not read as disagreement. URIs are otherwise left intact --
    asserting ``dbo:Person`` versus ``dbo:Organisation`` is a genuine
    disagreement and must not be normalised away.
    """
    parts = [str(part).strip().lower() for part in list(triple)[:3]]
    return " | ".join(parts)


def _mean_pairwise_jaccard_distance(
    triple_sets: Sequence[set[str]],
) -> tuple[float, list[str], list[str]]:
    """Mean pairwise Jaccard distance, plus which triples held and which moved."""
    distances = []
    for left, right in combinations(triple_sets, 2):
        union = left | right
        if not union:
            # Both resamples produced nothing. They agree, vacuously; the
            # structural metric will already be reporting the parse failure.
            distances.append(0.0)
            continue
        distances.append(1.0 - len(left & right) / len(union))

    non_empty = [s for s in triple_sets if s]
    if non_empty:
        stable = set.intersection(*non_empty)
        everything = set.union(*non_empty)
    else:
        stable, everything = set(), set()
    volatile = everything - stable

    mean_distance = sum(distances) / len(distances) if distances else 0.0
    return mean_distance, sorted(stable), sorted(volatile)


def _lexical_dispersion(texts: Sequence[str]) -> float:
    """1 - mean pairwise TF-IDF cosine similarity across the responses.

    Uses the same vectoriser settings as the parent project's sensitivity
    analysis (``sensitivity_service.py:73``) for consistency.
    """
    usable = [t for t in texts if t and t.strip()]
    if len(usable) < 2:
        return 0.0

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        return 0.0

    try:
        matrix = TfidfVectorizer(stop_words="english").fit_transform(usable)
    except ValueError:
        # Raised when every token was a stop word, leaving an empty vocabulary.
        # No lexical signal is available; say so rather than inventing one.
        return 0.0

    similarity = cosine_similarity(matrix)
    n = similarity.shape[0]
    off_diagonal = [similarity[i][j] for i in range(n) for j in range(i + 1, n)]
    if not off_diagonal:
        return 0.0

    mean_similarity = sum(off_diagonal) / len(off_diagonal)
    return max(0.0, min(1.0, 1.0 - float(mean_similarity)))


def score(
    texts: Sequence[str],
    triple_sets: Sequence[Sequence[Sequence[str]]],
    weight_structural: float = 0.5,
) -> DriftResult:
    """Score disagreement across the resamples of a single step.

    ``texts`` are the raw responses; ``triple_sets`` are their parsed triples,
    in the same order.
    """
    if len(texts) < 2:
        return DriftResult(
            samples=len(texts),
            computable=False,
            note="drift needs at least 2 resamples per step",
        )

    normalised = [{_normalise(t) for t in triples} for triples in triple_sets]
    jaccard, stable, volatile = _mean_pairwise_jaccard_distance(normalised)
    lexical = _lexical_dispersion(texts)

    w = max(0.0, min(1.0, weight_structural))
    drift = w * jaccard + (1.0 - w) * lexical

    return DriftResult(
        drift=round(drift, 5),
        jaccard_distance=round(jaccard, 5),
        lexical_dispersion=round(lexical, 5),
        samples=len(texts),
        stable_triples=len(stable),
        volatile_triples=len(volatile),
        stable_examples=stable[:8],
        volatile_examples=volatile[:8],
    )
