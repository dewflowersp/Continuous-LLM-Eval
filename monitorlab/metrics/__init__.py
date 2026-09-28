"""The four monitored metrics.

Three reproduce the parent project's own derived metrics; ``drift`` is new to
this demo. Each module exposes a ``score()`` returning a dataclass with the
headline number plus enough detail to explain it on stage.

    structure  -> robustness   = (ICR + IPR + CI) / 3
    grounding  -> hallucination = (rule_definition + ner_consistency) / 2
    drift      -> drift         = disagreement across k resamples
    framing    -> bias          = (sensitivity + sentiment) / 2
"""

from . import drift, framing, grounding, structure

__all__ = ["drift", "framing", "grounding", "structure"]
