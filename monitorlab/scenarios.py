"""Sample texts and mid-stream perturbations.

The abstract promises monitoring "while models, prompts, retrieval corpora,
safeguards, and the external world change". A perturbation is how the demo makes
one of those changes happen on cue, without telling the detectors:

``switch_model``     the model is swapped underneath the stream
``raise_temperature``  decoding gets noisier
``drop_safeguard``   the schema-enforcing system prompt is removed
``switch_corpus``    the input text is replaced with a different document

Each maps to a real production incident: a rollout, a config change, a prompt
regression, a shift in what users are sending. The detectors are never informed,
so whether they notice is an honest test rather than a scripted one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Sequence

from . import config

PERTURBATIONS = {
    "none": {
        "label": "No perturbation (clean baseline)",
        "blurb": "Nothing changes. Use it to show what a healthy signal looks like and to let the adaptive band converge.",
    },
    "switch_model": {
        "label": "Swap the model",
        "blurb": "Simulates a rollout. Expect an immediate, sustained level shift in robustness and latency.",
    },
    "raise_temperature": {
        "label": "Raise the temperature",
        "blurb": "Simulates a decoding-config change. Hits drift hardest, since the resamples start disagreeing.",
    },
    "drop_safeguard": {
        "label": "Remove the system prompt",
        "blurb": "Simulates a prompt regression. Robustness falls off a cliff because nothing is asking for the DBpedia schema any more.",
    },
    "switch_corpus": {
        "label": "Switch the input text",
        "blurb": "Simulates the world changing. The subtlest of the four, and the best argument for adaptive thresholds.",
    },
}


@dataclass
class Sample:
    id: str
    label: str
    text: str
    targets: str = ""
    expect: str = ""
    teaching_note: str = ""


@lru_cache(maxsize=1)
def load_samples() -> tuple[Sample, ...]:
    """Preset texts from data/samples.json."""
    if not config.SAMPLES_FILE.is_file():
        return ()
    payload = json.loads(config.SAMPLES_FILE.read_text(encoding="utf-8"))
    return tuple(
        Sample(
            id=entry["id"],
            label=entry.get("label", entry["id"]),
            text=entry.get("text", "").strip(),
            targets=entry.get("targets", ""),
            expect=entry.get("expect", ""),
            teaching_note=entry.get("teaching_note", ""),
        )
        for entry in payload.get("samples", [])
    )


def sample_by_id(sample_id: str) -> Sample | None:
    return next((s for s in load_samples() if s.id == sample_id), None)


@dataclass
class Perturbation:
    """A single scheduled change, applied from ``at_step`` onward."""

    kind: str = "none"
    at_step: int = 0
    to_model: str = ""
    to_temperature: float = 0.9
    to_sample_id: str = ""

    @property
    def active(self) -> bool:
        return self.kind != "none" and self.at_step > 0

    def applies_at(self, step: int) -> bool:
        return self.active and step >= self.at_step

    def fires_at(self, step: int) -> bool:
        """True only on the step the change first takes effect."""
        return self.active and step == self.at_step

    def describe(self) -> str:
        if not self.active:
            return "no perturbation"
        if self.kind == "switch_model":
            return f"step {self.at_step}: model -> {self.to_model or '(unset)'}"
        if self.kind == "raise_temperature":
            return f"step {self.at_step}: temperature -> {self.to_temperature}"
        if self.kind == "drop_safeguard":
            return f"step {self.at_step}: system prompt removed"
        if self.kind == "switch_corpus":
            return f"step {self.at_step}: text -> {self.to_sample_id or '(unset)'}"
        return f"step {self.at_step}: {self.kind}"


def describe_schedule(perturbations: Sequence[Perturbation]) -> str:
    active = sorted((p for p in perturbations if p.active), key=lambda p: p.at_step)
    if not active:
        return "no perturbation"
    return "; ".join(p.describe() for p in active)


@dataclass
class StepPlan:
    """The fully-resolved settings for one step, after any perturbation."""

    step: int
    model: str
    text: str
    temperature: float
    system_prompt: str
    perturbed: bool = False
    changed_now: bool = False
    notes: list[str] = field(default_factory=list)

    def as_row(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "model": self.model,
            "temperature": self.temperature,
            "safeguard": bool(self.system_prompt),
            "perturbed": self.perturbed,
        }

    @property
    def signature(self) -> tuple[str, str, float, str]:
        return (self.model, self.text, self.temperature, self.system_prompt)


def plan_step(
    step: int,
    base_model: str,
    base_text: str,
    base_temperature: float,
    base_system_prompt: str,
    perturbations: Sequence[Perturbation] = (),
    previous: StepPlan | None = None,
) -> StepPlan:
    """Resolve what to actually send for a given step.

    Every perturbation whose ``at_step`` has been reached is applied, earliest
    first, so a later change of the same kind overrides an earlier one
    (temperature 1.2 at step 6, then 1.5 at step 15).

    ``changed_now`` compares against ``previous`` rather than asking whether
    some ``at_step`` equals ``step``: the schedule can be edited mid-run, and a
    change scheduled for a step that has already passed takes effect now, which
    is when the chart should mark it.
    """
    plan = StepPlan(
        step=step,
        model=base_model,
        text=base_text,
        temperature=base_temperature,
        system_prompt=base_system_prompt,
    )

    due = sorted((p for p in perturbations if p.applies_at(step)), key=lambda p: p.at_step)
    for perturbation in due:
        if perturbation.kind == "switch_model" and perturbation.to_model:
            plan.notes.append(f"model {plan.model} -> {perturbation.to_model}")
            plan.model = perturbation.to_model
        elif perturbation.kind == "raise_temperature":
            plan.notes.append(f"temperature {plan.temperature} -> {perturbation.to_temperature}")
            plan.temperature = perturbation.to_temperature
        elif perturbation.kind == "drop_safeguard":
            plan.notes.append("system prompt removed")
            plan.system_prompt = ""
        elif perturbation.kind == "switch_corpus" and perturbation.to_sample_id:
            replacement = sample_by_id(perturbation.to_sample_id)
            if replacement and replacement.text:
                plan.notes.append(f"text -> {replacement.label}")
                plan.text = replacement.text

    plan.perturbed = plan.signature != (base_model, base_text, base_temperature, base_system_prompt)
    plan.changed_now = previous is not None and plan.signature != previous.signature
    return plan
