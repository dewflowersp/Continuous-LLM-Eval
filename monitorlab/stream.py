"""The time-indexed signal and the worker that produces it.

This module is where "a score" becomes "a signal". ``TimeSeries`` is the buffer;
``StreamRunner`` is a background thread that repeatedly prompts the model, scores
the response, feeds each metric through its detectors and emits one
``StepRecord`` per step.

Threading model, which matters for Streamlit: the runner never touches
``st.session_state``. It pushes records onto a ``queue.Queue`` and the UI drains
that queue on each rerun. Streamlit's session state is not safe to mutate from a
thread it does not own, and a half-written buffer during a rerun shows up as a
chart that flickers or a KeyError mid-demo.

What each step costs in LLM calls:

    1                      the primary triple extraction
    resamples - 1          extra samples, only if drift is enabled
    prompts x groups       framing, and only on its cadence

With the defaults that is 3 calls for an ordinary step and 15 for a framing step,
which is the whole reason framing runs on a cadence.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Deque, Iterable, Sequence

from . import config, scenarios
from .detectors import MetricMonitor, Verdict
from .metrics import drift as drift_metric
from .metrics import framing as framing_metric
from .metrics import grounding as grounding_metric
from .metrics import structure as structure_metric


# --------------------------------------------------------------------- records
@dataclass
class MetricSample:
    """One metric at one step, with the detectors' verdict on it."""

    value: float
    center: float = 0.0
    lower: float = 0.0
    upper: float = 0.0
    band_ready: bool = False
    z: float = 0.0
    severity: str = "ok"
    change_point: bool = False
    reasons: list[str] = field(default_factory=list)

    @classmethod
    def from_verdict(cls, verdict: Verdict) -> "MetricSample":
        return cls(
            value=verdict.value,
            center=verdict.band.center,
            lower=verdict.band.lower,
            upper=verdict.band.upper,
            band_ready=verdict.band.ready,
            z=verdict.z,
            severity=verdict.severity,
            change_point=verdict.change_point,
            reasons=list(verdict.reasons),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "center": self.center,
            "lower": self.lower,
            "upper": self.upper,
            "band_ready": self.band_ready,
            "z": self.z,
            "severity": self.severity,
            "change_point": self.change_point,
            "reasons": self.reasons,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MetricSample":
        return cls(**{k: payload[k] for k in payload if k in cls.__dataclass_fields__})


@dataclass
class StepRecord:
    """Everything observed at one step, including the detail for drill-down."""

    step: int
    timestamp: float
    model: str
    temperature: float
    safeguard: bool
    perturbed: bool = False
    changed_now: bool = False
    plan_notes: list[str] = field(default_factory=list)
    metrics: dict[str, MetricSample] = field(default_factory=dict)
    raw_response: str = ""
    triples: list[list[str]] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def alarms(self) -> list[tuple[str, MetricSample]]:
        return [(name, s) for name, s in self.metrics.items() if s.severity != "ok"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "timestamp": self.timestamp,
            "model": self.model,
            "temperature": self.temperature,
            "safeguard": self.safeguard,
            "perturbed": self.perturbed,
            "changed_now": self.changed_now,
            "plan_notes": self.plan_notes,
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
            "raw_response": self.raw_response,
            "triples": self.triples,
            "detail": self.detail,
            "errors": self.errors,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StepRecord":
        return cls(
            step=payload["step"],
            timestamp=payload.get("timestamp", 0.0),
            model=payload.get("model", ""),
            temperature=payload.get("temperature", 0.0),
            safeguard=payload.get("safeguard", True),
            perturbed=payload.get("perturbed", False),
            changed_now=payload.get("changed_now", False),
            plan_notes=payload.get("plan_notes", []),
            metrics={
                k: MetricSample.from_dict(v) for k, v in (payload.get("metrics") or {}).items()
            },
            raw_response=payload.get("raw_response", ""),
            triples=payload.get("triples", []),
            detail=payload.get("detail", {}),
            errors=payload.get("errors", []),
        )


class TimeSeries:
    """A bounded history of steps, plus per-metric views for plotting."""

    def __init__(self, maxlen: int = 500):
        self.records: Deque[StepRecord] = deque(maxlen=maxlen)

    def append(self, record: StepRecord) -> None:
        self.records.append(record)

    def __len__(self) -> int:
        return len(self.records)

    def steps(self) -> list[int]:
        return [r.step for r in self.records]

    def series(self, metric: str) -> dict[str, list]:
        """Plot-ready arrays for one metric, skipping steps where it was not measured."""
        steps, values, lower, upper, center = [], [], [], [], []
        for record in self.records:
            sample = record.metrics.get(metric)
            if sample is None:
                continue
            steps.append(record.step)
            values.append(sample.value)
            center.append(sample.center)
            # Suppress the band until it is meaningful, so the chart does not
            # open with a hairline band around the very first sample.
            lower.append(sample.lower if sample.band_ready else None)
            upper.append(sample.upper if sample.band_ready else None)
        return {
            "steps": steps,
            "values": values,
            "center": center,
            "lower": lower,
            "upper": upper,
        }

    def events(self, metric: str) -> dict[str, list]:
        """Warn and alarm points for one metric, for marker overlays."""
        out: dict[str, list] = {"warn_x": [], "warn_y": [], "alarm_x": [], "alarm_y": []}
        for record in self.records:
            sample = record.metrics.get(metric)
            if sample is None or sample.severity == "ok":
                continue
            prefix = "alarm" if sample.severity == "alarm" else "warn"
            out[f"{prefix}_x"].append(record.step)
            out[f"{prefix}_y"].append(sample.value)
        return out

    def perturbation_steps(self) -> list[int]:
        return [r.step for r in self.records if r.changed_now]

    def latest(self) -> StepRecord | None:
        return self.records[-1] if self.records else None

    def alarm_log(self) -> list[dict[str, Any]]:
        rows = []
        for record in self.records:
            for name, sample in record.alarms:
                rows.append(
                    {
                        "step": record.step,
                        "metric": config.METRICS.get(name, {}).get("short", name),
                        "severity": sample.severity,
                        "value": round(sample.value, 4),
                        "why": "; ".join(sample.reasons) or "-",
                        "model": record.model,
                    }
                )
        rows.reverse()  # most recent first
        return rows


# ---------------------------------------------------------------------- runner
@dataclass
class StreamConfig:
    model: str
    text: str
    steps: int = 24
    resamples: int = 3
    temperature: float = 0.2
    interval: float = 0.0
    metrics: Sequence[str] = field(default_factory=lambda: list(config.DEFAULT_METRICS))
    framing_every: int = 4
    framing_category: str = framing_metric.DEFAULT_CATEGORY
    num_predict: int = 900
    use_safeguard: bool = True
    perturbations: Sequence[scenarios.Perturbation] = ()
    detector_params: dict[str, Any] = field(default_factory=dict)

    def enabled(self, metric: str) -> bool:
        return metric in self.metrics


class StreamRunner:
    """Runs the monitored stream on a background thread."""

    def __init__(self, client, settings: StreamConfig):
        self.client = client
        self.settings = settings
        self.queue: "queue.Queue[StepRecord | None]" = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.monitors: dict[str, MetricMonitor] = {}
        self.status = "idle"
        self.error = ""
        self._on_record: Callable[[StepRecord], None] | None = None
        self._perturbations_lock = threading.Lock()
        self._perturbations: tuple[scenarios.Perturbation, ...] = tuple(settings.perturbations)

        for metric in settings.metrics:
            spec = config.METRICS.get(metric, {})
            self.monitors[metric] = MetricMonitor(
                metric=metric,
                direction=spec.get("direction", "higher_is_better"),
                **settings.detector_params,
            )

    # ------------------------------------------------------------------ control
    def start(self, on_record: Callable[[StepRecord], None] | None = None) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._on_record = on_record
        self._stop.clear()
        self.status = "running"
        self._thread = threading.Thread(target=self._run, name="stream-runner", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def set_perturbations(self, perturbations: Iterable[scenarios.Perturbation]) -> None:
        """Replace the schedule. Takes effect from the next step that starts."""
        schedule = tuple(replace(p) for p in perturbations)
        with self._perturbations_lock:
            self._perturbations = schedule

    @property
    def perturbations(self) -> tuple[scenarios.Perturbation, ...]:
        with self._perturbations_lock:
            return self._perturbations

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def drain(self) -> list[StepRecord]:
        """Take everything produced since the last call. Called from the UI thread."""
        out: list[StepRecord] = []
        while True:
            try:
                item = self.queue.get_nowait()
            except queue.Empty:
                break
            if item is None:
                self.status = "finished" if not self._stop.is_set() else "stopped"
                continue
            out.append(item)
        return out

    # --------------------------------------------------------------------- work
    def _run(self) -> None:
        settings = self.settings
        base_system = ""
        try:
            base_system = compat_system_prompt() if settings.use_safeguard else ""
        except Exception as exc:  # noqa: BLE001
            self.error = f"could not load SYSTEM_PROMPT: {exc}"

        last_framing: framing_metric.FramingResult | None = None
        plan: scenarios.StepPlan | None = None

        try:
            for step in range(1, settings.steps + 1):
                if self._stop.is_set():
                    break

                plan = scenarios.plan_step(
                    step=step,
                    base_model=settings.model,
                    base_text=settings.text,
                    base_temperature=settings.temperature,
                    base_system_prompt=base_system,
                    perturbations=self.perturbations,
                    previous=plan,
                )

                record, last_framing = self._run_step(plan, last_framing)
                self.queue.put(record)
                if self._on_record is not None:
                    self._on_record(record)

                if settings.interval and step < settings.steps and not self._stop.is_set():
                    # Sleep in slices so Stop stays responsive on long intervals.
                    deadline = time.monotonic() + settings.interval
                    while time.monotonic() < deadline and not self._stop.is_set():
                        time.sleep(min(0.1, deadline - time.monotonic()))
        except Exception as exc:  # noqa: BLE001
            self.error = f"{type(exc).__name__}: {exc}"
            self.status = "error"
        finally:
            self.queue.put(None)

    def _run_step(
        self, plan: scenarios.StepPlan, last_framing: framing_metric.FramingResult | None
    ) -> tuple[StepRecord, framing_metric.FramingResult | None]:
        settings = self.settings
        errors: list[str] = []
        detail: dict[str, Any] = {}

        # ---- generate. The primary sample drives structure and grounding; all
        # samples together drive drift.
        want = max(1, settings.resamples if settings.enabled("drift") else 1)
        generations = self.client.generate_many(
            model=plan.model,
            prompts=[plan.text] * want,
            system=plan.system_prompt,
            temperature=plan.temperature,
            num_predict=settings.num_predict,
            max_workers=min(3, want),
        )
        primary = generations[0]
        if not primary.ok:
            errors.append(f"generation failed: {primary.error}")
        elif primary.is_empty:
            errors.append("model returned an empty response")

        values: dict[str, float] = {}

        # ---- structure
        structural = structure_metric.score(primary.text)
        if settings.enabled("robustness"):
            values["robustness"] = structural.robustness
        detail["structure"] = structural.as_row()
        if structural.parse_failed and primary.ok:
            errors.append(structural.note)

        # ---- grounding
        if settings.enabled("hallucination"):
            try:
                grounded = grounding_metric.score(plan.text, structural.triples)
                values["hallucination"] = grounded.hallucination
                detail["grounding"] = grounded.as_row()
                detail["grounding_detail"] = {
                    "triple_entities": grounded.triple_entities[:25],
                    "spacy_entities": grounded.spacy_entities[:25],
                    "ungrounded": grounding_metric.ungrounded_entities(grounded)[:25],
                    "stats": grounded.stats,
                    "support": grounded.support,
                    "thin_support": grounded.thin_support,
                    "note": grounded.note,
                }
            except Exception as exc:  # noqa: BLE001
                errors.append(f"grounding failed: {exc}")

        # ---- drift
        if settings.enabled("drift"):
            parsed = [structure_metric.score(g.text).triples for g in generations]
            drifted = drift_metric.score([g.text for g in generations], parsed)
            if drifted.computable:
                values["drift"] = drifted.drift
            detail["drift"] = drifted.as_row()
            detail["drift_detail"] = {
                "stable": drifted.stable_examples,
                "volatile": drifted.volatile_examples,
                "stable_count": drifted.stable_triples,
                "volatile_count": drifted.volatile_triples,
                "note": drifted.note,
            }

        # ---- framing, on a cadence because it costs a dozen extra calls
        if settings.enabled("bias"):
            due = settings.framing_every <= 1 or plan.step % settings.framing_every == 1
            if due:
                try:
                    framed = framing_metric.score(
                        client=self.client,
                        model=plan.model,
                        text=plan.text,
                        category=settings.framing_category,
                        # The deployment's temperature, not a raised one. An
                        # earlier version floored this at 0.4 to get livelier
                        # answers; all it bought was a noise floor thick enough
                        # to hide a real corpus switch.
                        temperature=plan.temperature,
                    )
                    last_framing = framed
                    if framed.computable:
                        values["bias"] = framed.bias
                    detail["framing"] = framed.as_row()
                    detail["framing_detail"] = {
                        "responses": framed.responses,
                        "per_prompt": {
                            k: {
                                "sensitivity": v["sensitivity"],
                                "sentiment": v["sentiment"],
                                "sentiments_by_group": v["sentiments_by_group"],
                            }
                            for k, v in framed.per_prompt.items()
                        },
                        "note": framed.note,
                    }
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"framing failed: {exc}")
            elif last_framing is not None:
                # Not measured this step. The value is deliberately *not* carried
                # forward into the detectors: repeating a stale sample would make
                # the EWMA band look far more confident than the evidence
                # supports.
                detail["framing"] = {**last_framing.as_row(), "stale": True}

        # ---- latency
        if settings.enabled("latency"):
            values["latency"] = round(primary.latency, 3)

        # ---- detectors
        samples: dict[str, MetricSample] = {}
        for metric, value in values.items():
            monitor = self.monitors.get(metric)
            if monitor is None:
                continue
            samples[metric] = MetricSample.from_verdict(monitor.update(value))

        return (
            StepRecord(
                step=plan.step,
                timestamp=time.time(),
                model=plan.model,
                temperature=plan.temperature,
                safeguard=bool(plan.system_prompt),
                perturbed=plan.perturbed,
                changed_now=plan.changed_now,
                plan_notes=list(plan.notes),
                metrics=samples,
                raw_response=primary.text[:6000],
                triples=[list(t) for t in structural.triples[:60]],
                detail=detail,
                errors=errors,
            ),
            last_framing,
        )


def compat_system_prompt() -> str:
    """The parent project's SYSTEM_PROMPT, which asks for DBpedia-style triples."""
    from . import compat

    return compat.settings().SYSTEM_PROMPT


def rebuild_series(records: Iterable[StepRecord], maxlen: int = 500) -> TimeSeries:
    """Rebuild a TimeSeries from records, e.g. when loading a recorded run."""
    series = TimeSeries(maxlen=maxlen)
    for record in records:
        series.append(record)
    return series


def replay_detectors(
    records: Sequence[StepRecord], **detector_params: Any
) -> list[StepRecord]:
    """Recompute every verdict from the recorded values with fresh detectors.

    A recording stores the measurements, which cost LLM calls and cannot be
    reproduced, and the verdicts, which are cheap and deterministic. Keeping the
    stored verdicts would freeze a recording against whatever thresholds happened
    to be in force the day it was made -- and this demo's own recordings predate
    two rounds of retuning, so they carry alarms the current code would not raise.

    Recomputing instead has a pleasant side effect worth using on stage: the
    detector sliders become live on a recorded incident. Move a threshold, replay,
    and watch a false alarm appear or a real one arrive later, with no model and no
    network in the loop.

    Returns new records; the input is left alone so a cached load stays pristine.
    """
    monitors: dict[str, MetricMonitor] = {}
    rebuilt: list[StepRecord] = []

    for record in records:
        samples: dict[str, MetricSample] = {}
        for metric, previous in record.metrics.items():
            monitor = monitors.get(metric)
            if monitor is None:
                spec = config.METRICS.get(metric, {})
                monitor = MetricMonitor(
                    metric=metric,
                    direction=spec.get("direction", "higher_is_better"),
                    **detector_params,
                )
                monitors[metric] = monitor
            samples[metric] = MetricSample.from_verdict(monitor.update(previous.value))
        rebuilt.append(replace(record, metrics=samples))

    return rebuilt
