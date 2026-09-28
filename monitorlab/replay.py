"""Record runs to JSONL and play them back.

The insurance policy for a live session. A recorded run replays with every
number, alarm and drill-down intact, at whatever speed you choose, with no
Ollama and no network. If the venue wifi is hostile or a model decides to take
forty seconds a step, the demo continues.

Recording is also how you prepare material that is too slow to produce live: a
70B model, or a hundred-step run that shows a slow creep. Record it the night
before, replay it in seconds.

The format is one JSON object per line -- the same ``StepRecord.to_dict()`` the
UI already consumes -- preceded by a single metadata line. Being append-only
means a run interrupted by Ctrl-C is still a valid, replayable file.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from . import config
from .stream import StepRecord


@dataclass
class RunMeta:
    """The header line: enough context to know what you are looking at."""

    created: float = field(default_factory=time.time)
    model: str = ""
    sample_id: str = ""
    text_preview: str = ""
    steps: int = 0
    resamples: int = 0
    temperature: float = 0.0
    metrics: list[str] = field(default_factory=list)
    perturbation: str = ""
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"_meta": True, **self.__dict__}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RunMeta":
        clean = {k: v for k, v in payload.items() if k in cls.__dataclass_fields__}
        return cls(**clean)

    @property
    def label(self) -> str:
        stamp = time.strftime("%b %d %H:%M", time.localtime(self.created))
        bits = [stamp]
        if self.model:
            bits.append(self.model)
        if self.steps:
            bits.append(f"{self.steps} steps")
        if self.perturbation and self.perturbation != "no perturbation":
            bits.append(self.perturbation)
        return "  |  ".join(bits)


class RunWriter:
    """Append-only JSONL writer. Usable as a context manager."""

    def __init__(self, path: Path, meta: RunMeta):
        self.path = path
        self.meta = meta
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8")
        self._write(meta.to_dict())

    def _write(self, payload: dict[str, Any]) -> None:
        self._handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        # Flushed per record so a killed process still leaves a usable file.
        self._handle.flush()

    def append(self, record: StepRecord) -> None:
        self._write(record.to_dict())

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()

    def __enter__(self) -> "RunWriter":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def new_run_path(prefix: str = "run") -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return config.RUNS_DIR / f"{prefix}-{stamp}.jsonl"


def load_run(path: Path) -> tuple[RunMeta, list[StepRecord]]:
    """Read a recorded run. Malformed lines are skipped rather than fatal."""
    meta = RunMeta()
    records: list[StepRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("_meta"):
                meta = RunMeta.from_dict(payload)
            else:
                try:
                    records.append(StepRecord.from_dict(payload))
                except (KeyError, TypeError):
                    continue
    return meta, records


def list_runs() -> list[tuple[Path, RunMeta]]:
    """Recorded runs, newest first."""
    if not config.RUNS_DIR.is_dir():
        return []
    found = []
    for path in config.RUNS_DIR.glob("*.jsonl"):
        try:
            meta, _ = load_run(path)
        except OSError:
            continue
        found.append((path, meta))
    found.sort(key=lambda pair: pair[1].created, reverse=True)
    return found


def iter_replay(records: list[StepRecord], speed: float = 1.0) -> Iterator[StepRecord]:
    """Yield records with their original inter-step gaps, scaled by ``speed``.

    ``speed=0`` yields everything immediately, which is what you want when
    jumping straight to the interesting part of a run.
    """
    previous: float | None = None
    for record in records:
        if speed > 0 and previous is not None:
            gap = (record.timestamp - previous) / speed
            if 0 < gap < 30:
                time.sleep(gap)
        previous = record.timestamp
        yield record
