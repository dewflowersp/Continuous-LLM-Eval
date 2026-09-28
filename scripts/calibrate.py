#!/usr/bin/env python3
"""Tune the change detectors against recorded runs, offline.

A recorded run stores each metric's raw value, and the detectors are pure
functions of that sequence. So the whole parameter search runs in milliseconds
with no Ollama and no network: load the runs, replay the numbers through
candidate ``MetricMonitor`` settings, and count what matters.

Scoring needs one piece of judgement, and getting it wrong is instructive. You
cannot ask every metric to respond to every perturbation. Swapping the retrieval
corpus should move biased framing and should leave grounding alone; dropping the
system prompt made responses shorter, so latency *improved*. Counting those as
misses rewards a detector that fires on everything.

So each series is first classified by how far its mean actually moved after the
perturbation, measured in its own noise widths and signed so that positive means
"moved the harmful way":

    shift >= --detect-threshold    the metric should fire. Score the delay.
    otherwise                      the metric should stay quiet. Any alarm here
                                   is over-sensitivity, counted as a false alarm.

Then two numbers matter, and they trade off:

    false alarms     alarms before the perturbation, anywhere in a baseline run,
                     or in a series that had nothing to detect. Every one is a
                     reason for an on-call engineer to mute the monitor, which
                     is the real failure mode.
    delay            steps from the perturbation to the first alarm. A detector
                     that needs ten steps to notice a swapped model is a report,
                     not a monitor.

A run with no perturbation at all is pure baseline, so every alarm in it is a
false alarm. Those runs are the most useful thing you can record.

Examples:
    calibrate.py                      # score the current defaults
    calibrate.py --sweep              # grid search, ranked by false alarms then misses
    calibrate.py --noise              # per-metric noise width, to sanity-check the units
"""

from __future__ import annotations

import argparse
import itertools
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from monitorlab import config, replay  # noqa: E402
from monitorlab.detectors import MetricMonitor, RunningSpread  # noqa: E402

DIM, RESET, BOLD = "\033[2m", "\033[0m", "\033[1m"


@dataclass
class Series:
    """One metric's raw values from one recorded run."""

    run: str
    metric: str
    steps: list[int]
    values: list[float]
    injected_at: int  # 0 when the run has no perturbation

    @property
    def direction(self) -> str:
        return config.METRICS.get(self.metric, {}).get("direction", "higher_is_better")

    def baseline(self) -> tuple[float, float]:
        """Level and noise width from the steps before the perturbation."""
        spread = RunningSpread()
        for value in self.values[: self.cut]:
            spread.update(value)
        return spread.mean, max(spread.spread, spread.floor)

    @property
    def cut(self) -> int:
        """Index of the first post-perturbation sample."""
        if not self.injected_at:
            return len(self.values)
        for index, step in enumerate(self.steps):
            if step >= self.injected_at:
                return index
        return len(self.values)

    def shift(self) -> float:
        """Post-perturbation shift in noise widths, signed so + means harmful."""
        after = self.values[self.cut:]
        if not after or self.cut < 3:
            return 0.0
        level, noise = self.baseline()
        delta = sum(after) / len(after) - level
        if self.direction == "higher_is_better":
            delta = -delta
        return delta / noise


def collect(paths: list[Path]) -> list[Series]:
    out: list[Series] = []
    for path in paths:
        meta, records = replay.load_run(path)
        if not records:
            continue
        injected = next((r.step for r in records if r.changed_now), 0)
        by_metric: dict[str, tuple[list[int], list[float]]] = {}
        for record in records:
            for metric, sample in record.metrics.items():
                steps, values = by_metric.setdefault(metric, ([], []))
                steps.append(record.step)
                values.append(sample.value)
        for metric, (steps, values) in sorted(by_metric.items()):
            out.append(Series(path.stem, metric, steps, values, injected))
    return out


@dataclass
class Score:
    false_alarms: int = 0
    delays: list[int] = field(default_factory=list)
    misses: int = 0
    expected: int = 0
    quiet_ok: int = 0
    quiet_total: int = 0

    @property
    def mean_delay(self) -> float:
        return sum(self.delays) / len(self.delays) if self.delays else float("inf")

    def __str__(self) -> str:
        delay = f"{self.mean_delay:.1f}" if self.delays else "-"
        return (f"false={self.false_alarms:<3} delay={delay:<5} "
                f"caught={self.expected - self.misses}/{self.expected}  "
                f"quiet={self.quiet_ok}/{self.quiet_total}")


def evaluate(
    series: list[Series], detect_threshold: float = 3.0, **params
) -> tuple[Score, list[str]]:
    """Replay every series through one parameter set."""
    score = Score()
    notes: list[str] = []
    for item in series:
        should_fire = item.shift() >= detect_threshold
        monitor = MetricMonitor(metric=item.metric, direction=item.direction, **params)
        first_after = 0
        late_alarms = 0

        for step, value in zip(item.steps, item.values):
            verdict = monitor.update(value)
            if not verdict.fired:
                continue
            why = "; ".join(verdict.reasons)
            if not item.injected_at:
                score.false_alarms += 1
                notes.append(f"{item.run}/{item.metric} step {step} (baseline run): {why}")
            elif step < item.injected_at:
                score.false_alarms += 1
                notes.append(f"{item.run}/{item.metric} step {step} (pre-injection): {why}")
            elif should_fire:
                first_after = first_after or step
            else:
                late_alarms += 1

        if not item.injected_at:
            continue
        if should_fire:
            score.expected += 1
            if first_after:
                score.delays.append(first_after - item.injected_at)
            else:
                score.misses += 1
                notes.append(f"{item.run}/{item.metric}: missed a "
                             f"{item.shift():.1f}n shift")
        else:
            score.quiet_total += 1
            if late_alarms:
                score.false_alarms += late_alarms
                notes.append(f"{item.run}/{item.metric}: {late_alarms} alarm(s) on a "
                             f"{item.shift():+.1f}n shift, which is nothing to find")
            else:
                score.quiet_ok += 1
    return score, notes


def show_noise(series: list[Series], detect_threshold: float) -> None:
    """Per-metric noise width, so the noise-unit thresholds can be sanity-checked."""
    print(f"\n{BOLD}Noise width before the perturbation{RESET}")
    print(f"{DIM}The unit every ph_* and cusum_* threshold is measured in.{RESET}\n")
    print(f"{'run':<40} {'metric':<14} {'n':>3} {'level':>9} {'noise':>8} "
          f"{'shift':>8}  expectation")
    for item in series:
        if item.cut < 3:
            continue
        level, noise = item.baseline()
        shift = item.shift()
        if not item.injected_at:
            verdict = f"{DIM}baseline, must stay quiet{RESET}"
        elif shift >= detect_threshold:
            verdict = "should fire"
        else:
            verdict = f"{DIM}nothing to find, must stay quiet{RESET}"
        print(f"{item.run[:39]:<40} {item.metric:<14} {item.cut:>3} "
              f"{level:>9.3f} {noise:>8.4f} {shift:>+7.1f}n  {verdict}")
    print(f"\n{DIM}'shift' is signed so positive means the metric moved the harmful "
          f"way. Below about {detect_threshold:.0f}n a shift is at the edge of what "
          f"an online detector can find, and the honest fix is more samples per "
          f"step, not a looser threshold.{RESET}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", type=Path, help="defaults to every recorded run")
    ap.add_argument("--sweep", action="store_true", help="grid search instead of scoring defaults")
    ap.add_argument("--noise", action="store_true", help="report per-metric noise width")
    ap.add_argument("--detect-threshold", type=float, default=3.0,
                    help="noise widths a shift must exceed before detection is expected")
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    paths = args.runs or [path for path, _ in replay.list_runs()]
    if not paths:
        print("No recorded runs. Produce one with scripts/rehearse.py first.")
        sys.exit(1)

    series = collect(paths)
    print(f"{len(series)} metric series from {len(paths)} run(s)")
    for path in paths:
        print(f"  {DIM}{path.name}{RESET}")

    if args.noise:
        show_noise(series, args.detect_threshold)
        return

    if not args.sweep:
        score, notes = evaluate(series, detect_threshold=args.detect_threshold)
        print(f"\n{BOLD}Current defaults{RESET}: {score}")
        for note in notes:
            print(f"  {note}")
        return

    grid = {
        "z_threshold": [3.5, 4.0, 4.5, 5.0],
        "ph_delta": [0.25, 0.5, 0.75, 1.0],
        "ph_lambda": [2.0, 3.0, 4.0, 5.0, 6.0],
        "cusum_slack": [0.5, 0.75, 1.0],
        "cusum_threshold": [3.0, 4.0, 6.0, 8.0],
    }
    keys = list(grid)
    results = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, combo))
        score, _ = evaluate(series, detect_threshold=args.detect_threshold, **params)
        results.append((score, params))

    # False alarms first: a monitor people mute detects nothing at all. Misses
    # before delay, because a detector that is fast on the one thing it still
    # notices is not a good trade.
    results.sort(key=lambda pair: (pair[0].false_alarms, pair[0].misses, pair[0].mean_delay))

    print(f"\n{BOLD}Best of {len(results)} parameter sets{RESET}")
    for score, params in results[: args.top]:
        setting = "  ".join(f"{k}={v}" for k, v in params.items())
        print(f"  {str(score):<38} {setting}")

    best = results[0][1]
    print(f"\n{DIM}Lowest false-alarm count with the shortest delay:{RESET}")
    for key, value in best.items():
        print(f"    {key} = {value}")


if __name__ == "__main__":
    main()
