#!/usr/bin/env python3
"""Run a monitored stream headlessly, print the table, and record it.

Two jobs:

1. Rehearsal. Prints per-step values, alarms and a timing summary, so you know
   what a beat will cost in wall-clock minutes before you are standing in front
   of people.
2. Pre-recording. Every run is written to data/runs/*.jsonl, which the dashboard
   can replay instantly with no Ollama and no network. This is how you prepare
   material that is too slow to produce live -- a 70B model, or a long run that
   shows a slow creep -- and how you build the fallback for a bad-wifi day.

Examples:
    rehearse.py --sample grounded --steps 10
    rehearse.py --sample grounded --steps 16 --perturbation drop_safeguard --at 9
    rehearse.py --sample charged --metrics bias,latency --framing-every 1 --steps 6
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from monitorlab import config, replay, scenarios  # noqa: E402
from monitorlab.metrics import framing  # noqa: E402
from monitorlab.ollama_client import OllamaClient  # noqa: E402
from monitorlab.stream import StreamConfig, StreamRunner  # noqa: E402

GREEN, YELLOW, RED, DIM, RESET = "\033[32m", "\033[33m", "\033[31m", "\033[2m", "\033[0m"
SEVERITY = {"ok": "", "warn": YELLOW, "alarm": RED}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="llama3.2:3b")
    ap.add_argument("--sample", default="grounded", help="preset id from data/samples.json")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--resamples", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--metrics", default=",".join(config.DEFAULT_METRICS))
    ap.add_argument("--framing-every", type=int, default=4)
    ap.add_argument("--perturbation", default="none", choices=list(scenarios.PERTURBATIONS))
    ap.add_argument("--at", type=int, default=0, help="step at which the perturbation fires")
    ap.add_argument("--to-model", default="", help="target for --perturbation switch_model")
    ap.add_argument("--to-sample", default="", help="target for --perturbation switch_corpus")
    ap.add_argument("--to-temperature", type=float, default=1.2)
    ap.add_argument("--no-record", action="store_true")
    args = ap.parse_args()

    sample = scenarios.sample_by_id(args.sample)
    if sample is None:
        ap.error(f"unknown sample {args.sample!r}; available: "
                 + ", ".join(s.id for s in scenarios.load_samples()))

    client = OllamaClient()
    if not client.is_up():
        print(f"{RED}Ollama is not reachable at {client.base_url}{RESET}")
        sys.exit(1)

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip() in config.METRICS]
    if not metrics:
        ap.error("no valid metrics selected")

    perturbation = scenarios.Perturbation(
        kind=args.perturbation,
        at_step=args.at or (args.steps // 2 if args.perturbation != "none" else 0),
        to_model=args.to_model,
        to_sample_id=args.to_sample,
        to_temperature=args.to_temperature,
    )

    settings = StreamConfig(
        model=args.model,
        text=sample.text,
        steps=args.steps,
        resamples=args.resamples,
        temperature=args.temperature,
        metrics=metrics,
        framing_every=args.framing_every,
        perturbations=[perturbation],
    )

    writer = None
    if not args.no_record:
        writer = replay.RunWriter(
            replay.new_run_path(f"{args.sample}-{args.perturbation}"),
            replay.RunMeta(
                model=settings.model, sample_id=args.sample,
                text_preview=sample.text[:220], steps=settings.steps,
                resamples=settings.resamples, temperature=settings.temperature,
                metrics=list(metrics), perturbation=perturbation.describe(),
                note="recorded by rehearse.py",
            ),
        )

    print(f"\n{sample.label}  |  {args.model}  |  {len(metrics)} metrics  |  {perturbation.describe()}")
    calls = 1 + (settings.resamples - 1 if "drift" in metrics else 0)
    budget = f"~{calls} calls/step"
    if "bias" in metrics:
        budget += f", plus {framing.probe_calls(settings.framing_category)} on every framing step"
    print(f"{DIM}{budget}{RESET}\n")

    header = f"{'step':>4}  " + "  ".join(f"{config.METRICS[m]['short']:>10}" for m in metrics)
    print(header)
    print("-" * len(header))

    runner = StreamRunner(client, settings)
    started = time.time()
    runner.start(on_record=writer.append if writer else None)

    seen = 0
    alarm_rows: list[str] = []
    while runner.running or seen < settings.steps:
        drained = runner.drain()
        if not drained:
            if not runner.running:
                break
            time.sleep(0.2)
            continue
        for record in drained:
            seen += 1
            cells = []
            for metric in metrics:
                sampled = record.metrics.get(metric)
                if sampled is None:
                    cells.append(f"{DIM}{'-':>10}{RESET}")
                    continue
                colour = SEVERITY[sampled.severity]
                cells.append(f"{colour}{sampled.value:>10.3f}{RESET}")
            marker = "  <<< perturbation" if record.changed_now else ""
            print(f"{record.step:>4}  " + "  ".join(cells) + marker)
            for name, sampled in record.alarms:
                alarm_rows.append(
                    f"  step {record.step:>3}  {config.METRICS[name]['short']:<10} "
                    f"{sampled.severity:<5}  {'; '.join(sampled.reasons)}"
                )
            if record.errors:
                print(f"       {YELLOW}{' | '.join(record.errors[:2])}{RESET}")

    if writer:
        writer.close()

    elapsed = time.time() - started
    print(f"\n{DIM}{seen} steps in {elapsed:.0f}s "
          f"({elapsed / max(seen, 1):.1f}s/step){RESET}")
    if runner.error:
        print(f"{RED}runner error: {runner.error}{RESET}")

    if alarm_rows:
        print(f"\nAlarms and warnings ({len(alarm_rows)}):")
        for row in alarm_rows:
            print(row)
    else:
        print(f"\n{GREEN}No alarms.{RESET}")

    if perturbation.active:
        print(f"\n{DIM}Perturbation fired at step {perturbation.at_step}. "
              f"A detector that earns its keep alarms within a step or two of "
              f"that, and stays quiet before it.{RESET}")

    if writer:
        print(f"\nRecorded to {writer.path.relative_to(config.DEMO_DIR)}")
        print(f"{DIM}Replay it from the dashboard's Replay mode.{RESET}")


if __name__ == "__main__":
    main()
