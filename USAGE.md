# How to use the monitoring demo

This guide explains how the dashboard works and what every control in the left
sidebar does. For setup, see [README.md](README.md).

---

## How the demo works

At a high level the app does this on every **step**:

1. Send the current input text to a local **Ollama** model (with an optional
   system prompt that asks for DBpedia-style triples).
2. Parse the reply into subject–predicate–object triples.
3. Score the reply on the metrics you enabled (grounding, robustness, drift,
   biased framing, latency).
4. Feed each score into online **change detectors** (EWMA band, robust z-score,
   Page-Hinkley, CUSUM).
5. Plot the values over time, mark warnings/alarms, and keep a per-step drill-down.

A run is a **stream**: the same configuration is repeated for N steps so you can
see whether a signal stays stable or shifts when something underneath changes
(model, temperature, prompt, or input text).

```
Input text ──► Ollama (extract triples) ──► metric scorers ──► detectors ──► charts
                     │                              │
                     └── optional mid-stream ───────┘
                         perturbation (swap model,
                         raise temp, drop prompt,
                         switch corpus)
```

Two modes:

| Mode | Needs Ollama? | What it does |
| --- | --- | --- |
| **Live stream** | Yes | Calls your local model in real time and (optionally) records JSONL under `data/runs/`. |
| **Replay a recording** | No | Plays back a saved run. Detector sliders still work — verdicts are recomputed from the recorded metric values. |

Start the UI from this folder:

```bash
./setup.sh    # once
./run.sh      # opens http://localhost:8502
```

---

## Main panel

After you start a run, the centre of the page shows:

- **Header KPIs** — current model, step count, alarm/warning totals, status.
- **Metric tiles** — latest value and step-to-step delta for each enabled metric.
- **Charts** — one time series per metric:
  - solid line = measured value
  - dotted line = EWMA centre
  - shaded band = adaptive threshold
  - open circle = warning, **X** = alarm
  - magenta dashed line = where a perturbation was injected
- **Alarm log** — every warn/alarm with metric and reason.
- **Step detail** — triples, grounding entities, drift stable/volatile sets,
  framing per-group responses, and the raw model reply.
- **Download metrics as CSV** — export the numeric series after a run.

---

## Left sidebar — every control

Controls below apply to **Live stream** unless noted. Replay mode shows a shorter
set (recording picker, speed, Play/Reset, plus Detector settings).

### Theme

| Control | What it does |
| --- | --- |
| **Light / Dark** | Switches the UI colours and Plotly chart template. Does not restart the server. |

### Mode

| Control | What it does |
| --- | --- |
| **Live stream** | Run a fresh analysis against Ollama. |
| **Replay a recording** | Load a JSONL from `data/runs/` and play it back. |

### 1. Content to analyse

| Control | What it does |
| --- | --- |
| **Preset** | Choose a built-in sample from `data/samples.json`, or **Custom text**. |
| **Text** | The paragraph the model is asked to extract triples from. Presets fill this box; Custom leaves it empty for you to paste. |

Built-in presets:

| Preset | Description |
| --- | --- |
| **Grounded** | Real entities and clear relations. |
| **Fabricated** | Same shape with invented names. |
| **Ambiguous** | Underspecified prose. |
| **Contested** | Critics vs supporters framing. |
| **Neutral sport** | Dense entities, low-stakes sports prose. |

### 2. Model

| Control | What it does |
| --- | --- |
| **Ollama model** | Which installed model does the extraction (and framing probes). Slow/large models show a warning; they are fine for recording, awkward for a live loop. |

Requires a reachable Ollama at `http://localhost:11434` (override with
`DEMO_OLLAMA_URL`).

### 3. Metrics

Tick which signals to compute. Cost rises with each one you add.

| Checkbox | Signal | Notes |
| --- | --- | --- |
| **Grounding** | `(rule_definition + ner_consistency) / 2` | On by default. |
| **Robustness** | `(ICR + IPR + CI) / 3` | On by default. |
| **Drift** | Jaccard + lexical dispersion over resamples | Needs **Resamples ≥ 2**. On by default. |
| **Bias** | Sentiment spread across demographic groups | Extra LLM calls per step (shown as `+N calls/step`). Off by default. |
| **Latency** | Wall-clock seconds for the extraction call | On by default. |

You must leave at least one metric enabled or **Start** will refuse.

### 4. Stream

| Control | Range / default | What it does |
| --- | --- | --- |
| **Steps** | 4–120 (default 24) | How many times to score the (current) input. |
| **Resamples per step** | 1–5 (default 3) | Extra generation calls used only for **Drift**. Disabled when Drift is off. |
| **Temperature** | 0.0–1.5 (default 0.2) | Decoding temperature for the extraction call. |
| **Pause between steps (s)** | 0–10 (default 0) | Artificial delay between steps. |
| **Use the schema system prompt** | on/off (default on) | When on, the model is asked for DBpedia-style triples. Turning it off mid-run is what **Remove the system prompt** does as a perturbation. |
| **Framing cadence (every N steps)** | 1–10 (default 4) | How often Bias is recomputed. Disabled when Bias is off. Between measurements the chart holds the last value (marked stale in the drill-down). |

### 5. Perturbations

Optional mid-stream changes. Press **Add a perturbation** once per change; each
entry has its own **Remove** button. With no entries the run is a clean baseline.

| Change | What happens from the chosen step |
| --- | --- |
| **Swap the model** | Switches to another installed Ollama model. |
| **Raise the temperature** | Sets a new temperature (you pick the value). |
| **Remove the system prompt** | Turns off the schema system prompt. |
| **Switch the input text** | Swaps to another preset’s text. |

Extra fields appear when needed:

- **At step** — when the change fires (must be within the run length).
- **Switch to** (model or preset) — target for model / corpus swaps.
- **New temperature** — target for a temperature raise.

Changes stack: every entry whose step has been reached is applied, earliest
first, so a later change of the same kind overrides an earlier one (for example
temperature 1.2 at step 6, then 1.5 at step 15).

The list is live. Adding, editing or removing an entry while a run is going
takes effect from the next step. An entry set to a step that has already passed
applies immediately.

A magenta dashed line is drawn at every step where the settings actually sent to
the model changed, including changes made mid-run.

### Detector settings (expander)

Shared by Live and Replay. Thresholds are in **noise widths**, not raw metric
units, so the same numbers work for grounding in \[0, 1\] and latency in seconds.

| Control | What it does |
| --- | --- |
| **EWMA lambda** | How fast the centre line tracks the signal. |
| **Band width (k sigma)** | Width of the adaptive warn/alarm band around the EWMA. |
| **Warm-up steps** | Steps before the band is trusted (no alarms during warm-up). |
| **Robust z threshold** | Median/MAD z-score cutoff for a step-level spike. |
| **Page-Hinkley delta / lambda** | Sensitivity and threshold for gradual mean shifts. |
| **CUSUM slack / threshold** | Cumulative sum detector for sustained shifts. |

In **Replay**, change these and press Play again to recompute verdicts from the
same recorded metric values.

### Record / transport

| Control | What it does |
| --- | --- |
| **Record this run** | When checked, writes JSONL to `data/runs/` for later Replay. |
| **Start** | Begins the live stream (needs Ollama online and non-empty Text). |
| **Stop** | Stops the background worker after the current step finishes draining. |
| **Clear** | Clears charts and state when idle. |

### Replay-only controls

| Control | What it does |
| --- | --- |
| **Recording** | Pick a file from `data/runs/`. |
| **Speed** | `instant` dumps the whole run at once; `1x`–`8x` paces playback. |
| **Play / Reset** | Start or rewind the selected recording. |

---

## Using custom text

You do not need a preset. Any paragraph works.

1. Set **Mode** to **Live stream**.
2. Under **1. Content to analyse**, open **Preset** and choose **Custom text**.
3. Paste or type into the **Text** box.
4. Pick an **Ollama model** and the **metrics** you care about.
5. Optionally set Steps / Temperature / Detector settings / a Perturbation.
6. Press **Start**.

Notes:

- Named people, places, or organisations give Grounding more entities to check.
- For a **Switch the input text** perturbation, start from a preset — that option
  only lists other samples from `samples.json`, not free-form custom strings. To
  compare two custom texts, run them as two separate streams (or temporarily add
  a sample to `data/samples.json`).

### Editing a preset as a starting point

You can pick a built-in preset, then edit the **Text** area before Start. The run
uses whatever is in the box at Start time. The recorded `sample_id` still
reflects the preset you selected; the text itself is whatever you left in the box.

### Adding a lasting custom sample

To make a reusable sidebar entry, append an object to the `samples` array in
`data/samples.json` and restart the app.

---

## Quick troubleshooting

| Symptom | Likely fix |
| --- | --- |
| Sidebar shows “Ollama unreachable” | Start Ollama (`ollama serve`) and install at least one model (`ollama pull llama3.2:3b`). Replay still works. |
| Start is disabled | Text is empty, a stream is already running, or Ollama is offline. |
| First step is slow | Cold model load. Warm with `scripts/warm_cache.py --warm-models` before a live session. |
| `gpt-oss:*` / `gemma4:*` empty response / no triples | Reasoning models fill a `thinking` field first; that shares the `num_predict` budget with the answer. The client floors `num_predict` to 4096 for those families. Override with `DEMO_REASONING_NUM_PREDICT`. |
| Drift checkbox on but flat | Need Resamples ≥ 2; with 1 resample there is nothing to compare. |
| Bias updates infrequently | Check Framing cadence (it may only recompute every N steps). |
