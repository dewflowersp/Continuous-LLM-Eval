# LLM Monitoring Demo

Streamlit app that scores a local Ollama model on each step and runs online change detectors over those scores. You can run a live stream against Ollama, or replay a saved recording without a model.

**Docs:** [USAGE.md](USAGE.md) — dashboard controls, live/replay modes, custom text, and troubleshooting.

## Requirements

- Python **3.10–3.13** (not 3.14; spaCy/thinc has no cp314 wheels)
- [Ollama](https://ollama.com) running locally for **Live stream** mode (optional for Replay)
- macOS or Linux with `bash`, `curl`, and `pip`

## Install

From this folder:

```bash
./setup.sh
```

This creates `.venv`, installs `requirements.txt`, downloads the spaCy `en_core_web_sm` model, and runs a self-check. Safe to re-run.

Optional checks after setup:

```bash
.venv/bin/python scripts/selfcheck.py
.venv/bin/python scripts/warm_cache.py --warm-models
```

`warm_cache.py` preloads Ollama models and fills `.kg_cache/` so the first grounding steps do not hit DBpedia over a slow network.

## Run

```bash
./run.sh
```

Opens the dashboard at [http://localhost:8502](http://localhost:8502).

Override the port if needed:

```bash
DEMO_PORT=8503 ./run.sh
```

## How to use

See **[USAGE.md](USAGE.md)** for the full walkthrough of every sidebar control.

### Modes

| Mode | Needs Ollama? | What it does |
| --- | --- | --- |
| **Live stream** | Yes | Calls your local model each step and can save a JSONL under `data/runs/`. |
| **Replay a recording** | No | Plays back a file from `data/runs/`. Detector settings still apply to the recorded metrics. |

### Live stream (sidebar)

1. Choose **Live stream**.
2. Pick a content preset (or Custom text) and the metrics to enable.
3. Select an Ollama model, temperature, and optional system-prompt safeguard.
4. Optionally choose a mid-stream perturbation (`switch_model`, `raise_temperature`, `drop_safeguard`, `switch_corpus`) and the step where it starts.
5. Set the number of steps, then start the run.
6. Enable **Record run** if you want a replayable JSONL written to `data/runs/`.

### Replay

1. Choose **Replay a recording**.
2. Select a file from `data/runs/`.
3. Adjust replay speed and detector settings, then Play.

Sample recordings included (25 steps each):

| File | Model | Perturbation |
| --- | --- | --- |
| `run-20260925-122620.jsonl` | `lfm2.5:8b` | none |
| `run-20260925-225343.jsonl` | `gpt-oss:20b` | none |
| `run-20260927-190858.jsonl` | `gemma4:e4b` | none |
| `run-20260927-125117.jsonl` | `gpt-oss:20b` | raise temperature |
| `run-20260927-221451.jsonl` | `gemma4:e4b` | raise temperature |

### Useful environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `DEMO_PORT` | `8502` | Streamlit port |
| `DEMO_OLLAMA_URL` | `http://localhost:11434` | Ollama base URL |
| `DEMO_OLLAMA_TIMEOUT` | `180` | Request timeout (seconds) |
| `DEMO_REASONING_NUM_PREDICT` | `4096` | Token budget floor for reasoning models (`gpt-oss`, `gemma4`, …) |

## Project layout

```
.
├── README.md           # this file
├── USAGE.md            # how to use the dashboard
├── app.py              # Streamlit dashboard
├── setup.sh / run.sh   # install and launch
├── requirements.txt
├── monitorlab/         # scoring, detectors, Ollama client, replay
├── scripts/            # selfcheck, warm_cache, calibrate, rehearse
└── data/
    ├── samples.json    # preset input texts
    └── runs/           # JSONL recordings for replay
```

## Notes

- Pull any model you want to use live, e.g. `ollama pull llama3.2:3b`.
- New live runs with recording enabled appear under `data/runs/` and can be replayed later.
- First grounding scores may call DBpedia unless `.kg_cache/` was warmed.
