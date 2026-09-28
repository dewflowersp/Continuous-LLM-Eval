"""Continuous LLM monitoring -- live dashboard.

Companion demo for the tutorial "Continuous Evaluation and Monitoring of
Foundation Models". Paste any text, pick any model installed in Ollama, and
watch four metrics behave as time-indexed signals with adaptive thresholds and
change detection.

Run it with ./run.sh rather than `streamlit run app.py` directly so the working
directory stays this folder (data paths and Streamlit's import root stay stable).

Refresh strategy: while a stream is running the script re-runs itself on a timer
and drains the worker's queue. A full rerun is more predictable than partial
fragment updates when a background thread is producing data, and everything
expensive (the model list, the vendored ML imports) is cached, so the cost is
just a chart redraw.
"""

from __future__ import annotations

import time
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from monitorlab import compat, config, detectors, replay, scenarios
from monitorlab import stream as stream_module
from monitorlab.metrics import framing
from monitorlab.ollama_client import OllamaClient
from monitorlab.stream import StreamConfig, StreamRunner, TimeSeries

st.set_page_config(page_title="LLM Monitoring Demo", page_icon="~", layout="wide")

SEVERITY_COLOR = {"ok": "#22c55e", "warn": "#f59e0b", "alarm": "#ef4444"}

# Streamlit's native theme is fixed at launch (--theme.base). The in-app picker
# overlays CSS and switches the Plotly template so Light/Dark can change without
# restarting the server.
_DARK_THEME_CSS = """
<style>
    .stApp, [data-testid="stAppViewContainer"],
    [data-testid="stHeader"], [data-testid="stToolbar"] {
        background-color: #0e1117 !important;
        color: #fafafa !important;
    }
    [data-testid="stSidebar"], [data-testid="stSidebarContent"] {
        background-color: #262730 !important;
        color: #fafafa !important;
    }
    [data-testid="stSidebar"] * {
        color: inherit;
    }
    [data-testid="stMarkdownContainer"], p, span, label, .stCaption,
    [data-testid="stWidgetLabel"], [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"], [data-testid="stMetricDelta"] {
        color: #fafafa !important;
    }
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-baseweb="textarea"] > div,
    .stTextInput input, .stTextArea textarea, .stNumberInput input {
        background-color: #1a1c24 !important;
        color: #fafafa !important;
        border-color: #3d4450 !important;
    }
    [data-testid="stExpander"], [data-testid="stAlert"],
    [data-testid="stDataFrame"], [data-testid="stMetric"] {
        background-color: #1a1c24 !important;
        color: #fafafa !important;
    }
    hr { border-color: #3d4450 !important; }
    .stTabs [data-baseweb="tab-list"] button {
        color: #fafafa !important;
    }
</style>
"""


def apply_theme(theme: str) -> None:
    """Apply the selected UI theme for this rerun."""
    if theme == "dark":
        st.markdown(_DARK_THEME_CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------- cached setup
@st.cache_resource(show_spinner=False)
def get_client() -> OllamaClient:
    return OllamaClient()


@st.cache_data(ttl=30, show_spinner=False)
def get_models() -> tuple[list[dict], str]:
    """Installed Ollama models. Cached so the rerun loop does not re-query."""
    client = get_client()
    try:
        models = client.list_models()
    except Exception as exc:  # noqa: BLE001
        return [], str(exc)
    return [
        {"name": m.name, "label": m.label, "speed": m.speed, "size_gb": m.size_gb}
        for m in models
    ], ""


@st.cache_data(show_spinner=False)
def kg_cache_size() -> int:
    cache = config.DEMO_DIR / ".kg_cache"
    return len(list(cache.glob("*.json"))) if cache.is_dir() else 0


def init_state() -> None:
    state = st.session_state
    state.setdefault("series", TimeSeries())
    state.setdefault("runner", None)
    state.setdefault("writer", None)
    state.setdefault("mode", "live")
    state.setdefault("replay_raw", [])       # as recorded
    state.setdefault("replay_records", [])   # verdicts recomputed with current settings
    state.setdefault("replay_index", 0)
    state.setdefault("replay_playing", False)
    state.setdefault("selected_step", 0)
    state.setdefault("last_error", "")
    state.setdefault("ui_theme", "light")
    state.setdefault("perturb_ids", [])
    state.setdefault("perturb_next_id", 0)


# ----------------------------------------------------------------------- charts
def metric_figure(series: TimeSeries, metric: str, theme: str = "light") -> go.Figure:
    """One metric: value, adaptive band, alarm markers, perturbation lines."""
    spec = config.METRICS.get(metric, {})
    data = series.series(metric)
    events = series.events(metric)
    colour = spec.get("color", "#22d3ee")
    dark = theme == "dark"
    band_fill = "rgba(148,163,184,0.28)" if dark else "rgba(148,163,184,0.16)"
    centre_line = "rgba(226,232,240,0.55)" if dark else "rgba(148,163,184,0.7)"

    figure = go.Figure()

    # The band is drawn only over the steps where it is ready, which is a
    # contiguous tail, so a single filled region is safe.
    band_steps = [s for s, lo in zip(data["steps"], data["lower"]) if lo is not None]
    if band_steps:
        lower = [lo for lo in data["lower"] if lo is not None]
        upper = [up for up in data["upper"] if up is not None]
        figure.add_trace(
            go.Scatter(
                x=band_steps, y=upper, mode="lines", line=dict(width=0),
                hoverinfo="skip", showlegend=False, name="upper",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=band_steps, y=lower, mode="lines", line=dict(width=0),
                fill="tonexty", fillcolor=band_fill,
                hoverinfo="skip", name="adaptive band",
            )
        )

    figure.add_trace(
        go.Scatter(
            x=data["steps"], y=data["center"], mode="lines",
            line=dict(color=centre_line, width=1, dash="dot"),
            name="EWMA centre", hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=data["steps"], y=data["values"], mode="lines+markers",
            line=dict(color=colour, width=2), marker=dict(size=5),
            name=spec.get("short", metric),
            hovertemplate="step %{x}<br>%{y:.4f}<extra></extra>",
        )
    )

    if events["warn_x"]:
        figure.add_trace(
            go.Scatter(
                x=events["warn_x"], y=events["warn_y"], mode="markers",
                marker=dict(size=11, color=SEVERITY_COLOR["warn"], symbol="circle-open",
                            line=dict(width=2)),
                name="warn",
            )
        )
    if events["alarm_x"]:
        figure.add_trace(
            go.Scatter(
                x=events["alarm_x"], y=events["alarm_y"], mode="markers",
                marker=dict(size=13, color=SEVERITY_COLOR["alarm"], symbol="x",
                            line=dict(width=2)),
                name="alarm",
            )
        )

    # Mark where the input changed, so the room can compare when the detector
    # fired against when the world actually moved.
    for step in series.perturbation_steps():
        figure.add_vline(
            x=step, line=dict(color="#e879f9", width=2, dash="dash"),
            annotation_text="perturbation", annotation_position="top",
            annotation_font_color="#e879f9",
        )

    figure.update_layout(
        height=270,
        margin=dict(l=10, r=10, t=30, b=10),
        title=dict(text=spec.get("label", metric), font=dict(size=14)),
        xaxis_title="step",
        showlegend=False,
        hovermode="x unified",
        template="plotly_dark" if dark else "plotly_white",
    )
    return figure


# ------------------------------------------------------------------- UI: sidebar
def sidebar() -> dict:
    client = get_client()
    models, model_error = get_models()

    with st.sidebar:
        st.markdown("### Monitoring demo")

        online = bool(models)
        if online:
            st.caption(f"Ollama {client.version()} - {len(models)} models - KG cache {kg_cache_size()}")
        else:
            st.error(f"Ollama unreachable.\n\n{model_error}")
            st.caption("Replay mode still works without Ollama.")

        theme = st.radio(
            "Theme", ["light", "dark"],
            format_func=lambda t: "Light" if t == "light" else "Dark",
            key="ui_theme", horizontal=True,
        )
        apply_theme(theme)

        mode = st.radio(
            "Mode", ["live", "replay"],
            format_func=lambda m: "Live stream" if m == "live" else "Replay a recording",
            key="mode", horizontal=True,
        )

        if mode == "replay":
            controls = replay_controls()
            st.divider()
            return {
                "mode": "replay",
                "theme": theme,
                "detector_params": detector_settings(),
                **controls,
            }

        st.divider()

        # ---- content
        st.markdown("**1. Content to analyse**")
        samples = scenarios.load_samples()
        options = ["(custom)"] + [s.id for s in samples]
        labels = {"(custom)": "Custom text", **{s.id: s.label for s in samples}}
        chosen = st.selectbox(
            "Preset", options, index=1 if samples else 0,
            format_func=lambda k: labels.get(k, k), key="sample_choice",
        )
        preset = scenarios.sample_by_id(chosen) if chosen != "(custom)" else None
        if preset:
            st.caption(f"Targets {preset.targets}. {preset.expect}")

        text = st.text_area(
            "Text", value=preset.text if preset else "", height=170,
            key=f"text_{chosen}",
            placeholder="Paste any article, report or paragraph to monitor.",
        )

        # ---- model
        st.markdown("**2. Model**")
        names = [m["name"] for m in models]
        hint = {m["name"]: m for m in models}
        model = st.selectbox(
            "Ollama model", names or ["(none installed)"],
            format_func=lambda n: hint.get(n, {}).get("label", n),
            key="model_choice", disabled=not online,
        )
        if model in hint and hint[model]["speed"] == "slow":
            st.warning(
                f"{model} is {hint[model]['size_gb']:.0f} GB. Fine for a recording, "
                "slow for a live loop."
            )

        # ---- metrics
        st.markdown("**3. Metrics**")
        metrics = []
        for key, spec in config.METRICS.items():
            default = key in config.DEFAULT_METRICS
            extra = f" (+{framing.probe_calls()} calls/step)" if key == "bias" else ""
            if st.checkbox(spec["short"] + extra, value=default, key=f"metric_{key}",
                           help=spec["description"]):
                metrics.append(key)

        # ---- stream shape
        st.markdown("**4. Stream**")
        steps = st.slider("Steps", 4, 120, 24, key="steps")
        resamples = st.slider(
            "Resamples per step", 1, 5, 3, key="resamples",
            help="Drift needs at least 2. Each costs one extra call per step.",
            disabled="drift" not in metrics,
        )
        temperature = st.slider("Temperature", 0.0, 1.5, 0.2, 0.1, key="temperature")
        interval = st.slider("Pause between steps (s)", 0.0, 10.0, 0.0, 0.5, key="interval")
        use_safeguard = st.checkbox(
            "Use the schema system prompt", value=True, key="safeguard",
            help="The parent project's SYSTEM_PROMPT, which asks for DBpedia-style triples.",
        )
        framing_every = st.slider(
            "Framing cadence (every N steps)", 1, 10, 4, key="framing_every",
            disabled="bias" not in metrics,
        )

        # ---- perturbations
        st.markdown("**5. Perturbations**")
        st.caption(
            "Each change applies from its step onward; later changes of the same kind "
            "override earlier ones. Edits made while a run is going take effect from "
            "the next step."
        )
        perturbations = [
            perturbation_editor(pid, index, steps, names, model, samples, chosen, labels)
            for index, pid in enumerate(list(st.session_state.perturb_ids))
        ]
        st.button("Add a perturbation", on_click=add_perturbation, args=(steps,),
                  width="stretch")

        # ---- detectors
        detector_params = detector_settings()

        st.divider()
        record = st.checkbox(
            "Record this run", value=True, key="record",
            help="Writes JSONL to data/runs/ so it can be replayed without Ollama.",
        )

        runner: StreamRunner | None = st.session_state.runner
        busy = bool(runner and runner.running)
        c1, c2, c3 = st.columns(3)
        start = c1.button("Start", type="primary", disabled=busy or not online or not text.strip(),
                          width="stretch")
        stop = c2.button("Stop", disabled=not busy, width="stretch")
        clear = c3.button("Clear", disabled=busy, width="stretch")

    return {
        "mode": "live", "theme": theme, "text": text, "model": model, "metrics": metrics,
        "steps": steps, "resamples": resamples, "temperature": temperature,
        "interval": interval, "use_safeguard": use_safeguard, "framing_every": framing_every,
        "perturbations": perturbations, "detector_params": detector_params,
        "record": record, "start": start, "stop": stop, "clear": clear,
        "sample_id": chosen, "preset": preset, "online": online,
    }


PERTURBATION_KINDS = [k for k in scenarios.PERTURBATIONS if k != "none"]


def add_perturbation(steps: int) -> None:
    state = st.session_state
    pid = state.perturb_next_id
    state.perturb_next_id += 1
    ats = [state.get(f"perturb_at_{p}", 0) for p in state.perturb_ids]
    state[f"perturb_at_{pid}"] = min(steps, max(2, max(ats) + 4 if ats else steps // 2))
    state.perturb_ids.append(pid)


def remove_perturbation(pid: int) -> None:
    st.session_state.perturb_ids.remove(pid)


def perturbation_editor(
    pid: int, index: int, steps: int, names: list[str], model: str,
    samples, chosen: str, labels: dict[str, str],
) -> scenarios.Perturbation:
    """One entry in the perturbation list. Widget keys are per-entry so edits survive reruns."""
    state = st.session_state
    at_key = f"perturb_at_{pid}"
    upper = max(3, steps)
    # Streamlit refuses a stored slider value outside the range, and Steps can shrink.
    if state.get(at_key, 2) > upper:
        state[at_key] = upper

    with st.container(border=True):
        kind = st.selectbox(
            f"Change {index + 1}", PERTURBATION_KINDS,
            format_func=lambda k: scenarios.PERTURBATIONS[k]["label"],
            key=f"perturb_kind_{pid}",
        )
        st.caption(scenarios.PERTURBATIONS[kind]["blurb"])
        perturbation = scenarios.Perturbation(kind=kind)
        perturbation.at_step = st.slider("At step", 2, upper, key=at_key)
        if kind == "switch_model":
            others = [n for n in names if n != model] or names
            perturbation.to_model = st.selectbox("Switch to", others, key=f"perturb_model_{pid}")
        elif kind == "raise_temperature":
            perturbation.to_temperature = st.slider(
                "New temperature", 0.0, 2.0, 1.2, 0.1, key=f"perturb_temp_{pid}"
            )
        elif kind == "switch_corpus":
            others = [s.id for s in samples if s.id != chosen]
            perturbation.to_sample_id = st.selectbox(
                "Switch to", others or [chosen],
                format_func=lambda k: labels.get(k, k), key=f"perturb_sample_{pid}",
            )
        st.button("Remove", key=f"perturb_remove_{pid}", on_click=remove_perturbation,
                  args=(pid,))
    return perturbation


def detector_settings() -> dict:
    """The detector sliders, shared by both modes.

    Available in replay too, because replay recomputes verdicts from the recorded
    values rather than reading back stored ones. That makes these sliders live on
    a recorded incident, which is the cheapest way to show the false-alarm /
    detection-delay trade-off without waiting on a model.
    """
    with st.expander("Detector settings"):
        st.caption(
            "Page-Hinkley and CUSUM thresholds are in noise widths, not metric "
            "units, so one set of numbers serves grounding in [0, 1] and latency "
            "in seconds. Defaults come from `scripts/calibrate.py`, which scores "
            "them against the recorded runs. Worth moving live: a high EWMA "
            "lambda follows a slow drift instead of flagging it."
        )
        base = detectors.default_params()
        return {
            "lam": st.slider("EWMA lambda", 0.05, 0.9, base["lam"], 0.05, key="d_lam"),
            "k": st.slider("Band width (k sigma)", 1.0, 5.0, base["k"], 0.5, key="d_k"),
            "warmup": st.slider("Warm-up steps", 3, 20, base["warmup"], key="d_warmup"),
            "z_threshold": st.slider("Robust z threshold", 2.0, 6.0,
                                     base["z_threshold"], 0.5, key="d_z"),
            "ph_delta": st.slider("Page-Hinkley delta (noise widths)", 0.1, 2.0,
                                  base["ph_delta"], 0.05, key="d_phd"),
            "ph_lambda": st.slider("Page-Hinkley lambda (noise widths)", 1.0, 8.0,
                                   base["ph_lambda"], 0.5, key="d_phl"),
            "cusum_slack": st.slider("CUSUM slack (noise widths)", 0.1, 2.0,
                                     base["cusum_slack"], 0.05, key="d_cuss"),
            "cusum_threshold": st.slider("CUSUM threshold (noise widths)", 1.0, 10.0,
                                         base["cusum_threshold"], 0.5, key="d_cus"),
        }


def replay_controls() -> dict:
    runs = replay.list_runs()
    if not runs:
        st.info(
            "No recordings yet. Run a live stream with **Record this run** "
            "enabled and it will appear here."
        )
        return {"records": [], "play": False, "reset": False, "speed": 0.0, "meta": None}

    paths = [p for p, _ in runs]
    metas = {p: m for p, m in runs}
    chosen = st.selectbox(
        "Recording", paths, format_func=lambda p: metas[p].label, key="replay_file"
    )
    meta = metas[chosen]
    st.caption(f"{meta.text_preview[:110]}...")

    speed = st.select_slider(
        "Speed", options=[0.0, 1.0, 2.0, 4.0, 8.0],
        value=4.0, format_func=lambda v: "instant" if v == 0 else f"{v:g}x", key="replay_speed",
    )
    c1, c2 = st.columns(2)
    play = c1.button("Play", type="primary", width="stretch")
    reset = c2.button("Reset", width="stretch")

    if st.session_state.get("replay_loaded") != str(chosen):
        _, records = replay.load_run(Path(chosen))
        st.session_state.replay_raw = records
        st.session_state.replay_loaded = str(chosen)
        st.session_state.replay_index = 0
        st.session_state.series = TimeSeries()

    return {"play": play, "reset": reset, "speed": speed, "meta": meta}


# ---------------------------------------------------------------- UI: main panel
def header(controls: dict) -> None:
    series: TimeSeries = st.session_state.series
    runner: StreamRunner | None = st.session_state.runner
    latest = series.latest()

    st.markdown("#### Continuous evaluation of a local LLM")

    cols = st.columns([2, 1, 1, 1, 2])
    if controls["mode"] == "live":
        cols[0].metric("Model", latest.model if latest else controls.get("model", "-"))
        total = controls.get("steps", 0)
    else:
        meta = controls.get("meta")
        cols[0].metric("Model", meta.model if meta else "-")
        total = len(controls.get("records") or [])
    cols[1].metric("Step", f"{len(series)}/{total}" if total else str(len(series)))

    alarms = series.alarm_log()
    cols[2].metric("Alarms", sum(1 for a in alarms if a["severity"] == "alarm"))
    cols[3].metric("Warnings", sum(1 for a in alarms if a["severity"] == "warn"))

    if runner and runner.running:
        cols[4].success("streaming")
    elif runner and runner.error:
        cols[4].error(runner.error[:80])
    elif st.session_state.replay_playing:
        cols[4].info("replaying")
    elif len(series):
        cols[4].info("idle")

    if latest and latest.errors:
        st.warning(" | ".join(latest.errors[:3]))


def kpi_row(metrics: list[str]) -> None:
    series: TimeSeries = st.session_state.series
    latest = series.latest()
    if latest is None:
        return

    cols = st.columns(len(metrics) or 1)
    for column, metric in zip(cols, metrics):
        spec = config.METRICS.get(metric, {})
        sample = latest.metrics.get(metric)
        with column:
            if sample is None:
                st.metric(spec.get("short", metric), "-")
                continue

            history = series.series(metric)["values"]
            delta = None
            if len(history) > 1:
                delta = round(history[-1] - history[-2], 4)

            fmt = f"{sample.value:.2f}s" if metric == "latency" else f"{sample.value:.3f}"
            st.metric(spec.get("short", metric), fmt, delta=delta,
                      delta_color="off" if metric == "latency" else "normal")
            if sample.severity != "ok":
                colour = SEVERITY_COLOR[sample.severity]
                st.markdown(
                    f"<span style='color:{colour};font-size:0.78rem'>"
                    f"{sample.severity.upper()}: {'; '.join(sample.reasons)}</span>",
                    unsafe_allow_html=True,
                )


def charts(metrics: list[str], mode: str = "live", theme: str = "light") -> None:
    series: TimeSeries = st.session_state.series
    if not len(series):
        if mode == "replay":
            st.info(
                "Nothing replayed yet. Pick a recording and press **Play**.\n\n"
                "Speed *instant* jumps straight to the end, which is what you want "
                "when you need the interesting part and not the warm-up."
            )
        else:
            st.info(
                "Nothing streamed yet. Pick a preset and a model in the sidebar, then "
                "press **Start**.\n\nFor a first run, keep the defaults and choose the "
                "*Grounded* preset: it establishes the baseline every other preset is "
                "compared against."
            )
        return

    plottable = [m for m in metrics if series.series(m)["steps"]]
    for left, right in zip(plottable[::2], plottable[1::2] + [None]):
        c1, c2 = st.columns(2)
        c1.plotly_chart(metric_figure(series, left, theme), width="stretch")
        if right:
            c2.plotly_chart(metric_figure(series, right, theme), width="stretch")


def alarm_table() -> None:
    series: TimeSeries = st.session_state.series
    rows = series.alarm_log()
    st.markdown("##### Alarm log")
    if not rows:
        st.caption("Quiet so far. On a clean run that is the correct outcome.")
        return
    st.dataframe(rows, width="stretch", hide_index=True, height=min(280, 40 + 35 * len(rows)))


def drilldown() -> None:
    series: TimeSeries = st.session_state.series
    if not len(series):
        return

    st.markdown("##### Step detail")
    steps = series.steps()
    # No explicit key on purpose. The widget's identity includes its options, so
    # while a stream is running (options growing every step) it resets to the
    # newest step, and once the stream stops a manual selection sticks. An
    # explicit key would instead retain a step number that may no longer exist.
    step = st.select_slider("Step", options=steps, value=steps[-1])
    record = next((r for r in series.records if r.step == step), None)
    if record is None:
        return

    meta = f"model `{record.model}` - temperature {record.temperature} - "
    meta += "schema prompt on" if record.safeguard else "**schema prompt off**"
    if record.plan_notes:
        meta += f" - perturbed: {', '.join(record.plan_notes)}"
    st.caption(meta)

    tabs = st.tabs(["Triples", "Grounding", "Drift", "Framing", "Raw response"])

    with tabs[0]:
        structure = record.detail.get("structure", {})
        if structure:
            c = st.columns(4)
            c[0].metric("ICR", f"{structure.get('icr', 0):.3f}")
            c[1].metric("IPR", f"{structure.get('ipr', 0):.3f}")
            c[2].metric("CI", f"{structure.get('ci', 0):.3f}")
            c[3].metric("Triples", structure.get("triple_count", 0))
        if record.triples:
            st.dataframe(
                [{"subject": s, "predicate": p, "object": o} for s, p, o in record.triples],
                width="stretch", hide_index=True, height=260,
            )
        else:
            st.warning("No parsable triples at this step.")

    with tabs[1]:
        detail = record.detail.get("grounding_detail")
        scores = record.detail.get("grounding", {})
        if not detail:
            st.caption("Grounding was not enabled for this run.")
        else:
            c = st.columns(4)
            c[0].metric("NER consistency", f"{scores.get('ner_consistency_score', 0):.3f}")
            c[1].metric("Rule / KG", f"{scores.get('rule_definition_score', 0):.3f}")
            c[2].metric("Grounding", f"{scores.get('hallucination', 0):.3f}")
            c[3].metric("Entities compared", detail.get("support", 0),
                        help="How many entities the F1 was computed over. A score "
                             "from one or two entities can only take a few values.")
            if detail.get("thin_support"):
                st.warning(detail.get("note") or "Too few entities for a stable score.")
            ungrounded = detail.get("ungrounded") or []
            if ungrounded:
                st.markdown(
                    "**Asserted but not found in the source text** - the entities "
                    "driving the score down:"
                )
                st.dataframe(
                    [{"entity": u.get("entity_text"), "claimed type": u.get("rule_label")}
                     for u in ungrounded],
                    width="stretch", hide_index=True,
                )
            else:
                st.caption("Every asserted entity was corroborated in the source text.")
            with st.expander("All extracted entities"):
                c1, c2 = st.columns(2)
                c1.caption("From the model's triples")
                c1.dataframe(detail.get("triple_entities") or [], width="stretch",
                             hide_index=True, height=200)
                c2.caption("From spaCy, on the source text")
                c2.dataframe(detail.get("spacy_entities") or [], width="stretch",
                             hide_index=True, height=200)

    with tabs[2]:
        detail = record.detail.get("drift_detail")
        scores = record.detail.get("drift", {})
        if not detail:
            st.caption("Drift was not enabled for this run.")
        else:
            c = st.columns(3)
            c[0].metric("Drift", f"{scores.get('drift', 0):.3f}")
            c[1].metric("Structural (Jaccard)", f"{scores.get('jaccard_distance', 0):.3f}")
            c[2].metric("Lexical", f"{scores.get('lexical_dispersion', 0):.3f}")
            st.caption(
                f"{detail.get('stable_count', 0)} triples appeared in every resample, "
                f"{detail.get('volatile_count', 0)} did not. A large volatile set on a "
                "clear text means the model is guessing."
            )
            c1, c2 = st.columns(2)
            c1.markdown("**Stable across resamples**")
            c1.code("\n".join(detail.get("stable") or ["(none)"]), language=None)
            c2.markdown("**Volatile**")
            c2.code("\n".join(detail.get("volatile") or ["(none)"]), language=None)

    with tabs[3]:
        detail = record.detail.get("framing_detail")
        scores = record.detail.get("framing", {})
        if not scores:
            st.caption("Framing was not enabled for this run.")
        elif scores.get("stale"):
            st.caption(
                f"Not measured at this step (cadence). Last value: "
                f"{scores.get('bias', 0):.3f}"
            )
        elif detail:
            c = st.columns(2)
            c[0].metric("Bias (sentiment spread)", f"{scores.get('sentiment_spread', 0):.3f}")
            c[1].metric(
                "Production silhouette bias",
                f"{scores.get('production_silhouette_bias', 0):.3f}",
                help="The parent project's (sensitivity + sentiment) / 2. Shown as a "
                     "diagnostic only -- it is not the monitored signal.",
            )
            st.caption(
                f"Dimension: {scores.get('category')}. Same question, one response per "
                "group. The spread is the monitored signal because it grows with the "
                "disparity; the silhouette measures how cleanly the groups cluster, "
                "and on these presets it can score a contested article and a "
                "sports report nearly the same."
            )
            per_prompt = detail.get("per_prompt") or {}
            if per_prompt:
                rows = []
                for name, values in per_prompt.items():
                    row = {"prompt": name, "spread": values.get("spread")}
                    row.update(values.get("sentiments_by_group") or {})
                    rows.append(row)
                st.markdown("**Sentiment by group**")
                st.dataframe(rows, width="stretch", hide_index=True)
            st.dataframe(
                [{"prompt": r["prompt"], "group": r["group"], "sentiment": r["sentiment"],
                  "response": (r["response"] or "")[:300]}
                 for r in (detail.get("responses") or [])],
                width="stretch", hide_index=True, height=240,
            )

    with tabs[4]:
        st.caption("Exactly what the model returned, before parsing.")
        st.code(record.raw_response or "(empty)", language=None)


def footer(controls: dict) -> None:
    series: TimeSeries = st.session_state.series
    with st.expander("What this is computing, and where the code comes from"):
        st.markdown(
            "Three of the four metrics use the same production scoring code as "
            "MonitorLLM, vendored under `monitorlab/vendor/` so this folder runs "
            "alone — no Reflex, torch or Postgres."
        )
        st.dataframe(compat.provenance(), width="stretch", hide_index=True)
        for key, spec in config.METRICS.items():
            st.markdown(f"**{spec['label']}** - {spec['description']}")

    if len(series):
        rows = []
        for record in series.records:
            row = {"step": record.step, "model": record.model,
                   "temperature": record.temperature, "safeguard": record.safeguard}
            for name, sample in record.metrics.items():
                row[name] = round(sample.value, 5)
                row[f"{name}_severity"] = sample.severity
            rows.append(row)
        header_line = ",".join(rows[0].keys())
        body = "\n".join(",".join(str(v) for v in r.values()) for r in rows)
        st.download_button(
            "Download metrics as CSV", f"{header_line}\n{body}",
            file_name=f"monitoring-{int(time.time())}.csv", mime="text/csv",
        )


# ------------------------------------------------------------------- run control
def start_stream(controls: dict) -> None:
    if not controls["metrics"]:
        st.session_state.last_error = "Select at least one metric."
        return

    st.session_state.series = TimeSeries()
    st.session_state.last_error = ""

    settings = StreamConfig(
        model=controls["model"],
        text=controls["text"],
        steps=controls["steps"],
        resamples=controls["resamples"],
        temperature=controls["temperature"],
        interval=controls["interval"],
        metrics=controls["metrics"],
        framing_every=controls["framing_every"],
        use_safeguard=controls["use_safeguard"],
        perturbations=controls["perturbations"],
        detector_params=controls["detector_params"],
    )

    writer = None
    if controls["record"]:
        meta = replay.RunMeta(
            model=settings.model,
            sample_id=controls["sample_id"],
            text_preview=settings.text[:220],
            steps=settings.steps,
            resamples=settings.resamples,
            temperature=settings.temperature,
            metrics=list(settings.metrics),
            perturbation=scenarios.describe_schedule(controls["perturbations"]),
        )
        writer = replay.RunWriter(replay.new_run_path(), meta)
    st.session_state.writer = writer

    runner = StreamRunner(get_client(), settings)
    st.session_state.runner = runner
    # The writer is appended to from the worker thread, which is safe: it only
    # touches its own file handle, never Streamlit state.
    runner.start(on_record=writer.append if writer else None)


def pump_live() -> bool:
    """Move finished records from the worker into the UI. True while running."""
    runner: StreamRunner | None = st.session_state.runner
    if runner is None:
        return False
    for record in runner.drain():
        st.session_state.series.append(record)
    if not runner.running:
        writer = st.session_state.writer
        if writer is not None:
            writer.close()
            st.session_state.writer = None
        return False
    return True


def pump_replay(controls: dict) -> bool:
    """Advance the replay by one step. True while more remain."""
    records = controls.get("records") or []
    index = st.session_state.replay_index
    if index >= len(records):
        st.session_state.replay_playing = False
        return False
    st.session_state.series.append(records[index])
    st.session_state.replay_index = index + 1
    return True


def main() -> None:
    init_state()
    controls = sidebar()

    if controls["mode"] == "replay":
        # Verdicts are recomputed from the recorded values on every rerun rather
        # than read back from the file, so the detector sliders apply to
        # recordings too and a recording never carries stale thresholds.
        raw = st.session_state.get("replay_raw") or []
        fingerprint = (st.session_state.get("replay_loaded"),
                       tuple(sorted(controls["detector_params"].items())))
        if st.session_state.get("replay_fingerprint") != fingerprint:
            st.session_state.replay_records = stream_module.replay_detectors(
                raw, **controls["detector_params"]
            )
            st.session_state.replay_fingerprint = fingerprint
            st.session_state.replay_index = 0
            st.session_state.series = TimeSeries()
        controls["records"] = st.session_state.replay_records

    if controls["mode"] == "live":
        if controls.get("clear"):
            st.session_state.series = TimeSeries()
            st.session_state.runner = None
        if controls.get("stop"):
            runner: StreamRunner | None = st.session_state.runner
            if runner:
                runner.stop()
        if controls.get("start"):
            start_stream(controls)
        runner = st.session_state.runner
        if runner is not None and runner.running:
            runner.set_perturbations(controls["perturbations"])
        busy = pump_live()
    else:
        if controls.get("reset"):
            st.session_state.replay_index = 0
            st.session_state.series = TimeSeries()
            st.session_state.replay_playing = False
        if controls.get("play"):
            st.session_state.replay_playing = True
        busy = False
        if st.session_state.replay_playing:
            if controls.get("speed", 0) == 0:
                # Instant: drain the whole recording in one pass.
                while pump_replay(controls):
                    pass
                st.session_state.replay_playing = False
            else:
                busy = pump_replay(controls)

    if st.session_state.last_error:
        st.error(st.session_state.last_error)

    header(controls)
    metrics = controls.get("metrics") or list(config.DEFAULT_METRICS)
    latest = st.session_state.series.latest()
    if latest:
        # In replay the enabled set comes from the recording, not the sidebar.
        metrics = [m for m in config.METRICS if m in latest.metrics] or metrics
    kpi_row(metrics)
    charts(metrics, controls["mode"], controls.get("theme", "light"))

    left, right = st.columns([1, 1])
    with left:
        alarm_table()
    with right:
        if controls["mode"] == "live" and controls.get("preset"):
            st.markdown("##### What to watch for")
            st.info(controls["preset"].teaching_note)

    drilldown()
    footer(controls)

    if busy:
        # Re-run to pick up the next record. The sleep sets the refresh rate and
        # keeps the rerun loop from spinning the CPU.
        speed = controls.get("speed", 1.0) if controls["mode"] == "replay" else 1.0
        time.sleep(0.35 if controls["mode"] == "replay" and speed >= 4 else 0.9)
        st.rerun()


if __name__ == "__main__":
    main()
