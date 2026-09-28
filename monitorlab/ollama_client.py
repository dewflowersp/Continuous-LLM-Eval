"""A small, synchronous Ollama client.

The parent project's ``LLMService`` resolves a model's endpoint by querying
Postgres (``llm_service.py:36-51``), so the demo cannot use it without a
database. This is the same idea with the lookup removed: point it at a base URL
and call it.

Two deliberate differences from ``LLMService._call_ollama``:

* The token cap is sent as ``num_predict``. Ollama ignores ``max_tokens`` in
  ``options``, so the parent's cap silently has no effect; the demo needs a real
  cap to keep step durations predictable.
* Latency is measured and returned alongside the text, because here it is a
  monitored signal rather than a side effect.

Uses ``urllib`` so the demo needs no HTTP dependency of its own.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from . import config

# Ollama's `think` field is model-specific: gpt-oss wants low|medium|high;
# gemma4 wants a boolean (string "low" is ignored and it still thinks).
ThinkValue = str | bool


def is_reasoning_model(model: str) -> bool:
    name = model.lower()
    return any(tag in name for tag in config.REASONING_MODEL_TAGS)


def short_probe_think(model: str) -> ThinkValue | None:
    """`think` value for short probe calls (framing, warm) that should stay cheap."""
    name = model.lower()
    if "gpt-oss" in name:
        return "low"
    if "gemma4" in name:
        return False
    return None


class OllamaUnavailable(RuntimeError):
    """Raised when the Ollama server cannot be reached at all."""


@dataclass
class Generation:
    """One completion, with the timing the monitor cares about."""

    text: str
    latency: float
    model: str
    ok: bool = True
    error: str = ""
    eval_count: int = 0

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


@dataclass
class ModelInfo:
    name: str
    size_bytes: int = 0
    parameter_size: str = ""
    family: str = ""
    speed: str = field(default="unknown")

    @property
    def size_gb(self) -> float:
        return self.size_bytes / 1e9

    @property
    def label(self) -> str:
        bits = [self.name]
        if self.parameter_size:
            bits.append(self.parameter_size)
        if self.size_bytes:
            bits.append(f"{self.size_gb:.1f} GB")
        return "  ".join(bits)


class OllamaClient:
    def __init__(self, base_url: str | None = None, timeout: int | None = None):
        self.base_url = (base_url or config.OLLAMA_URL).rstrip("/")
        self.timeout = timeout or config.OLLAMA_TIMEOUT

    # ------------------------------------------------------------------ plumbing
    def _post(self, path: str, payload: dict, timeout: int | None = None) -> dict:
        request = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout or self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def _get(self, path: str, timeout: int = 5) -> dict:
        with urllib.request.urlopen(self.base_url + path, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    # -------------------------------------------------------------------- status
    def is_up(self) -> bool:
        try:
            self._get("/api/version")
            return True
        except Exception:  # noqa: BLE001
            return False

    def version(self) -> str:
        try:
            return str(self._get("/api/version").get("version", "unknown"))
        except Exception:  # noqa: BLE001
            return "unreachable"

    def list_models(self) -> list[ModelInfo]:
        """Installed models, fast ones first, then by size.

        This is what populates the model picker, so it reflects whatever is
        actually on the machine rather than a hardcoded list.
        """
        try:
            payload = self._get("/api/tags")
        except Exception as exc:  # noqa: BLE001
            raise OllamaUnavailable(
                f"Could not reach Ollama at {self.base_url}. Start it with "
                f"'ollama serve'. ({exc})"
            ) from exc

        models = []
        for entry in payload.get("models", []):
            name = entry.get("name") or entry.get("model") or ""
            if not name:
                continue
            details = entry.get("details") or {}
            models.append(
                ModelInfo(
                    name=name,
                    size_bytes=entry.get("size", 0) or 0,
                    parameter_size=details.get("parameter_size", "") or "",
                    family=details.get("family", "") or "",
                    speed=config.classify_model(name),
                )
            )

        rank = {"fast": 0, "unknown": 1, "slow": 2}
        models.sort(key=lambda m: (rank.get(m.speed, 1), m.size_bytes))
        return models

    # ---------------------------------------------------------------- generation
    def generate(
        self,
        model: str,
        prompt: str,
        system: str = "",
        temperature: float = 0.2,
        num_predict: int = 1024,
        seed: int | None = None,
        timeout: int | None = None,
        think: ThinkValue | None = None,
    ) -> Generation:
        """One non-streaming completion.

        Never raises for per-call failures: a failed generation is itself a
        monitoring signal, so it comes back as ``ok=False`` and the stream
        continues. Only a total inability to reach the server raises.

        ``think`` is for reasoning models. Shape depends on the family:
        ``gpt-oss`` wants ``low`` / ``medium`` / ``high`` (boolean false is
        ignored); ``gemma4`` wants a boolean. Leave it ``None`` for the
        model's default effort so thinking-time variation shows up in
        monitored latency. Use :func:`short_probe_think` for framing/warm.
        """
        # Default thinking shares num_predict with the visible answer. Raise
        # the floor so the trace can finish and still leave room for triples.
        if is_reasoning_model(model) and think is None and num_predict >= 256:
            num_predict = max(num_predict, config.REASONING_MIN_NUM_PREDICT)

        options: dict[str, object] = {
            "temperature": temperature,
            "num_predict": num_predict,
        }
        if seed is not None:
            options["seed"] = seed

        payload: dict[str, object] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
        if system:
            payload["system"] = system
        if think is not None:
            payload["think"] = think

        started = time.perf_counter()
        try:
            data = self._post("/api/generate", payload, timeout=timeout)
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", "replace")[:200]
            except Exception:  # noqa: BLE001
                pass
            return Generation("", time.perf_counter() - started, model, False, f"HTTP {exc.code}: {body}")
        except Exception as exc:  # noqa: BLE001
            return Generation("", time.perf_counter() - started, model, False, str(exc))

        text = (data.get("response") or "").strip()
        thinking = (data.get("thinking") or "").strip()
        eval_count = int(data.get("eval_count") or 0)
        done_reason = str(data.get("done_reason") or "")
        # Do not fall back to `thinking` as the answer: it is chain-of-thought,
        # not triples. Surface why the call looked "successful" but empty.
        error = ""
        if not text and thinking:
            error = (
                f"empty response after {eval_count} tokens "
                f"(done_reason={done_reason or 'unknown'}); "
                "model spent the budget on thinking. "
                f"Raise num_predict (reasoning floor is "
                f"{config.REASONING_MIN_NUM_PREDICT}) or disable thinking "
                "for short probes via short_probe_think()."
            )

        return Generation(
            text=text,
            latency=time.perf_counter() - started,
            model=model,
            ok=not error,
            error=error,
            eval_count=eval_count,
        )

    def generate_many(
        self,
        model: str,
        prompts: Sequence[str],
        system: str = "",
        temperature: float = 0.2,
        num_predict: int = 1024,
        max_workers: int = 4,
        think: ThinkValue | None = None,
    ) -> list[Generation]:
        """Several completions concurrently, returned in the order given.

        Ollama serialises work on a single model internally, so this mostly
        removes per-request overhead rather than giving true parallelism. It
        still measurably shortens a step that needs six sensitivity calls.
        """
        if not prompts:
            return []
        workers = max(1, min(max_workers, len(prompts)))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ollama") as pool:
            futures = [
                pool.submit(
                    self.generate,
                    model=model,
                    prompt=prompt,
                    system=system,
                    temperature=temperature,
                    num_predict=num_predict,
                    think=think,
                )
                for prompt in prompts
            ]
            return [future.result() for future in futures]

    def warm(self, models: Iterable[str]) -> dict[str, float]:
        """Force each model to load, so the first monitored step is not an outlier.

        A cold model pays its load time on the first token, which would show up
        as a latency spike and a false alarm in the first few steps.
        """
        timings: dict[str, float] = {}
        for name in models:
            # Short load probe: suppress reasoning so warm stays cheap.
            result = self.generate(
                name,
                "ok",
                temperature=0.0,
                num_predict=1,
                think=short_probe_think(name),
            )
            timings[name] = result.latency if result.ok else float("nan")
        return timings
