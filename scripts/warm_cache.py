#!/usr/bin/env python3
"""Pre-warm what the first live step would otherwise pay for.

Two costs get moved off the critical path:

Model load
    A cold Ollama model pays its load time on the first token. Left unwarmed
    that lands as a large latency spike in the first few steps, which both
    poisons the adaptive band's warm-up and produces a false alarm in front of
    the audience.

DBpedia lookups
    The grounding metric asks DBpedia about each entity over SPARQL. Cached on
    disk, that is free; uncached it is a few hundred milliseconds per entity,
    and unreachable if the venue network is hostile.

On cache keys: ``KGClient`` hashes ``"sync:" + url`` on its synchronous path and
``"async:" + url`` on its aiohttp path, and the two produce different keys for
the same question. The parent project's existing cache turns out to be entirely
sync-keyed, and the demo deliberately omits aiohttp (see requirements.txt), so
warmed entries and read entries always agree.

Usage:
    warm_cache.py                      # warm DBpedia for every preset
    warm_cache.py --warm-models        # load the recommended models
    warm_cache.py --model llama3.2:3b  # also warm the exact triple surfaces
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from monitorlab import compat, config, scenarios  # noqa: E402
from monitorlab.ollama_client import OllamaClient  # noqa: E402

GREEN, YELLOW, DIM, RESET = "\033[32m", "\033[33m", "\033[2m", "\033[0m"


def spacy_surfaces(texts: list[str]) -> set[str]:
    """Entity surfaces spaCy finds in the preset texts.

    A good approximation of what the grounding metric will look up, available
    without calling an LLM at all: the entities in the triples are largely the
    entities in the source text.
    """
    settings = compat.settings()
    nlp = getattr(settings, "NLP_MODEL", None)
    if nlp is None:
        return set()
    surfaces: set[str] = set()
    for text in texts:
        for entity in nlp(text).ents:
            cleaned = entity.text.strip()
            if len(cleaned) > 2:
                surfaces.add(cleaned)
    return surfaces


def triple_surfaces(model: str, texts: list[str]) -> set[str]:
    """Exact surfaces, by asking the model for triples the way the demo will."""
    client = OllamaClient()
    if not client.is_up():
        print(f"  {YELLOW}Ollama unreachable; skipping the exact pass{RESET}")
        return set()

    settings = compat.settings()
    parser = compat.triple_parser()
    checker = compat.ner_checker()

    surfaces: set[str] = set()
    for i, text in enumerate(texts, 1):
        result = client.generate(
            model=model,
            prompt=text,
            system=settings.SYSTEM_PROMPT,
            temperature=0.2,
            num_predict=900,
        )
        if not result.ok or result.is_empty:
            print(f"  {YELLOW}sample {i}: no usable response{RESET}")
            continue
        triples = parser.parse_triples_from_text(result.text)
        for entity in checker.extract_entities_from_triplets(triples):
            if entity.text and len(entity.text.strip()) > 2:
                surfaces.add(entity.text.strip())
        print(f"  sample {i}: {len(triples)} triples, {result.latency:.1f}s")
    return surfaces


def warm_dbpedia(surfaces: set[str]) -> None:
    """Resolve each surface once, populating demo/.kg_cache."""
    client = compat.kg_client_class()(cache_dir=str(config.DEMO_DIR / ".kg_cache"))

    resolved = unknown = 0
    for surface in sorted(surfaces):
        try:
            types = client.get_types(surface)
        except Exception as exc:  # noqa: BLE001
            print(f"  {YELLOW}{surface}: {exc}{RESET}")
            continue
        if types:
            resolved += 1
            print(f"  {GREEN}ok{RESET}   {surface} {DIM}-> {types[0].rsplit('/', 1)[-1]}{RESET}")
        else:
            unknown += 1
            print(f"  {DIM}--   {surface} (no DBpedia type){RESET}")

    print(f"\n  {resolved} resolved, {unknown} with no type, {len(surfaces)} total")
    if unknown and not resolved:
        print(
            f"  {YELLOW}Nothing resolved at all -- DBpedia is probably unreachable.\n"
            f"  The demo still runs; grounding scores will just be lower.{RESET}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--warm-models", action="store_true", help="load the recommended models")
    ap.add_argument("--model", default="", help="also warm exact triple surfaces using this model")
    ap.add_argument("--skip-dbpedia", action="store_true", help="only warm models")
    args = ap.parse_args()

    samples = scenarios.load_samples()
    if not samples:
        print(f"{YELLOW}No presets found at {config.SAMPLES_FILE}{RESET}")
        return
    texts = [s.text for s in samples if s.text]

    if args.warm_models:
        client = OllamaClient()
        installed = {m.name for m in client.list_models()} if client.is_up() else set()
        wanted = [m for m in config.FAST_MODELS if m in installed]
        if not wanted:
            print(f"{YELLOW}None of the recommended fast models are installed.{RESET}")
            print(f"{DIM}  suggested: ollama pull llama3.2:3b{RESET}")
        else:
            print(f"Warming {len(wanted)} model(s): {', '.join(wanted)}")
            for name, seconds in client.warm(wanted).items():
                print(f"  {GREEN}ok{RESET}   {name} {DIM}loaded in {seconds:.1f}s{RESET}")

    if args.skip_dbpedia:
        return

    print("\nCollecting entity surfaces from the presets")
    surfaces = spacy_surfaces(texts)
    print(f"  spaCy found {len(surfaces)} distinct surfaces")
    if args.model:
        extra = triple_surfaces(args.model, texts)
        print(f"  triples added {len(extra - surfaces)} more")
        surfaces |= extra

    if surfaces:
        print("\nWarming DBpedia")
        warm_dbpedia(surfaces)


if __name__ == "__main__":
    main()
