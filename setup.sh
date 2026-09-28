#!/usr/bin/env bash
#
# One-time setup for the monitoring demo.
#
# Creates .venv, installs dependencies, fetches the spaCy English model, and
# verifies that the vendored ML modules import cleanly. Safe to re-run; nothing
# outside this folder is written to.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
DEMO_DIR="$(pwd)"
VENV="$DEMO_DIR/.venv"

say()  { printf '\n\033[1;36m==> %s\033[0m\n' "$1"; }
warn() { printf '\033[1;33m    warning: %s\033[0m\n' "$1"; }
die()  { printf '\n\033[1;31merror: %s\033[0m\n' "$1" >&2; exit 1; }

# ---------------------------------------------------------------- interpreter
# spaCy needs <3.15 and Streamlit needs >=3.10. 3.14 is excluded on purpose:
# thinc (which spaCy pins to <8.4) publishes no cp314 wheels, so spaCy cannot
# be installed there without a source build.
say "Locating a suitable Python (3.10-3.13)"
PYENV_HOME="${PYENV_ROOT:-$HOME/.pyenv}"
PYTHON=""
for candidate in \
    $(ls -d "$PYENV_HOME"/versions/3.13.*/bin/python3 2>/dev/null | sort -rV) \
    $(ls -d "$PYENV_HOME"/versions/3.1[012].*/bin/python3 2>/dev/null | sort -rV) \
    python3.13 python3.12 python3.11 python3.10 python3
do
    command -v "$candidate" >/dev/null 2>&1 || continue
    ver="$("$candidate" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null)" || continue
    case "$ver" in
        3.10|3.11|3.12|3.13) PYTHON="$candidate"; break ;;
    esac
done

if [ -z "$PYTHON" ]; then
    die "No Python 3.10-3.13 found. Install one, e.g.:
    pyenv install 3.13.11
  Python 3.14 will not work: spaCy's pinned thinc has no cp314 wheels."
fi
printf '    using %s (%s)\n' "$PYTHON" "$("$PYTHON" -V 2>&1)"

# ------------------------------------------------------------------- the venv
if [ ! -x "$VENV/bin/python" ]; then
    say "Creating virtualenv at .venv"
    "$PYTHON" -m venv "$VENV"
else
    say "Reusing existing .venv"
fi
PY="$VENV/bin/python"

say "Installing dependencies"
"$PY" -m pip install --quiet --upgrade pip
"$PY" -m pip install --quiet -r "$DEMO_DIR/requirements.txt"

# The model wheel is pure Python; letting spaCy choose the version keeps it
# compatible with whichever spaCy the resolver picked above.
say "Installing the spaCy English model (en_core_web_sm)"
if "$PY" -c 'import spacy; spacy.load("en_core_web_sm")' >/dev/null 2>&1; then
    printf '    already present\n'
else
    "$PY" -m spacy download en_core_web_sm
fi

# ----------------------------------------------------------------- DBpedia cache
# DBpedia lookups are filled on demand (or via scripts/warm_cache.py).
say "Checking DBpedia cache"
mkdir -p "$DEMO_DIR/.kg_cache"
n="$(find "$DEMO_DIR/.kg_cache" -name '*.json' 2>/dev/null | wc -l | tr -d ' ')"
if [ "$n" -gt 0 ]; then
    printf '    %s cached entries ready\n' "$n"
else
    warn ".kg_cache is empty; first grounding scores will hit DBpedia"
fi

# ------------------------------------------------------------------ self-check
say "Verifying vendored ML modules import cleanly"
"$PY" "$DEMO_DIR/scripts/selfcheck.py"

# ---------------------------------------------------------------------- Ollama
say "Checking Ollama"
if curl -sf --max-time 3 http://localhost:11434/api/version >/dev/null 2>&1; then
    printf '    reachable: %s\n' "$(curl -s http://localhost:11434/api/version)"
    say "Warming the recommended demo models"
    "$PY" "$DEMO_DIR/scripts/warm_cache.py" --warm-models
else
    warn "Ollama is not answering on http://localhost:11434 -- start it with 'ollama serve'.
    The dashboard still runs in Replay mode without it."
fi

cat <<'EOF'

Setup complete. Start the dashboard with:

    ./run.sh

EOF
