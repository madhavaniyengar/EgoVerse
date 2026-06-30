#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRATCH="/scratch/madhavai"
VENV_DIR="$SCRATCH/emimic"
UV_CACHE="$SCRATCH/uv-cache"
WANDB_CACHE="$SCRATCH/wandb-cache"
LINK="$REPO_DIR/emimic"

echo "==> Node: $(hostname)"
echo "==> Repo: $REPO_DIR"
echo "==> Venv: $VENV_DIR"

# --- Scratch dirs ---
mkdir -p "$SCRATCH" "$UV_CACHE" "$WANDB_CACHE"

# --- Create venv if missing ---
if [ ! -d "$VENV_DIR" ]; then
    echo "==> Creating venv at $VENV_DIR"
    UV_PYTHON=/home/madhavai/.local/share/uv/python/cpython-3.11-linux-x86_64-gnu/bin/python3.11 \
        uv venv "$VENV_DIR"
else
    echo "==> Venv already exists at $VENV_DIR, skipping creation"
fi

# --- Symlink emimic -> scratch venv ---
if [ -L "$LINK" ]; then
    current_target="$(readlink "$LINK")"
    if [ "$current_target" != "$VENV_DIR" ]; then
        echo "==> Relinking $LINK -> $VENV_DIR (was $current_target)"
        ln -sfn "$VENV_DIR" "$LINK"
    else
        echo "==> Symlink already correct"
    fi
elif [ -d "$LINK" ]; then
    echo "==> $LINK is a real directory; renaming to emimic.bak then symlinking"
    mv "$LINK" "${LINK}.bak"
    ln -s "$VENV_DIR" "$LINK"
else
    echo "==> Creating symlink $LINK -> $VENV_DIR"
    ln -s "$VENV_DIR" "$LINK"
fi

# --- Install ---
export UV_CACHE_DIR="$UV_CACHE"
export VIRTUAL_ENV="$VENV_DIR"

echo "==> Installing requirements.txt"
uv pip install -r "$REPO_DIR/requirements.txt"

echo "==> Installing package in editable mode"
uv pip install -e "$REPO_DIR"

echo "==> Installing projectaria-tools (no deps, skips rerun-sdk)"
uv pip install "projectaria-tools==2.0.0" --no-deps

echo ""
echo "Done. Activate with:"
echo "  source $LINK/bin/activate"
echo ""
echo "Run training with:"
echo "  export WANDB_CACHE_DIR=$WANDB_CACHE"
echo "  export UV_CACHE_DIR=$UV_CACHE"
