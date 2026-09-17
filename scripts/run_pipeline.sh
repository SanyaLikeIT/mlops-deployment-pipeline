#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin:/snap/bin:${PATH:-}"
printf '[%s] Pipeline starting\n' "$(date --iso-8601=seconds)"
trap 'status=$?; printf "[%s] Pipeline finished (exit %s)\n" "$(date --iso-8601=seconds)" "$status"' EXIT

if [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
fi
for command in python dvc docker flock; do
    if ! command -v "$command" >/dev/null 2>&1; then
        echo "Missing command: $command. Create .venv and follow README.md." >&2
        exit 1
    fi
done
exec 9> .pipeline.lock
if ! flock -n 9; then
    echo "Another pipeline run is active; skipping this invocation."
    exit 0
fi
docker compose version >/dev/null
if ! docker info >/dev/null; then
    echo "Docker is unavailable or access is denied. Check Docker setup in README.md." >&2
    exit 1
fi
if [[ ! -f data/raw/wine.csv ]]; then
    python code/datasets/create_raw_data.py
fi
dvc repro -f
# Recreate services so the API loads the latest model at startup.
docker compose -f code/deployment/docker-compose.yml up -d --build --force-recreate --wait
echo "Web app: http://localhost:8501 | API docs: http://localhost:8000/docs"
