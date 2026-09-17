#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if ! command -v crontab >/dev/null 2>&1; then
    echo "crontab is missing. Install the Ubuntu cron package first." >&2
    exit 1
fi
if [[ "$ROOT" == *$'\n'* ]]; then
    echo "Cron does not support repository paths containing newlines." >&2
    exit 1
fi
# Quote absolute paths for the POSIX shell used by cron.
quote() { printf "'%s'" "${1//\'/\'\\\'\'}"; }
command="/bin/bash $(quote "$ROOT/scripts/run_pipeline.sh") >> $(quote "$ROOT/pipeline.log") 2>&1"
command="${command//%/\\%}"
entry="*/5 * * * * $command"
temporary="$(mktemp)"
trap 'rm -f "$temporary" "$temporary.error"' EXIT
if ! LC_ALL=C crontab -l > "$temporary" 2> "$temporary.error"; then
    if ! grep -q 'no crontab for' "$temporary.error"; then
        cat "$temporary.error" >&2
        exit 1
    fi
fi
if ! grep -Fqx -- "$entry" "$temporary"; then
    printf '\n%s\n' "$entry" >> "$temporary"
    crontab "$temporary"
fi
printf 'Installed schedule:\n%s\n' "$entry"
