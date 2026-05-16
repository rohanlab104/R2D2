#!/usr/bin/env bash
# Typo helper: people often type "num" — forward to the real NIM script.
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_nim_nano.sh" "$@"
