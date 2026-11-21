#!/usr/bin/env bash
# Run one configuration R times to check run-to-run noise.
# Usage: scripts/repeat.sh R bin/<kernel> [args...] > results/<name>.csv
R=$1; BIN=$2; shift 2
for r in $(seq 1 "$R"); do
  "$BIN" "$@" | grep '^CSV,' | sed "s/^CSV,/$r,/"
done
