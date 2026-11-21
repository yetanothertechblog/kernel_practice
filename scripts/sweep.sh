#!/usr/bin/env bash
# Power-of-two size sweep. Usage: scripts/sweep.sh bin/<kernel> [extra args] > results/<name>.csv
BIN=$1; shift
echo "n,threads,ms,gbps"
for e in $(seq 10 27); do
  "$BIN" $((1 << e)) "$@" | grep '^CSV,' | cut -d, -f2-
done
