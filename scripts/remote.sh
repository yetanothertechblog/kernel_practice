#!/usr/bin/env bash
# Develop on the Mac, build and run on the GPU server.
#   scripts/remote.sh sync                      push code to the server
#   scripts/remote.sh make [target]             push + build
#   scripts/remote.sh run ./bin/vecadd 1024     push + build + run (output streams back here)
#   scripts/remote.sh pull                      copy the server's results/ back here
#   scripts/remote.sh ssh                       open a shell in the remote project dir
#
# Server: put an ssh target (ssh-config Host or user@host) in .remote, or set KP_REMOTE.
# Remote dir: ~/kernel_practice, or set KP_RDIR.
set -eo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE="${KP_REMOTE:-$(cat "$ROOT/.remote" 2>/dev/null || true)}"
RDIR="${KP_RDIR:-kernel_practice}"
CUDA_ENV='export PATH=/usr/local/cuda/bin:$PATH;'

if [ -z "$REMOTE" ]; then
  echo "No server set. Run: echo 'user@host' > $ROOT/.remote" >&2
  exit 1
fi

push() {
  rsync -az --delete \
    --exclude bin/ --exclude results/ --exclude .git/ --exclude .remote \
    "$ROOT/" "$REMOTE:$RDIR/" >&2
}
on_server() { ssh "$REMOTE" "$CUDA_ENV cd $RDIR && $1"; }

cmd="$1"
[ $# -gt 0 ] && shift
case "$cmd" in
  sync) push ;;
  make) push; on_server "make $*" ;;
  run)
    # Build only the binary being run, so a half-written kernel elsewhere doesn't block it.
    target=""
    for a in "$@"; do case "$a" in *bin/*) target="bin/${a##*bin/}"; break ;; esac; done
    push; on_server "make -s $target >&2 && $(printf '%q ' "$@")" ;;
  pull) rsync -az "$REMOTE:$RDIR/results/" "$ROOT/results/" ;;
  ssh)  ssh -t "$REMOTE" "$CUDA_ENV cd $RDIR && exec \$SHELL -l" ;;
  *)    sed -n '2,7p' "$0"; exit 1 ;;
esac
