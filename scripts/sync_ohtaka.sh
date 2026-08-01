#!/usr/bin/env bash
set -euo pipefail
SOURCE="$(cd "$(dirname "$0")/.." && pwd)/"
DESTINATION="${1:?usage: sync_ohtaka.sh USER@HOST:/path/to/symfc-vasp/}"
rsync -a --delete --exclude '.pytest_cache' --exclude '*.egg-info' \
  "$SOURCE" "$DESTINATION"
