#!/usr/bin/env bash
# Prove the README quickstart works from a clean clone: clone this repo into a
# throwaway directory and run ./verify there, with no local state to lean on.
# This is the ship-gate "fresh-environment rebuild from docs alone" check.
set -euo pipefail

here="$(cd "$(dirname "$0")/.." && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

echo "fresh clone: $here -> $tmp/shono"
git clone --quiet "$here" "$tmp/shono"
cd "$tmp/shono"
./verify
echo "fresh-clone rebuild OK"
