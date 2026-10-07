#!/usr/bin/env bash
set -euo pipefail
repository="$(cd "$(dirname "$0")/.." && pwd)"
shono_check_dir="$(mktemp -d)"
trap 'rm -rf "$shono_check_dir"' EXIT
git clone --quiet --no-local "$repository" "$shono_check_dir/shono"
cd "$shono_check_dir/shono"
unset UV_PROJECT_ENVIRONMENT
./verify
