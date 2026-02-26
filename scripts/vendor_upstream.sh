#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/vendor_upstream.sh <commit-ish>

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <commit-ish>" >&2
  exit 1
fi

COMMIT="$1"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENDOR_DIR="$ROOT_DIR/cpp/vendor/leiden_communities_openmp"
TMP_DIR="$(mktemp -d)"

trap 'rm -rf "$TMP_DIR"' EXIT

git clone --depth 1 https://github.com/puzzlef/leiden-communities-openmp.git "$TMP_DIR/repo"
cd "$TMP_DIR/repo"
git fetch --depth 1 origin "$COMMIT"
git checkout "$COMMIT"

rm -rf "$VENDOR_DIR/upstream"
mkdir -p "$VENDOR_DIR/upstream"
cp -R . "$VENDOR_DIR/upstream"

echo "Vendored leiden-communities-openmp at $(git rev-parse HEAD)"
