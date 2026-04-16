#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: $0 <version>"
  echo "Example: $0 0.2.0"
  exit 1
fi

VERSION="$1"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# portable in-place sed (macOS + Linux)
_sed_i() {
  if sed --version >/dev/null 2>&1; then
    sed -i "$@"
  else
    sed -i '' "$@"
  fi
}

_sed_i "s/^version = \".*\"/version = \"$VERSION\"/" "$REPO_ROOT/pyproject.toml"
_sed_i "s/^__version__ = \".*\"/__version__ = \"$VERSION\"/" "$REPO_ROOT/src/parallect/__init__.py"

echo "parallect bumped to $VERSION"
echo ""
echo "Next steps:"
echo "  1. Update CHANGELOG.md"
echo "  2. git commit -am \"release: v$VERSION\""
echo "  3. git tag v$VERSION"
echo "  4. git push origin main --tags"
