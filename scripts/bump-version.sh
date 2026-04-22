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

# Keep the legacy `parallect` compatibility shim in lockstep with the canonical
# `parallect-cli` distribution. Both its own version AND its `parallect-cli==`
# dependency pin must match the tag being released — the release workflow
# verifies this and fails fast if they drift.
SHIM_PYPROJECT="$REPO_ROOT/packages/parallect-legacy/pyproject.toml"
if [ -f "$SHIM_PYPROJECT" ]; then
  _sed_i "s/^version = \".*\"/version = \"$VERSION\"/" "$SHIM_PYPROJECT"
  _sed_i "s/\"parallect-cli==[^\"]*\"/\"parallect-cli==$VERSION\"/" "$SHIM_PYPROJECT"
  echo "parallect-cli bumped to $VERSION"
  echo "parallect (legacy shim) bumped to $VERSION (pins parallect-cli==$VERSION)"
else
  echo "parallect-cli bumped to $VERSION"
fi

echo ""
echo "Next steps:"
echo "  1. Update CHANGELOG.md"
echo "  2. git commit -am \"release: v$VERSION\""
echo "  3. git tag v$VERSION"
echo "  4. git push origin main --tags"
