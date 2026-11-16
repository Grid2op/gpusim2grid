#!/usr/bin/env bash
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Type stubs of the compiled modules for editors and type checkers (Pylance / pyright read
# typings/, see [tool.pyright] in pyproject.toml): lightsim2grid's lightsim2grid_cpp, whose
# wheel ships none, and gpusim2grid's own _gpusim2grid. Generated from the installed
# binaries: run again after a rebuild or a lightsim2grid upgrade. typings/ is git-ignored.
#
#   tools/generate_stubs.sh [python]     # default: the python on PATH
set -eu
PY=${1:-python}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
"$PY" -c "import pybind11_stubgen" >/dev/null 2>&1 || {
  echo "pybind11-stubgen is missing: $PY -m pip install pybind11-stubgen" >&2; exit 1; }
rm -rf "$ROOT/typings"
for module in lightsim2grid.lightsim2grid_cpp gpusim2grid._gpusim2grid; do
  "$PY" -m pybind11_stubgen "$module" -o "$ROOT/typings" --ignore-all-errors
done
echo "stubs in $ROOT/typings:"
find "$ROOT/typings" -name '*.pyi' -printf '  %P\n'
