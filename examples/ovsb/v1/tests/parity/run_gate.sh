#!/bin/bash
# Whole-recipe parity gate: dump this recipe, compare with the reference.
#
#   tests/parity/run_gate.sh SITE_FILE HARNESS_DIR [OUT_DIR]
#
# SITE_FILE    site file (see README.md and site.example.yaml)
# HARNESS_DIR  directory with parity.py and the reference dump (ref_*.json)
# OUT_DIR      scratch directory for new.json and the views
#              (default: $TMPDIR/ovsb_gate)
#
# Environment:
#   PYTHON        python of the training environment (default: python)
#   OVSB_MODULES  environment modules to load first, e.g. the FFmpeg module
#                 the codec chain needs (space separated)
#   THREADS       torch threads (default 4, the value of the reference dump)
#   REFERENCE     reference dump, if HARNESS_DIR holds none or several
#
# CPU only, one process, about 4 minutes and 14 GB of memory.
# Exit code 0 = parity, 1 = mismatch.
set -eo pipefail

if [ $# -lt 2 ]; then
  sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
fi
here=$(cd "$(dirname "$0")" && pwd)
site=$1
harness=$2
out=${3:-${TMPDIR:-/tmp}/ovsb_gate}
python=${PYTHON:-python}
reference=${REFERENCE:-}
if [ -z "$reference" ]; then
  set -- "$harness"/ref_*.json
  if [ $# -ne 1 ] || [ ! -f "$1" ]; then
    echo "run_gate.sh: need exactly one ref_*.json in $harness" \
         "(or set REFERENCE)" >&2
    exit 2
  fi
  reference=$1
fi

# a bare call per module: inside a pipe or $( ) the load would be lost
for module in $OVSB_MODULES; do
  if ! type ml > /dev/null 2>&1; then
    echo "run_gate.sh: no 'ml' command; load $OVSB_MODULES before" >&2
    exit 2
  fi
  ml "$module"
done

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p "$out"

nice -n 10 "$python" "$here/ovsb_adapter.py" dump \
  --harness "$harness/parity.py" --site "$site" \
  --out "$out/new.json" --tmp "$out" --threads "${THREADS:-4}"

"$python" "$here/ovsb_adapter.py" compare --harness "$harness/parity.py" \
  "$reference" "$out/new.json"
