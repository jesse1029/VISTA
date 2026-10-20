#!/usr/bin/env bash
# Reproduce the principal comparison of the manuscript. Usage: bash reproduce.sh DATA_DIR [OUT_DIR]
set -euo pipefail
DATA="$1"; HERE="$(cd "$(dirname "$0")" && pwd)"; OUT="${2:-$HERE/reproduced}"
DESIS="$DATA/desis_2021.csv"; PRISMA="$DATA/prisma_2021.csv"
echo "85ac21e38fb990a75714f90be0d2ced643275adf4e46570f326e765709c48ccc  $DESIS" | sha256sum -c -
echo "0c6ef3dff0e5eff1c3e9ebb349494465d4459ee189c2aeb089af4e7622822647  $PRISMA" | sha256sum -c -
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
ORIG="$HERE/vendor/original_common_utility.py"; S="$HERE/scripts"
if [ -e "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then
  echo "Output directory $OUT is not empty; choose a new one so old results cannot mix in." >&2; exit 1
fi
mkdir -p "$OUT/base" "$OUT/ext"
pids=()
python "$S/run_vista_comparison.py" --original "$ORIG" --desis "$DESIS" --prisma "$PRISMA" --out "$OUT/base" & pids+=($!)
for folds in 0,1 2,3 4; do
  python "$S/run_additional_harmonizers.py" --original "$ORIG" --desis "$DESIS" --prisma "$PRISMA" \
    --out "$OUT/ext" --folds "$folds" --no-aggregate & pids+=($!)
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if [ "$status" -ne 0 ]; then echo "A comparison worker failed; stopping before aggregation." >&2; exit 1; fi
python - "$OUT/ext" <<'PY'
import sys, pandas as pd
from pathlib import Path
d = Path(sys.argv[1])
df = pd.concat([pd.read_csv(d / f"fold_{k}_additional.csv") for k in range(5)]).sort_values("row_index")
assert len(df) == 16581 and df.row_index.nunique() == 16581
df.to_csv(d / "heldout_additional.csv", index=False)
PY
python "$S/crossed_scale_analysis.py" "$ORIG" "$DESIS" "$PRISMA" "$OUT/base" "$OUT/base/crossed_scale.json" >/dev/null
python "$S/controlled_ablation.py" "$ORIG" "$DESIS" "$PRISMA" "$OUT/base" "$OUT/base/controlled_ablation.json" >/dev/null
python "$S/reference_anchor.py" "$ORIG" "$DESIS" "$PRISMA" "$OUT/base" "$OUT/reference_anchor.json" >/dev/null
python "$S/paper_numbers.py" "$OUT" "$DESIS" "$OUT/paper_numbers.json"
python "$S/supp_extra_numbers.py" "$OUT" "$OUT/base/split_manifest.csv" "$DESIS" "$OUT/supp_extra.json"
echo "Done: $OUT"
