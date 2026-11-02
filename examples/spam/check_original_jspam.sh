#!/bin/bash
# Compile and run the original, unmodified JSPAM Fortran on two collisions, and check the results.
# This shows the code compiles and runs the same as before, not that its physics is correct.
# Usage: check_original_jspam.sh [path-to-JSPAM-clone]      (default: JSPAM next to this repo)
# Needs git and Docker on this machine. Run it with bash (Git Bash on Windows).
# It reads the pinned commit from the clone, so changes in the clone's working folder or current
# branch don't affect it. Nothing is written to either repo. Exits non-zero if the check fails.
set -euo pipefail

commit=d6a61dbacc884e140b97eb4b191446bb4239a00f   # latest commit of jfwallin/JSPAM
image=ogdenm12/gtc-base-image:v2

here=$(dirname "$(realpath "$0")")
jspam=${1:-$here/../../../JSPAM}
if ! git -C "$jspam" cat-file -e "$commit^{commit}" 2>/dev/null; then
    echo "Commit $commit not found in JSPAM clone: $jspam" >&2
    exit 2
fi

# Runs in the container, with the Fortran source arriving on stdin as a tar file.
in_container=$(cat <<'EOF'
set -euo pipefail
mkdir /tmp/spam
tar -x -C /tmp/spam
cd /tmp/spam/fortran
for m in parameters_module io_module df_module setup_module integrator init_module basic_run; do
    gfortran -O3 -std=legacy -c "$m.f90"
done
gfortran -O3 -o basic_run ./*.o

# Only the last row (the second galaxy's center) is checked: disk particles start from unseeded
# random positions. The expected rows came from this image, built with these flags, on 2026-11-02.
check() {
    python - "$@" <<'PY'
import sys

import numpy as np

name, n_rows, *expected = sys.argv[1:]
rows = np.loadtxt("a.101")
assert rows.shape == (int(n_rows), 6), f"expected {n_rows} rows of 6 values, got {rows.shape}"
assert np.isfinite(rows).all(), "output contains NaN or infinity"
print(f"{name}: last row {rows[-1]}")
np.testing.assert_allclose(rows[-1], [float(v) for v in expected], rtol=1e-5, atol=1e-6)
PY
}

# Built-in test collision: 1000 + 500 particles, point-mass gravity.
./basic_run > run.log
check "Point mass" 1501     8.9999667e-01 -2.9296680e-03 -1.0663120e-03 -3.3333221e-03 -1.4648304e+00 -5.3315466e-01

# Galaxy Zoo model SDSS 587722984435351614: 1000 + 1000 particles. A model string selects the
# extended halo/disk/bulge potential (NBI in the Fortran), which the research uses.
model=$(awk '/SDSS_587722984435351614/ {print $1}' ../archive/data.txt)
./basic_run "$model" > run.log
check "Extended (NBI)" 2001     -9.9400701e+00 -4.5834789e+00 1.5438172e+01 -1.5399158e+00 -2.9788359e+00 4.6918210e+00

echo "OK: original JSPAM compiled, ran and matched the expected results"
EOF
)

# No path conversion: Git Bash on Windows would otherwise rewrite the /tmp paths in the script.
git -C "$jspam" archive "$commit" fortran archive/data.txt |
    MSYS_NO_PATHCONV=1 docker run --rm -i "$image" bash -c "$in_container"
