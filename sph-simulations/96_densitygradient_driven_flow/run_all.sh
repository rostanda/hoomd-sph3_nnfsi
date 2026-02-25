#!/usr/bin/env bash
# ==============================================================================
# run_all.sh  –  Density-gradient-driven flow benchmarks
#
# Runs all benchmarks in 96_densitygradient_driven_flow/.
# Each benchmark: create geometry → simulate with mpirun.
#
# MPI ranks chosen per domain geometry:
#   Quasi-1D slab  (x-periodic, y-walls, z-thin) : np=2
#   2-D cavity / tall domain                      : np=4
#
# Usage:  bash run_all.sh [num_length]
#   num_length : particles across reference length  (default: 20)
# ==============================================================================
set -euo pipefail

NL=${1:-20}    # particles across reference length
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

banner() {
    echo
    echo "══════════════════════════════════════════════════════════════"
    printf "  %-58s\n" "$*"
    echo "══════════════════════════════════════════════════════════════"
}

banner "Density-gradient benchmarks  |  NL=${NL}"

# ─── 01 Thermal Couette ───────────────────────────────────────────────────────
# Quasi-1D slab: x-periodic, y-walls, z one-particle-thick.
# domain_decomposition=(None,None,1)  →  only x/y split possible  →  np=2
banner "01  Thermal Couette  (np=2)"
pushd "${SCRIPT_DIR}/01_thermal_couette" > /dev/null
python3 create_input_geometry.py ${NL}
INIT=$(ls -t *_init.gsd | head -1)
mpirun -np 2 python3 run_thermal_couette.py ${NL} "${INIT}"
popd > /dev/null

# ─── 02 Heated Cavity ─────────────────────────────────────────────────────────
# 2-D square cavity (Ra=1000): NL×NL fluid cells, z one-particle-thick.
# domain_decomposition=(None,None,1)  →  2×2×1 decomposition  →  np=4
banner "02  Heated Cavity  (np=4)"
pushd "${SCRIPT_DIR}/02_heated_cavity" > /dev/null
python3 create_input_geometry.py ${NL}
INIT=$(ls -t *_init.gsd | head -1)
mpirun -np 4 python3 run_heated_cavity.py ${NL} "${INIT}"
popd > /dev/null

# ─── 03 Boussinesq Rayleigh–Taylor ────────────────────────────────────────────
# 2-D tall domain: NL × 4·NL fluid cells, z one-particle-thick.
# domain_decomposition=(None,None,1)  →  2×2×1 decomposition  →  np=4
banner "03  Boussinesq Rayleigh-Taylor  (np=4)"
pushd "${SCRIPT_DIR}/03_boussinesq_rayleigh_taylor" > /dev/null
python3 create_input_geometry.py ${NL}
INIT=$(ls -t *_init.gsd | head -1)
mpirun -np 4 python3 run_boussinesq_rt.py ${NL} "${INIT}"
popd > /dev/null

banner "All density-gradient benchmarks complete."
