#!/usr/bin/env bash
# ==============================================================================
# run_all.sh  –  Two-phase flow benchmarks
#
# Runs all benchmarks in 98_twophaseflow_benchmarks/.
# Each benchmark: create geometry → simulate with mpirun.
#
# MPI ranks chosen per domain geometry (z always fixed to 1 rank):
#   Quasi-1D slab  (x-periodic, y-walls, z-thin) : np=2
#   2-D domain     (x/y-periodic or walls)        : np=4
#   True 3-D       (all three directions active)  : np=8
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

banner "Two-phase flow benchmarks  |  NL=${NL}"

# ─── 01 Floating Droplet (3-D) ────────────────────────────────────────────────
# True 3-D spherical droplet: all directions active  →  np=8
banner "01  Floating Droplet  (np=8)"
pushd "${SCRIPT_DIR}/01_floating_droplet" > /dev/null
python3 create_input_geometry_floating_droplet_3d.py ${NL}
INIT=$(ls -t *_init.gsd | head -1)
mpirun -np 8 python3 run_floating_droplet.py ${NL} "${INIT}"
mpirun -np 8 python3 run_floating_droplet_TV.py ${NL} "${INIT}"
popd > /dev/null

# ─── 01 Floating Droplet FS (3-D, free-surface) ───────────────────────────────
# Same 3-D geometry with free-surface model  →  np=8
banner "01  Floating Droplet FS  (np=8)"
pushd "${SCRIPT_DIR}/01_floating_droplet_fs" > /dev/null
python3 create_input_geometry_floating_droplet_3d.py ${NL}
INIT=$(ls -t *_init.gsd | head -1)
mpirun -np 8 python3 run_floating_droplet.py ${NL} "${INIT}"
popd > /dev/null

# ─── 02 Static Droplet ────────────────────────────────────────────────────────
# 2-D (z-thin) square domain with periodic x/y  →  np=4
banner "02  Static Droplet  (np=4)"
pushd "${SCRIPT_DIR}/02_static_droplet" > /dev/null
python3 create_input_geometry.py ${NL}
INIT=$(ls -t *_init.gsd | head -1)
mpirun -np 4 python3 run_static_droplet.py ${NL} "${INIT}"
mpirun -np 4 python3 run_static_droplet_TV.py ${NL} "${INIT}"
popd > /dev/null

# ─── 03 Layered Couette ───────────────────────────────────────────────────────
# Quasi-1D slab: x-periodic, y-walls, z one-particle-thick  →  np=2
banner "03  Layered Couette  (np=2)"
pushd "${SCRIPT_DIR}/03_layered_couette" > /dev/null
python3 create_input_geometry.py ${NL}
INIT=$(ls -t *_init.gsd | head -1)
mpirun -np 2 python3 run_layered_couette.py ${NL} "${INIT}"
mpirun -np 2 python3 run_layered_couette_TV.py ${NL} "${INIT}"
popd > /dev/null

# ─── 04 Rising Bubble ─────────────────────────────────────────────────────────
# 2-D tall domain: NL × 4·NL fluid cells, z one-particle-thick  →  np=4
banner "04  Rising Bubble  (np=4)"
pushd "${SCRIPT_DIR}/04_rising_bubble" > /dev/null
python3 create_input_geometry.py ${NL}
INIT=$(ls -t *_init.gsd | head -1)
mpirun -np 4 python3 run_rising_bubble.py ${NL} "${INIT}"
popd > /dev/null

# ─── 05 Rayleigh–Taylor ───────────────────────────────────────────────────────
# 2-D tall domain: NL × 4·NL fluid cells, z one-particle-thick  →  np=4
banner "05  Rayleigh-Taylor  (np=4)"
pushd "${SCRIPT_DIR}/05_rayleigh_taylor" > /dev/null
python3 create_input_geometry.py ${NL}
INIT=$(ls -t *_init.gsd | head -1)
mpirun -np 4 python3 run_rayleigh_taylor.py ${NL} "${INIT}"
popd > /dev/null

# ─── 06 Rayleigh–Taylor Boxed ─────────────────────────────────────────────────
# 2-D domain with solid walls on all sides  →  np=4
banner "06  Rayleigh-Taylor Boxed  (np=4)"
pushd "${SCRIPT_DIR}/06_rayleigh_taylor_boxed" > /dev/null
python3 create_input_geometry.py ${NL}
INIT=$(ls -t *_init.gsd | head -1)
mpirun -np 4 python3 run_rayleigh_taylor_boxed.py ${NL} "${INIT}"
popd > /dev/null

# ─── 06 Sessile Droplet Shear ─────────────────────────────────────────────────
# 3-D domain with shear flow and wetting  →  np=8
banner "06  Sessile Droplet Shear  (np=8)"
pushd "${SCRIPT_DIR}/06_sessile_droplet_shear" > /dev/null
python3 create_input_geometry.py ${NL}
INIT=$(ls -t *_init.gsd | head -1)
mpirun -np 8 python3 run_sessile_droplet_shear.py ${NL} "${INIT}"
popd > /dev/null

# ─── 07 Capillary Rise ────────────────────────────────────────────────────────
# 3-D round tube in square box: 7 contact angles, wetting → non-wetting  →  np=4
banner "07  Capillary Rise  (np=4, 7 configs)"
pushd "${SCRIPT_DIR}/07_capillary_rise" > /dev/null
bash run_all_configs.sh ${NL}
popd > /dev/null

banner "All two-phase flow benchmarks complete."
