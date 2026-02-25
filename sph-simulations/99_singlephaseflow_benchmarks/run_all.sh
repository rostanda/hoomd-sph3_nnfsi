#!/usr/bin/env bash
# ==============================================================================
# run_all.sh  –  Single-phase flow benchmarks
#
# Runs all benchmarks in 99_singlephaseflow_benchmarks/.
# Each benchmark: create geometry → simulate with mpirun.
#
# MPI ranks chosen per domain geometry (z always fixed to 1 rank):
#   Quasi-1D slab  (x-periodic, y-walls, z-thin) : np=2
#   2-D domain     (x/y-periodic or walls)        : np=4
#   3-D tube       (x-periodic, z active)         : np=4
#
# NOTE: Benchmarks 06_spherepackings* require pre-existing packing data
#       (input*.txt + raw geometry files) and are not included here.
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

banner "Single-phase flow benchmarks  |  NL=${NL}"

# ─── 01 Hagen–Poiseuille tube flow ────────────────────────────────────────────
# 3-D circular tube: x-periodic, y/z confined by wall particles  →  np=4
banner "01  Poiseuille Tube  (np=4)"
pushd "${SCRIPT_DIR}/01_poiseuille" > /dev/null
python3 create_input_geometries_tube.py ${NL}
INIT=$(ls -t *_init.gsd | head -1)
mpirun -np 4 python3 run_tube.py ${NL} "${INIT}"
mpirun -np 4 python3 run_tube_TV.py ${NL} "${INIT}"
popd > /dev/null

# ─── 02 Parallel Plates ───────────────────────────────────────────────────────
# Quasi-1D slab: x-periodic, y-walls, z one-particle-thick  →  np=2
banner "02  Parallel Plates  (np=2)"
pushd "${SCRIPT_DIR}/02_parallel_plates" > /dev/null
python3 create_input_geometries_parallel_plates.py ${NL}
INIT=$(ls -t *_init.gsd | head -1)
mpirun -np 2 python3 run_parallel_plates.py ${NL} "${INIT}"
mpirun -np 2 python3 run_parallel_plates_TV.py ${NL} "${INIT}"
popd > /dev/null

# ─── 03 Couette Flow ──────────────────────────────────────────────────────────
# Quasi-1D slab: x-periodic, y-walls (moving), z one-particle-thick  →  np=2
banner "03  Couette Flow  (np=2)"
pushd "${SCRIPT_DIR}/03_couette_flow" > /dev/null
python3 create_input_geometries_couette_flow.py ${NL}
INIT=$(ls -t *_init.gsd | head -1)
mpirun -np 2 python3 run_couette_flow.py ${NL} "${INIT}"
mpirun -np 2 python3 run_couette_flow_TV.py ${NL} "${INIT}"
popd > /dev/null

# ─── 04 Channel Flow ──────────────────────────────────────────────────────────
# 2-D channel: x-periodic, y-walls, z one-particle-thick  →  np=4
banner "04  Channel Flow  (np=4)"
pushd "${SCRIPT_DIR}/04_channel_flow" > /dev/null
python3 create_input_geometries_channel_flow.py ${NL}
INIT=$(ls -t *_init.gsd | head -1)
mpirun -np 4 python3 run_channel_flow.py ${NL} "${INIT}"
mpirun -np 4 python3 run_channel_flow_TV.py ${NL} "${INIT}"
popd > /dev/null

# ─── 05 Lid-driven Cavity ─────────────────────────────────────────────────────
# 2-D square cavity: all solid walls, moving lid  →  np=4
# Uses OptionParser: -n resolution, -S initgsd, -R reynolds
banner "05  Lid-driven Cavity Re=100  (np=4)"
pushd "${SCRIPT_DIR}/05_liddriven_cavity" > /dev/null
python3 create_input_geometries_ldc.py -n ${NL} -R 100
INIT=$(ls -t *_init.gsd | head -1)
mpirun -np 4 python3 run_ldc.py -n ${NL} -S "${INIT}" -R 100
mpirun -np 4 python3 run_ldc_tv.py -n ${NL} -S "${INIT}" -R 100
popd > /dev/null

# ─── 07 Adami Cylinder 1 ──────────────────────────────────────────────────────
# 2-D duct with cylinder: num_length hardcoded to 100 in create_domain.py
# domain_decomposition=(None,None,1)  →  np=4
banner "07  Adami Cylinder 1  (np=4, NL_CYL=100)"
pushd "${SCRIPT_DIR}/07_adami_cylinder1" > /dev/null
python3 create_domain.py
INIT=$(ls -t *_init.gsd | head -1)
mpirun -np 4 python3 run_adami_cylinder1.py 100 "${INIT}"
popd > /dev/null

# ─── 08 Adami Cylinder 2 ──────────────────────────────────────────────────────
# Wider duct with cylinder: num_length hardcoded to 96 in create_domain.py
# domain_decomposition=(None,None,1)  →  np=4
banner "08  Adami Cylinder 2  (np=4, NL_CYL=96)"
pushd "${SCRIPT_DIR}/08_adami_cylinder2" > /dev/null
python3 create_domain.py
INIT=$(ls -t *_init.gsd | head -1)
mpirun -np 4 python3 run_adami_cylinder2.py 96 "${INIT}"
popd > /dev/null

# ─── 09 Drag Cylinder ─────────────────────────────────────────────────────────
# Same geometry as 08, with solid-force logging for drag  →  np=4
banner "09  Drag Cylinder  (np=4, NL_DRAG=96)"
pushd "${SCRIPT_DIR}/09_drag_cylinder" > /dev/null
python3 create_domain.py
INIT=$(ls -t *_init.gsd | head -1)
mpirun -np 4 python3 run_adami_cylinder2.py 96 "${INIT}"
popd > /dev/null

# ─── 10 Hydrostatic ───────────────────────────────────────────────────────────
# Quasi-1D slab: gravity, no flow, checks pressure profile  →  np=2
banner "10  Hydrostatic  (np=2)"
pushd "${SCRIPT_DIR}/10_hydrostatic" > /dev/null
python3 create_input_geometry.py ${NL}
INIT=$(ls -t *_init.gsd | head -1)
mpirun -np 2 python3 run_hydrostatic.py ${NL} "${INIT}"
popd > /dev/null

# ─── 11 Taylor–Green Vortex ───────────────────────────────────────────────────
# 2-D periodic box: decaying vortices, checks viscous decay  →  np=4
banner "11  Taylor-Green Vortex  (np=4)"
pushd "${SCRIPT_DIR}/11_taylor_green" > /dev/null
python3 create_input_geometry.py ${NL}
INIT=$(ls -t *_init.gsd | head -1)
mpirun -np 4 python3 run_taylor_green.py ${NL} "${INIT}"
popd > /dev/null

# ─── 12 Stokes' Second Problem ────────────────────────────────────────────────
# Quasi-1D: oscillating plate, Stokes boundary layer  →  np=2
banner "12  Stokes Second Problem  (np=2)"
pushd "${SCRIPT_DIR}/12_stokes_second" > /dev/null
python3 create_input_geometry.py ${NL}
INIT=$(ls -t *_init.gsd | head -1)
mpirun -np 2 python3 run_stokes_second.py ${NL} "${INIT}"
popd > /dev/null

# ─── 13 BCC Sphere Packing — Darcy–Forchheimer sweep ─────────────────────────
# Full 3-D: no z constraint  →  2×2×2 decomposition  →  np=8
# Runs 22 body-force values; see 13_bcc_spherepacking_permeability/run_all_re.sh
# for details.  NL is ignored here (geometry is fixed at 100³).
banner "13  BCC Sphere Packing  permeability sweep  (np=8)"
pushd "${SCRIPT_DIR}/13_bcc_spherepacking_permeability" > /dev/null
bash run_all_re.sh
popd > /dev/null

banner "All single-phase flow benchmarks complete."
