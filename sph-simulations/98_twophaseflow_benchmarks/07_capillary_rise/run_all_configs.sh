#!/usr/bin/env bash
# ==============================================================================
# run_all_configs.sh  –  Capillary rise: sweep all wetting/non-wetting configs
#
# Creates the 3-D capillary geometry once, then runs one SPH simulation per
# contact-angle case defined in capillary_rise_params.txt.
#
# PHYSICAL SETUP
# --------------
#   Tube    : round (circular cross-section), inner radius R_cap = 1 mm
#   Domain  : square cross-section, 4×12×4 R_cap, x/z periodic, y solid walls
#   Fluids  : liquid W (ρ=1000, μ=0.1), gas N (ρ=100, μ=0.001), σ=0.01 N/m
#   Gravity : g = 9.81 m/s² in −y direction
#
# CONTACT ANGLE SWEEP
# -------------------
#   case 0 :  θ = 30°   → h_Jurin ≈ +1.77 mm  (strong wetting, rise)
#   case 1 :  θ = 45°   → h_Jurin ≈ +1.44 mm  (moderate wetting, rise)
#   case 2 :  θ = 60°   → h_Jurin ≈ +1.02 mm  (weak wetting, rise)
#   case 3 :  θ = 90°   → h_Jurin =   0.00 mm  (neutral, no rise)
#   case 4 :  θ = 120°  → h_Jurin ≈ −1.02 mm  (weak non-wetting, depression)
#   case 5 :  θ = 135°  → h_Jurin ≈ −1.44 mm  (moderate non-wetting)
#   case 6 :  θ = 150°  → h_Jurin ≈ −1.77 mm  (strong non-wetting, depression)
#
# VALIDATION
#   At steady state:  h_meas = y_meniscus_inside − y_reservoir_outside ≈ h_Jurin
#   Results are collected in capillary_rise_summary.dat (one line per case).
#   Plot h_meas vs cos(θ) — should be a straight line through the origin.
#
# MPI: np=4  (2×2×1 decomposition of the quasi-2D-tall 3D domain)
#
# Usage:  bash run_all_configs.sh [num_length] [steps]
#   num_length : particles across R_cap     (default: 10)
#   steps      : simulation steps per case  (default: 50001)
# ==============================================================================
set -euo pipefail

NL=${1:-10}
STEPS=${2:-50001}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

banner() {
    echo
    echo "══════════════════════════════════════════════════════════════"
    printf "  %-58s\n" "$*"
    echo "══════════════════════════════════════════════════════════════"
}

banner "Capillary rise sweep  |  NL=${NL}  steps=${STEPS}"

pushd "${SCRIPT_DIR}" > /dev/null

# ─── Step 1: create geometry (once) ──────────────────────────────────────────
banner "Create 3-D capillary geometry (NL=${NL})"
if [[ ! -f caprise_*_init.gsd ]]; then
    python3 create_capillary_geometry.py ${NL}
else
    echo "  caprise_*_init.gsd already exists – skipping geometry creation."
fi
INIT=$(ls -t caprise_*_init.gsd | head -1)
echo "  Using: ${INIT}"

# ─── Step 2: run all cases ───────────────────────────────────────────────────
for CASE_ID in 0 1 2 3 4 5 6; do
    # Extract contact angle from params file for the banner
    THETA=$(awk -v id="${CASE_ID}" '
        !/^#/ && NF>=7 { line=$0; sub(/#.*$/,"",line); split(line,a); if(a[1]+0==id) print a[2] }
    ' "${SCRIPT_DIR}/capillary_rise_params.txt")
    banner "Case ${CASE_ID}  θ=${THETA}°  (np=4)"
    mpirun -np 4 python3 "${SCRIPT_DIR}/run_capillary_rise.py" \
        ${NL} "${INIT}" ${CASE_ID} ${STEPS}
done

# ─── Step 3: print summary ───────────────────────────────────────────────────
banner "Summary"
SUMMARY="${SCRIPT_DIR}/capillary_rise_summary.dat"
if [[ -f "${SUMMARY}" ]]; then
    echo "  ${SUMMARY}"
    echo
    printf "  %-8s  %-9s  %-12s  %-12s  %-8s\n" \
           "case_id" "theta[°]" "h_Jurin[mm]" "h_meas[mm]" "err[%]"
    echo "  --------  ---------  ------------  ------------  --------"
    awk 'NF>=7 && !/^#/ {
        printf "  %8d  %9.1f  %12.3f  %12.3f  %8.2f\n",
               $1, $2, $3*1e3, $4*1e3, $5
    }' "${SUMMARY}"
    echo
    echo "  Plot h_meas vs cos(theta) — should be a straight line: h = 2σcos(θ)/(ρgR)"
else
    echo "  No summary file found."
fi

popd > /dev/null
banner "All capillary rise cases complete."
echo "  GSD trajectories  : caprise_*_case*_theta*_run.gsd"
echo "  Log files         : caprise_*_case*_theta*_run.log"
echo "  Summary table     : capillary_rise_summary.dat"
