"""
Plot permeability vs Reynolds number for all BCC sphere-packing runs.
Data taken from the last timestep of each bcc100_fx*_run.log file.
"""

import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── collect data ──────────────────────────────────────────────────────────────
log_dir = os.path.dirname(os.path.abspath(__file__))
log_files = sorted(glob.glob(os.path.join(log_dir, "bcc100_fx*_run.log")))

Re_list, k_list, knorm_list, fx_list = [], [], [], []

for fpath in log_files:
    if os.path.getsize(fpath) == 0:
        print(f"  SKIP (empty): {os.path.basename(fpath)}")
        continue
    # parse fx from filename
    fname = os.path.basename(fpath)
    fx_str = fname.replace("bcc100_fx", "").replace("_run.log", "")
    fx = float(fx_str)

    # last data line
    with open(fpath) as f:
        lines = [l for l in f if l.strip() and l.strip()[0].isdigit()]
    if not lines:
        continue
    cols = lines[-1].split()
    Re    = float(cols[8])   # Re_grain
    k_m2  = float(cols[9])   # custom.k_m2
    k_norm = float(cols[10]) # k_norm

    Re_list.append(Re)
    k_list.append(k_m2)
    knorm_list.append(k_norm)
    fx_list.append(fx)

Re_arr    = np.array(Re_list)
k_arr     = np.array(k_list)
knorm_arr = np.array(knorm_list)

# sort by Re
order = np.argsort(Re_arr)
Re_arr    = Re_arr[order]
k_arr     = k_arr[order]
knorm_arr = knorm_arr[order]

# reference values
k_KC = 7.6367e-07   # m²  (Kozeny-Carman)

# ── plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("BCC sphere-packing permeability  (not fully converged — jobs hit walltime)",
             fontsize=11, color="firebrick")

marker_kw = dict(marker="o", markersize=6, linewidth=1.5, color="steelblue")

# ── left: k [m²] vs Re ───────────────────────────────────────────────────────
ax = axes[0]
ax.loglog(Re_arr, k_arr, **marker_kw, label="SPH (last timestep)")
ax.axhline(k_KC, color="tomato", linestyle="--", linewidth=1.5, label=f"$k_{{KC}}$ = {k_KC:.3e} m²")
ax.set_xlabel("$Re_{grain}$  [-]", fontsize=12)
ax.set_ylabel("$k$  [m²]", fontsize=12)
ax.set_title("Permeability vs Reynolds number")
ax.legend(fontsize=10)
ax.grid(True, which="both", alpha=0.3)
ax.set_xlim([Re_arr.min() * 0.5, Re_arr.max() * 2])

# annotate Darcy / inertial regions
ax.axvline(1.0, color="gray", linestyle=":", linewidth=1)
ax.text(0.6, ax.get_ylim()[0] * 3, "Darcy", ha="right", va="bottom",
        color="gray", fontsize=9)
ax.text(1.5, ax.get_ylim()[0] * 3, "inertial", ha="left", va="bottom",
        color="gray", fontsize=9)

# ── right: k/k_KC vs Re ──────────────────────────────────────────────────────
ax2 = axes[1]
ax2.semilogx(Re_arr, knorm_arr, **marker_kw, label="SPH (last timestep)")
ax2.axhline(1.0, color="tomato", linestyle="--", linewidth=1.5, label="$k/k_{KC}$ = 1  (Darcy)")
ax2.axvline(1.0, color="gray", linestyle=":", linewidth=1)
ax2.set_xlabel("$Re_{grain}$  [-]", fontsize=12)
ax2.set_ylabel("$k / k_{KC}$  [-]", fontsize=12)
ax2.set_title("Normalised permeability vs Reynolds number")
ax2.legend(fontsize=10)
ax2.grid(True, which="both", alpha=0.3)
ax2.set_ylim([0, max(knorm_arr) * 1.2])

# colour-code points by convergence quality ($\Delta$/step proxy: k_norm)
# add colourbar showing k_norm as rough "how close to converged"
sc = ax2.scatter(Re_arr, knorm_arr, c=knorm_arr, cmap="RdYlGn",
                 s=60, zorder=5, vmin=0, vmax=1)
cbar = fig.colorbar(sc, ax=ax2, pad=0.02)
cbar.set_label("$k_{norm}$ (higher = more developed)", fontsize=9)

plt.tight_layout()
out = os.path.join(log_dir, "permeability_vs_re.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.show()
