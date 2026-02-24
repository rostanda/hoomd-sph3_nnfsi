# 95 — Free-Surface Flow Benchmarks (SinglePhaseFlowFS)

Benchmark suite for the `SinglePhaseFlowFS` solver, which extends
`SinglePhaseFlowTV` with free-surface detection (Shepard completeness ratio),
contact-angle enforcement (Huber et al. 2016), mean-curvature estimation, and
CSF surface tension (Colagrossi & Landrini 2003).

All simulations use the WendlandC4 kernel with SUMMATION density and the
KickDriftKickTV integrator (required by the TV base class).

---

## 01 — Hydrostatic column with free surface

**Physics:** A fluid column of height `lref = 1 mm` under gravity
(`gy = -9.81 m/s²`) is enclosed by a solid bottom plate but has a **free
surface on top** (no top solid wall).  `σ = 0` — only surface detection.

**Checks:**
1. Top-layer particles are correctly flagged as free surface (λ < 0.75).
2. Pressure clamping: no fluid particle has P < 0.
3. Hydrostatic density profile matches `ρ(y) ≈ ρ₀(1 + ρ₀|g|(y_top − y)/c²ρ₀)`.
4. RMS spurious velocity remains small.

```bash
cd 01_hydrostatic_fs
python3 create_input_geometry.py 20
python3 run_hydrostatic_fs.py 20 hydrostatic_fs_*_init.gsd 10001
```

---

## 02 — 2-D Dam break

**Physics:** A rectangular water column (width `a = lref = 10 mm`,
height `H₀ = 2a`) collapses under gravity into an empty channel of
length `4a`.  The top and right side of the column are free surfaces.
`σ = 0`.

**Validation:** front position `X*(T*)` is compared to the shallow-water
analytical prediction (Martin & Moyce 1952):

    X*(T*) = 1 + 2√2 · T*,    T* = t √(g/a)

(valid for `T* ≲ 3`).

```bash
cd 02_dam_break
python3 create_input_geometry.py 20
python3 run_dam_break.py 20 dam_break_*_init.gsd 10001
```

---

## 03 — Sessile droplet — contact-angle enforcement

**Physics:** A 2-D liquid droplet (initially a semicircle, `θ_init = 90°`)
rests on a flat solid wall.  Surface tension (`σ = 0.072 N/m`) and
contact-angle enforcement drive it to the prescribed equilibrium angle
`θ_eq` (default 60°).

**Validation:** the equilibrium contact angle is recovered from the
final droplet geometry:

    θ = 2 atan(h / r)

where `h` is the droplet height and `r` the base half-width.
For volume-conserving relaxation (2-D area = π R²/2), the theoretical
equilibrium cap dimensions are also reported.

```bash
cd 03_sessile_droplet
python3 create_input_geometry.py 20
python3 run_sessile_droplet.py 20 sessile_droplet_*_init.gsd 30001 60
# Second angle (120° — partial non-wetting):
python3 run_sessile_droplet.py 20 sessile_droplet_*_init.gsd 30001 120
```

---

## Calling convention

```
python3 run_<benchmark>.py <num_length> <init_gsd_file> [steps] [extra_args]
```

| Argument     | Default | Description                       |
|--------------|---------|-----------------------------------|
| `num_length` | —       | particles per reference length    |
| `init_gsd`   | —       | initial GSD file from `create_*`  |
| `steps`      | varies  | number of integration steps       |
| extra args   | —       | benchmark-specific (see scripts)  |

## References

- Marrone et al. (2010), *Comput. Fluids* — free-surface detection (λ criterion)
- Huber et al. (2016), *Int. J. Numer. Meth. Fluids* — contact-angle BC
- Colagrossi & Landrini (2003), *J. Comput. Phys.* — CSF surface tension for SPH
- Adami et al. (2013), *J. Comput. Phys.* — transport-velocity formulation
- Martin & Moyce (1952), *Phil. Trans. R. Soc. Lond. A* — dam-break reference
