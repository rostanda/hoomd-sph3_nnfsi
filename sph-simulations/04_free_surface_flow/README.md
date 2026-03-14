# 04 — Free-Surface Flow Benchmarks (SinglePhaseFlowFS)

Benchmark suite for the `SinglePhaseFlowFS` solver, which extends
`SinglePhaseFlowTV` with free-surface detection (Shepard completeness ratio),
contact-angle enforcement (Huber et al. 2016), mean-curvature estimation, and
CSF surface tension (Colagrossi & Landrini 2003).

All simulations use the WendlandC4 kernel with SUMMATION density and the
KickDriftKickTV integrator (required by the TV base class).

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

- Marrone et al. (2010), *Comput. Fluids* — free-surface detection (lambda criterion)
- Huber et al. (2016), *Int. J. Numer. Meth. Fluids* — contact-angle BC
- Colagrossi & Landrini (2003), *J. Comput. Phys.* — CSF surface tension for SPH
- Adami et al. (2013), *J. Comput. Phys.* — transport-velocity formulation
- Martin & Moyce (1952), *Phil. Trans. R. Soc. Lond. A* — dam-break reference
