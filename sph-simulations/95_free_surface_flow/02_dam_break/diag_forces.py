#!/usr/bin/env python3
"""
Diagnostic: find which particle has the largest force/acceleration at step 0
for the dam-break benchmark.  Run prepRun (sim.run(0)) to compute step-0
forces, then use a vlimit+xlimit-protected single step to infer accelerations.
"""
import sys, os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
sys.path.insert(0, os.path.join(_ROOT, 'helper_modules'))

import hoomd
from hoomd import sph
import numpy as np
import gsd.hoomd
import sph_helper

INITGSD = os.path.join(os.path.dirname(__file__),
                        'dam_break_88_52_17_vs_0.000500_init.gsd')

# ─── Setup (identical to run_dam_break.py) ────────────────────────────────────
device = hoomd.device.CPU(notice_level=2)
sim    = hoomd.Simulation(device=device)

num_length = 20
lref       = 0.01
H0         = 2.0 * lref
dx         = lref / num_length
rho0       = 1000.0
viscosity  = 1e-3
gy         = -9.81
drho       = 0.01
backpress  = 0.01
c0_wave    = np.sqrt(abs(gy) * H0)
refvel     = c0_wave
sigma      = 0.0
fs_threshold  = 0.75
contact_angle = np.pi / 2

kernel     = 'WendlandC4'
slength    = hoomd.sph.kernel.OptimalH[kernel] * dx
rcut       = hoomd.sph.kernel.Kappa[kernel] * slength
kernel_obj = hoomd.sph.kernel.Kernels[kernel]()
kappa      = kernel_obj.Kappa()

sim.create_state_from_gsd(filename=INITGSD, domain_decomposition=(None, None, 1))

nlist = hoomd.nsearch.nlist.Cell(buffer=rcut * 0.05,
                                  rebuild_check_delay=1, kappa=kappa)
eos = hoomd.sph.eos.Tait()
eos.set_params(rho0, backpress)

filterfluid = hoomd.filter.Type(['F'])
filtersolid = hoomd.filter.Type(['S'])

model = hoomd.sph.sphmodel.SinglePhaseFlowFS(
    kernel=kernel_obj, eos=eos, nlist=nlist,
    fluidgroup_filter=filterfluid, solidgroup_filter=filtersolid,
    densitymethod='SUMMATION',
    sigma=sigma, fs_threshold=fs_threshold, contact_angle=contact_angle)

model.mu                  = viscosity
model.gy                  = gy
model.damp                = 0
model.artificialviscosity = True
model.alpha               = 0.1
model.beta                = 0.0
model.densitydiffusion    = False

maximum_smoothing_length = sph_helper.set_max_sl(sim, device, model)
c, cond = model.compute_speedofsound(
    LREF=lref, UREF=refvel, DX=dx, DRHO=drho,
    H=maximum_smoothing_length, MU=viscosity, RHO0=rho0)
print(f'Speed of sound: {c:.4f} m/s  ({cond})')

sph_helper.update_min_c0(device, model, c, mode='uref', lref=lref,
                          uref=refvel, cfactor=10.0)
dt, dt_cond = model.compute_dt(
    LREF=lref, UREF=refvel, DX=dx, DRHO=drho,
    H=maximum_smoothing_length, MU=viscosity, RHO0=rho0)
print(f'Timestep: {dt:.3e} s  ({dt_cond})')

integrator = hoomd.sph.Integrator(dt=dt)
kdktv = hoomd.sph.methods.KickDriftKickTV(filter=filterfluid,
                                           densitymethod='SUMMATION')
# Tight position limiter to prevent out-of-box crash
kdktv.xlimit = dx * 0.001   # 1e-6 m max displacement per step
integrator.methods.append(kdktv)
integrator.forces.append(model)
sim.operations.integrator = integrator

# ─── Run 0 steps to trigger prepRun and compute step-0 forces ─────────────────
print("\nRunning sim.run(0) to trigger prepRun (step-0 force computation)...")
sim.run(0, write_at_start=False)
print("prepRun complete.")

# ─── Read accelerations via local snapshot ─────────────────────────────────────
# After run(0), accelerations were set by prepRun::computeAccelerations(0)
# Try to read them via cpu_local_snapshot
try:
    with sim.state.cpu_local_snapshot as snap:
        # tag, typeid, position, velocity
        tag    = np.array(snap.particles.tag)
        typeid = np.array(snap.particles.typeid)
        pos    = np.array(snap.particles.position)
        vel    = np.array(snap.particles.velocity)  # includes mass in vel[:,3]?

        fluid  = (typeid == 0)
        tag_f  = tag[fluid]
        pos_f  = pos[fluid]
        vel_f  = vel[fluid]

        print(f"\nFluid particles: {fluid.sum()}")
        print(f"Fluid velocity (should all be 0): max|v| = {np.max(np.abs(vel_f)):.3e} m/s")

        # Try auxiliary arrays to get BPC (aux2) and lambda (aux4)
        try:
            aux2 = np.array(snap.particles.getAuxiliaries2())
            aux4 = np.array(snap.particles.getAuxiliaries4())
            bpc_f = aux2[fluid]
            lam_f = aux4[fluid, 0]
            kap_f = aux4[fluid, 1]
            print(f"Max |BPC| on fluid: {np.max(np.abs(bpc_f)):.3e}")
            print(f"Lambda range on fluid: [{lam_f.min():.3f}, {lam_f.max():.3f}]")
        except Exception as e:
            print(f"(aux arrays not accessible: {e})")

except Exception as e:
    print(f"cpu_local_snapshot failed: {e}")

# ─── Run 1 step with xlimit, read result ──────────────────────────────────────
print("\nRunning sim.run(1) with xlimit to infer accelerations...")
try:
    sim.run(1, write_at_start=False)
    print("Step 1 complete (no crash with xlimit).")
except RuntimeError as e:
    print(f"Crashed even with xlimit: {e}")

# Read state after (hopefully successful) step
try:
    with sim.state.cpu_local_snapshot as snap:
        tag    = np.array(snap.particles.tag)
        typeid = np.array(snap.particles.typeid)
        pos    = np.array(snap.particles.position)
        vel    = np.array(snap.particles.velocity)

        fluid  = (typeid == 0)
        tag_f  = tag[fluid]
        pos_f  = pos[fluid]
        vel_f  = vel[fluid]

        # After 1 step: v_full ≈ a(0)*dt + a(1)*dt/2 ≈ a_avg * dt
        # Use |v| / dt as proxy for |accel|
        vmag   = np.sqrt(np.sum(vel_f[:, :3]**2, axis=1))
        amag   = vmag / dt

        n_top  = min(20, len(amag))
        top_idx = np.argsort(amag)[-n_top:][::-1]

        print(f"\nTop {n_top} fluid particles by estimated |accel| (proxy = |v|/dt):")
        print(f"{'tag':>8}  {'|accel|':>12}  {'x':>10}  {'y':>10}  {'vx':>12}  {'vy':>12}")
        for idx in top_idx:
            print(f"{tag_f[idx]:8d}  {amag[idx]:12.3e}"
                  f"  {pos_f[idx,0]:10.6f}  {pos_f[idx,1]:10.6f}"
                  f"  {vel_f[idx,0]:12.3e}  {vel_f[idx,1]:12.3e}")

        # Check particle 17831 specifically
        mask_17831 = (tag_f == 17831)
        if mask_17831.any():
            idx = np.where(mask_17831)[0][0]
            print(f"\nParticle 17831: |accel_proxy| = {amag[idx]:.3e} m/s²"
                  f", pos=({pos_f[idx,0]:.6f},{pos_f[idx,1]:.6f})"
                  f", v=({vel_f[idx,0]:.3e},{vel_f[idx,1]:.3e})")
        else:
            print("\nParticle 17831 not found in fluid group on this rank")

except Exception as e:
    print(f"Reading snapshot after run failed: {e}")
    import traceback
    traceback.print_exc()
