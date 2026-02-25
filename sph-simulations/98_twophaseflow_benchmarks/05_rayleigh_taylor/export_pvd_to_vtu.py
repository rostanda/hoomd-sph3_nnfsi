#!/usr/bin/env python3
"""Re-export every frame from the PVD collection to individual VTU files via vtk."""

import vtk, os, glob, xml.etree.ElementTree as ET

pvd_file = 'rt_20_86_20_vs_0.000050_run.pvd'
out_dir  = 'rt_20_86_20_vs_0.000050_run_pvd'

os.makedirs(out_dir, exist_ok=True)

# Parse PVD to get (timestep, vtu_path) pairs
tree = ET.parse(pvd_file)
datasets = [(float(ds.get('timestep')), ds.get('file'))
            for ds in tree.iter('DataSet')]
print(f'PVD contains {len(datasets)} timesteps')

writer = vtk.vtkXMLUnstructuredGridWriter()
writer.SetDataModeToBinary()

for i, (t, src) in enumerate(datasets):
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(src)
    reader.Update()

    out = os.path.join(out_dir, f'run_{int(t):09d}.vtu')
    writer.SetFileName(out)
    writer.SetInputConnection(reader.GetOutputPort())
    writer.Write()

    if i % 50 == 0:
        print(f'  [{i+1}/{len(datasets)}] step {int(t):>6d} → {out}')

print(f'Done. {len(datasets)} VTU files written to {out_dir}/')
