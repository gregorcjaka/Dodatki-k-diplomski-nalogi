"""
Kodo pozenemo v 'Scripting' zavihku v Blenderju.
Iz obeh .csv datotek o vozliscih in povrsinah nam sestavi objekt.
Pred zagonom spremenimo lokacijo datotek in imena .csv datotek (ne ce smo uporabili Cdb_to_csv_conversion.py).
"""

import bpy
from pathlib import Path
from bpy import context

# 0) Lokacija in imena datotek
folder = Path(r"C:\Location\to\both\csvs")
v_path = folder / "beam_vertices.csv"
f_path = folder / "beam_faces.csv"
if not (v_path.is_file() and f_path.is_file()):
    raise FileNotFoundError(f"CSV files not found in: {folder}")

# 1) Urejanje in prebiranje .csv
try:
    import numpy as np
    verts = np.loadtxt(v_path, delimiter=",", skiprows=1)
    faces = np.loadtxt(f_path, delimiter=",", skiprows=1, dtype=np.int64)
    verts = [tuple(v) for v in verts]
    faces = [tuple(f) for f in faces]
except ImportError:
    import csv
    with open(v_path, newline="") as vf:
        verts = [tuple(map(float, row)) for i, row in enumerate(csv.reader(vf)) if i]
    with open(f_path, newline="") as ff:
        faces = [tuple(map(int, row))   for i, row in enumerate(csv.reader(ff)) if i]

print(f"Loaded  {len(verts):,} vertices and {len(faces):,} faces")

# 2) Naredimo mrezo
mesh = bpy.data.meshes.new("BeamMesh")
mesh.from_pydata(verts, [], faces)
mesh.validate(verbose=True)
mesh.update(calc_edges=True)

obj = bpy.data.objects.new("Beam", mesh)
context.collection.objects.link(obj)
context.view_layer.objects.active = obj
obj.select_set(True)

print("Undeformed beam imported")
