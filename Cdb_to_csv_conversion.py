"""
Dobljeno .cdb datoteko spremenimo v 2 .csv datoteki.
V prvi so posamezna vozlisca in x,y,z koordinate vsakega.
V drugi so shranjeni podatki, katere 3 tocke tvorijo posamezen trikotnik.

Datoteko zazenemo preko zazenemo preko ukazne vrstice: 
python Cdb_to_csv_conversion.py "C:\lokacija\cdb\datoteke.cdb"
"""

import sys
from pathlib import Path
import numpy as np
import ansys.mapdl.reader as rdr


# 0) Potrdimo, da podana datoteka obstaja
if len(sys.argv) != 2:
    sys.exit("Run as: python Cdb_to_csv_conversion.py C:\location\cdb\file.cdb")

cdb_path = Path(sys.argv[1]).expanduser().resolve()
if not cdb_path.is_file():
    sys.exit(f"File not found: {cdb_path}")


# 1) Branje CDB datoteke
arc  = rdr.Archive(str(cdb_path))      # Nalozimo Ansys Cdb z PyMAPDL Readerjem
grid = arc.grid                        # Pridobimo celotno mrezo
print(f"Volume mesh : {grid.n_points} points  |  {grid.n_cells} cells")


# 2) Pridobivanje povrsine (potrebujemo le povrsinske tocke), triangulacija
surf = grid.extract_surface().triangulate()
print(f"Surface mesh: {surf.n_points} verts   |  {surf.n_faces} tris")


# 3) Podatke shranimo v Numpy matrike
verts = surf.points                               
faces = surf.faces.reshape(-1, 4)[:, 1:]           


# 4) Zapisemo obe .csv datoteki v direktorij .cdb datoteke
out_dir = cdb_path.parent
vfile   = out_dir / "beam_vertices.csv"
ffile   = out_dir / "beam_faces.csv"

np.savetxt(vfile, verts, delimiter=",", header="x,y,z", comments="")
np.savetxt(ffile, faces, delimiter=",", header="v1,v2,v3", fmt="%d", comments="")

print(f"CSVs written:\n   {vfile}\n   {ffile}")

