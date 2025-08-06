"""
Skripta ustvari nedeformirano obliko, ji pripise kljuce oblike in ustvari animacijo.
Pozenemo jo v Blender 'Scripting' zavihku.
Pred zagonom spremenimo lokacijo in imena datotek, stevilo korakov, faktor skaliranja in presledek med slicicami.
Skripta predvideva, da imamo v osnovnem direktoriju obe .csv datoteki in v njem se eno mapo s smernimi deformacijami.
Imena datotek s smernimi deformacijami morajo biti oblikovana kot "DirDeform_{koordinata}_T{korak}.txt".
"""

# 0) Zacetne nastavitve
from pathlib import Path
import bpy, re, numpy as np
from bpy import context
folder       = Path(r"C:\Location\to\csv\files")
deform_dir   = folder / "Folder_with_dirResults_location"

stages = [1, 2, 3, 4, 5]     # koraki deformacije
tol = 1e-5                   # toleranca za ujemanje koordinat
def_scale = 1                # faktor skaliranja deformacije
add_keyframes = True         # ce ne zelimo animacije, spremenimo v False
frame_step = 20              # stevilo slicic med posameznimi kljuci

v_path = folder / "beam_vertices.csv"
f_path = folder / "beam_faces.csv"


# Pomagalna funkcija, float -> int za hitrejse iskanje po slovarju
factor = 1.0 / tol                 # npr. 1e5
def converter(pt):
    """
    Pretvori stevila s plavajoco vejico v integer, za hitrejse iskanje.
    Pricakovan vhod: numpy.ndarray ali 3-mestna terka (x, y, z).
    Izhod: Terka pretvorjenih koordinat.
    """                      
    return tuple(int(round(c*factor)) for c in pt)


# Pomagalna funkcija za branje .txt datotek z deformacijami
def load_def_file(path):
    """
    Prebere eno datoteko z deformacijami.
    Pricakovan vhod: Lokacija datoteke z njenim imenom.
    Izhod: Slovar, ki ima za kljuc kvantizirane trojke integerjev in shrani velikost deformacije.
    """
    d = {}
    with open(path, encoding="utf-8") as f:
        next(f)                    # Spustimo glavo
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            # Locilo je poljubno stevilo presledkov
            cols = re.split(r"[ \t]+", line.replace(",", "."))
            if len(cols) < 5:       # Spustimo nepopolne vrstice
                continue
            x, y, z = map(float, cols[1:4])   # Preberemo potrebne podatke
            disp = float(cols[4])
            d[converter((x, y, z))] = disp   # Shrani se le (x,y,z): deformacija
    return d


# Pomagalna funkcija za pretvorbo v 'bucket' slovar
def bucketisation(raw_map):
    """
    Spremeni slovar v 'bucket' slovar.
    Pricakovan vhod: Slovar, ki ima za kljuc koordinate in shranjuje velikost deformacije.
    Izhod: Slovar oblike (x, y, z): [[x, y, z], displ]; kjer koordinate v vrednosti pretvorimo 
    nazaj v osnovne enote (metri).
    """
    b = {}          # Nov prazen slovar
    for key, disp in raw_map.items():
        b.setdefault(key, []).append((np.array(key, float)/factor, disp))
    return b


nb_shifts = [(i, j, k) for i in (-1,0,1) for j in (-1,0,1) for k in (-1,0,1)]   # 27 integer trojic vseh kombinacij koordinat
tol2 = tol*tol     # Vzamemo kvadrat tolerance, saj bo primerjan kvadrat razdalje

# Pomagalna funkcija za iskanje najbližjega pomika
def nearest_disp(pt, buckets):
    """
    Najde najblizjo deformacijsko vrednost. Ce je najdena tocka v radiusu tol, vrne def., sicer 0.
    Pricakovan vhod: pt = terka koordinat iz .csv, buckets = bucket slovar.
    Izhod: Deformacija tocke pt oz. 0, ce ta ni najdena.
    """
    k = converter(pt)        # Poklicemo converter, da so koordinate enake oblike kot v bucketu
    best_d  = 0.0
    best_r2 = tol2 + 1.0
    for s in nb_shifts:
        # Iscemo v blizini tocke iz bucketa
        for coord, disp in buckets.get((k[0]+s[0], k[1]+s[1], k[2]+s[2]), []):
            r2 = np.dot(pt - coord, pt - coord)   # Ubistvu vsota kvadratov spremembe vseh koordinat
            if r2 < best_r2:
                best_r2 = r2
                best_d  = disp
    return best_d if best_r2 < tol2 else 0.0


# 1)  Rekonstrukcija mreze
verts = np.loadtxt(v_path, delimiter=",", skiprows=1, dtype=float)
faces = np.loadtxt(f_path, delimiter=",", skiprows=1, dtype=np.int64)
print(f"Loaded {len(verts):,} verts | {len(faces):,} faces")

mesh = bpy.data.meshes.new("BeamMesh")
mesh.from_pydata(verts.tolist(), [], faces.tolist())
mesh.validate(); mesh.update()

obj = bpy.data.objects.new("Beam", mesh)
context.collection.objects.link(obj)
context.view_layer.objects.active = obj


# 2) Dodamo osnoven kljuc oblike
obj.shape_key_add(name="Basis", from_mix=False)


# 3) Procesiramo vsako stopnjo deformacije
for step in stages:
    print(f"Stage {step}: reading files…")
    dx_map = load_def_file(deform_dir / f"DirDeform_X_T{step}.txt")
    dy_map = load_def_file(deform_dir / f"DirDeform_Y_T{step}.txt")
    dz_map = load_def_file(deform_dir / f"DirDeform_Z_T{step}.txt")

    dx_b = bucketisation(dx_map)
    dy_b = bucketisation(dy_map)
    dz_b = bucketisation(dz_map)

    # Dobimo nove koordinate deformiranega telesa
    new_vs = []
    miss   = 0
    for v in verts:
        dx = nearest_disp(v, dx_b)
        dy = nearest_disp(v, dy_b)
        dz = nearest_disp(v, dz_b)
        if dx==dy==dz==0.0:
            miss += 1
        new_vs.append(v + def_scale*np.array([dx,dy,dz]))
    print(f"Unmatched vertices : {miss:,}")

    # Za vsak korak posebej dodamo svoj kljuc oblike
    key = obj.shape_key_add(name=f"Step_{step}", from_mix=False)
    for i, co in enumerate(new_vs):
        key.data[i].co = co

print("All shape keys added.")


# 4) Opcijsko naredimo animacijo
if add_keyframes:
    obj.animation_data_clear()   # Najprej odstranimo obstojece animacije

    # V prvi slicici nastavimo vrednosti vseh kljucev na 0
    for kb in obj.data.shape_keys.key_blocks:
        kb.value = 0.0
        kb.keyframe_insert("value", frame=0)

    # Prehod iz enega na drug kljuc vsakih 20 slicic (privzeto)
    for idx, step in enumerate(stages):                # idx = 0 … 4
        frame = (idx + 1) * frame_step                 # privzeto 20, 40, 60, ...
        # Najprej spet “pocisti” vse
        for kb in obj.data.shape_keys.key_blocks:
            kb.value = 0.0
            kb.keyframe_insert("value", frame=frame)
        # Potem vkljuci zeljeni kljuc
        sk = obj.data.shape_keys.key_blocks[f"Step_{step}"]
        sk.value = 1.0
        sk.keyframe_insert("value", frame=frame)

    # Prehod nazaj na osnovo
    start_rev = len(stages) * frame_step          # Obrat pri sredinski slicici
    for jdx, step in enumerate(reversed(stages[:-1]), start=1):  # 4,3,2,1
        frame = start_rev + jdx * frame_step      
        # vse na 0
        for kb in obj.data.shape_keys.key_blocks:
            kb.value = 0.0
            kb.keyframe_insert("value", frame=frame)
        # vključi ustrezen korak nazaj
        sk = obj.data.shape_keys.key_blocks[f"Step_{step}"]
        sk.value = 1.0
        sk.keyframe_insert("value", frame=frame)

    # Koncna slicica – osnova (vsi = 0)
    end_frame = len(stages) * 2 * frame_step      # Zadnja slicica
    for kb in obj.data.shape_keys.key_blocks:
        kb.value = 0.0
        kb.keyframe_insert("value", frame=end_frame)

    # Nastavimo linearno interpolacijo za lepse prehode
    action = obj.data.shape_keys.animation_data.action
    for fcurve in action.fcurves:
        for kp in fcurve.keyframe_points:
            kp.interpolation = 'LINEAR'


print("Done — play the Timeline to see the deformation!")


