"""
Koda poljubne rezultate numericne analize pripise vozliscem objekta v Blenderju.
Skripta pricakuje .txt datoteko oblike:
Node Number	X Location (m)	Y Location (m)	Z Location (m)	Equivalent (von-Mises) Stress (Pa)
1	1.1959e-003	0.10182	5.e-003	2.5958e+009
2	1.1959e-003	0.10182	1.e-002	2.4928e+009
3	1.1959e-003	0.10182	1.5e-002	2.3612e+009
4	1.1959e-003	0.10182	2.e-002	2.3229e+009
5	1.1959e-003	0.10182	2.5e-002	2.2931e+009
...

Skripto pozenemo v Blender 'Scripting' zavihku.
Pred zagonom spremenimo lokacijo rezultatov in ime rezultatov ter ime dodanega objekta.
"""
import bpy, csv, mathutils

txt_path = r"C:\Location\to\Stress_results.txt"  # Lokacija do .txt datoteke
object = "object_name"                           # Ime objekta v sceni, na katerega natisnemo rezultate
field = "result_header_name"                     # Ime rezultatov
scale = 1.0                                      # Ce so enote v metrih, pusti na 1
num_neighbours = 9                               # Stevilo uporabljenih sosedov za glajenje

def str2f(s): return float(s.replace(',', '.'))

# 1) Preberemo datoteko z rezultati, sestavimo mathutils vektor, shranimo napetost v nodes_val
nodes_co = []
nodes_val = []

with open(txt_path, encoding="utf-8", newline='') as f:
    # Spustimo vse vrstice s komentarji
    for line in f:
        if not line.startswith('#'):
            header_raw = line.rstrip('\r\n').split('\t')
            break
    header = [h.strip() for h in header_raw]
    data = {h: i for i, h in enumerate(header)}
    
    # Preverimo, da imamo vse potrebne podatke
    coord_headers = ("X Location (m)", "Y Location (m)", "Z Location (m)")
    need = (*coord_headers, field)
    if not all(k in data for k in need):
        raise RuntimeError(f"Missing one of {need} in TXT header: {header}")

    for row in csv.reader(f, delimiter='\t'):
        if not row or len(row) < len(header):
            continue
        co = mathutils.Vector((str2f(row[data["X Location (m)"]]) * scale, str2f(row[data["Y Location (m)"]]) * scale, str2f(row[data["Z Location (m)"]]) * scale))
        nodes_co.append(co)
        nodes_val.append(str2f(row[data[field]]))

# 2) Zgradimo drevo velikosti nodes_co in ga napolnimo s koordinatami vozlisc
kd_tree = mathutils.kdtree.KDTree(len(nodes_co))
for i, co in enumerate(nodes_co):
    kd_tree.insert(co, i)
kd_tree.balance()        # Poskrbi, da so tocke v optimalnem uravnotezenem drevesu

# 3) Pripisemo vrednosti na verteks-barvo
obj = bpy.data.objects[object]
mesh = obj.data
world = obj.matrix_world          # V primeru, da transformacije niso potrjene

# Ustvarimo verteks sloj (oz. ga pridobimo, ce je ta ze ustvarjen)
layer = mesh.color_attributes.get(field) \
        or mesh.color_attributes.new(field, type='FLOAT_COLOR', domain='POINT')

# Glavna zanka
for v in mesh.vertices:
    co_world = world @ v.co
    # Najdemo najblizja sosednja vozlisca
    neighbors = kd_tree.find_n(co_world, num_neighbours)

    # Racunanje utezi vsakega vozlisca
    val_num = 0.0
    val_den = 0.0
    for co, idx, dist in neighbors:
        w = 1.0 / (dist*dist + 1e-12)   # Vsak sosed prispeva s tezo ∝ 1 / r² (teza pada z razdaljo)
        val_num += w * nodes_val[idx]
        val_den += w

    val = val_num / val_den             # Utezeno povprecje
    layer.data[v.index].color = (val, val, val, 1.0)       
mesh.update()

print("Result value min:", min(nodes_val))      # Dobimo se informacijo o min in max resitvi
print("Result value max:", max(nodes_val))
print(f"Stamped stress on {len(mesh.vertices):,} of {len(mesh.vertices):,} vertices.")



