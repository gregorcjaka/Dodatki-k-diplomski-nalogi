"""
Skripto podano v Ansys pod zavihek 'Automation' in jo pozenemo po resenem problemu.
Dobimo 3 .txt datoteke za vsak casovni trenutek.
Pred zagonom spremenimo izhodno datoteko in vektor time_points na casovne tocke, kjer smo definirali korake.
"""

import os
export_folder = r"C:\Location\for\results"
time_points = [0.2, 0.4, 0.6, 0.8, 1.0]
directions = [("X", NormalOrientationType.XAxis), ("Y", NormalOrientationType.YAxis), ("Z", NormalOrientationType.ZAxis)]

# 0) Preverimo, da mapa obstaja
if not os.path.exists(export_folder):
    os.makedirs(export_folder)

# 1) Najdemo resitev modela
solution = Model.Analyses[0].Solution

# 2) Zanka za kreiranje rezultatov in izvazanje datotek
for dir_label, dir_enum in directions:
    for i, t in enumerate(time_points, start=1):
        # Dodamo rezultat
        result = solution.AddDirectionalDeformation()
        result.NormalOrientation = dir_enum
        result.By = SetDriverStyle.Time
        result.DisplayTime = Quantity(str(t) + "[s]")
        result.CalculateTimeHistory = False
        result.EvaluateAllResults()

        # Shranimo pod ime datoteke: DirDeform_X_T1.txt, ...
        filename = os.path.join(
            export_folder,
            "DirDeform_{}_T{}.txt".format(dir_label, i)
        )
        result.ExportToTextFile(filename)

        # Za urejenost zbrisemo evaluiran rezultat
        result.Delete()

print("Export completed.")
