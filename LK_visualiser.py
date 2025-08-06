"""
Koda vzame podan mp4 dokument, ga odpre, dovoli da izberemo poljubne tocke in 
jih po metodi Lucas-Kanade spremlja v vseh slicicah.

Potrebujemo modula opencv-python ≥ 4.5, numpy.
Pred zagonom spremenimo imena datotek in LK parametre.
Pricakovan rezultat je mp4 datoteka z barvnimi sledmi gibanja in .csv datoteka s koordinatami v px.
Pozenemo preko ukazne vrstice z:
python LK_visualiser.py
"""

import cv2
import numpy as np
import csv
from pathlib import Path

video_in = r"input_video.mp4"      # Vhodna mp4 datoteka
video_out = r"output_video.mp4"    # Izhodna mp4 datoteka
csv_out = r"output_csv.csv"        # Izhodna csv datoteka
max_frames = None                  # None = uporabimo celoten posnetek

# Parametri za LK
lk_win = (21, 21)       # Velikost okna za iskanje
lk_maxlevel = 3         # Stevilo piramidnih slojev
lk_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)  # Algoritem se ustavi pri 30 iteracijah oz. epsilonu pod 0.01


# 1) Inicializacija
cap = cv2.VideoCapture(video_in)
if not cap.isOpened():
    raise IOError(f"Could not open video: {video_in}")

# Izvlecemo osnovne podatke o posnetku
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = None
if video_out:
    writer = cv2.VideoWriter(video_out, fourcc, fps, (width, height))

# Preberemo prvo slicico
ret, first = cap.read()
assert ret, "Failed to read the first frame"
gray_prev = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)  # Pretvorimo jo v crno-belo


# 2) Model za pametno izbiro tock 
clicked = []                # Mesta nasih klikov
refined_pts = []                # Dodelana mesta klikov
display = first.copy()

search_w = 20   # Velikost okna za iskanje (px)

def draw_overlay():
    """
    Pomozna funkcija, ki narise kliknjene tocke.
    """
    tmp = display.copy()
    for px, py in clicked:
        cv2.circle(tmp, (px, py), 3, (255, 0, 0), 1)
    for p in refined_pts:
        cv2.circle(tmp, (int(round(p[0])), int(round(p[1]))), 4, (0, 0, 255), -1)
    cv2.imshow(win_name, tmp)   # Takoj ob kliku pokazemo novo sliko

def refine_point(x, y):
    """
    Vrne natancno tocko okoli (x,y) ali None.
    """
    # Obrezemo sliko, poskrbimo, da se nahajamo znotraj mej posnetka
    x1 = max(x - search_w // 2, 0)
    y1 = max(y - search_w // 2, 0)
    x2 = min(x + search_w // 2, gray_prev.shape[1] - 1)
    y2 = min(y + search_w // 2, gray_prev.shape[0] - 1)

    roi = gray_prev[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    # S pomocjo vgrajene funkcije najdemo oglisce oz. kot
    corners = cv2.goodFeaturesToTrack(
        roi,
        maxCorners = 1,
        qualityLevel = 0.001,
        minDistance = 2,
        blockSize = 7
    )
    if corners is None:
        return None

    # Prevedemo nazaj na globalne koordinate
    cx = float(corners[0, 0, 0] + x1)
    cy = float(corners[0, 0, 1] + y1)

    # Izpopolnimo na sub-pikselno raven
    corner = np.array([[cx, cy]], dtype=np.float32)  # Oblike (1,2)
    cv2.cornerSubPix(
        gray_prev,
        corner,
        winSize = (5, 5),
        zeroZone = (-1, -1),
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
    )
    return corner[0]          # Primer: [1123.4, 482.9]

def on_mouse(event, x, y, flags, param):
    """
    Pomozna funkcija, ki registrira pritisk miske.
    Zanima jo le klik levega gumba.
    """
    global clicked, refined_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked.append((x, y))
        pt = refine_point(x, y)
        if pt is not None:
            refined_pts.append(pt)
        draw_overlay()

# Poklicemo funkcije za risanje tock in klikanje
win_name = "Select points - ENTER when done | Backspace=undo | r=reset"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(win_name, on_mouse)
draw_overlay()

# Zanka za razlicne funkcionalnosti ob zbiranju
while True:
    key = cv2.waitKey(20) & 0xFF
    if key in (13, 10):                 # Enter ➔ koncano
        if len(refined_pts) == 0:
            print("No points selected – exiting")
            cap.release(); cv2.destroyAllWindows(); exit()
        break
    elif key == 8:                      # Backspace ➔ zbrisemo zadnjo tocko
        if clicked:
            clicked.pop()
            refined_pts.pop()
            draw_overlay()
    elif key in (ord('r'), ord('R')):   # r ➔ ponastavimo vse
        clicked.clear(); refined_pts.clear(); draw_overlay()

cv2.destroyWindow(win_name)     # Zapre okno

p0 = np.float32(refined_pts).reshape(-1,1,2)
print(f"{len(p0)} point(s) selected (sub-pixel refined):", [tuple(np.round(p[0],2)) for p in p0])   # Dobimo informacijo o stevilu in pozicijah izbranih tock

# Generiramo nakljucno barvo za vsako tocko v sledi gibanja
colors = np.random.randint(0, 255, (p0.shape[0], 3), dtype=np.uint8)
mask = np.zeros_like(first)    # Priprava za tocke    
frame_idx = 0
trajectory_rows = []           # Za .csv


# 3) Glavna zanka
while True:
    # Konca se le, ko zmanjka slicic
    if max_frames is not None and frame_idx >= max_frames:
        break

    ret, frame = cap.read()
    if not ret:
        break

    # Pripravi naslednjo slicico v crno-belo
    gray_next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Pozenemo piramidni LK
    p1, st, err = cv2.calcOpticalFlowPyrLK(
        gray_prev, gray_next,
        p0, None,
        winSize=lk_win,
        maxLevel=lk_maxlevel,
        criteria=lk_crit)

    # Obdrzimo le tocke, ki smo jim lahko sledili
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # Narisemo sledi gibanja in pripravimo .csv
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        cv2.line(mask, (int(a), int(b)), (int(c), int(d)),
                 colors[i].tolist(), 2)
        cv2.circle(frame, (int(a), int(b)), 4, colors[i].tolist(), -1)
        trajectory_rows.append((frame_idx, i, float(a), float(b)))

    img_overlay = cv2.add(frame, mask)
    cv2.imshow("Lucas-Kanade tracking  (Esc to quit)", img_overlay)
    if writer:
        writer.write(img_overlay)

    if cv2.waitKey(1) & 0xFF == 27:   # Esc za predcasno prekinitev
        break

    # Naslednja iteracija
    gray_prev = gray_next.copy()
    p0 = good_new.reshape(-1, 1, 2)
    frame_idx += 1


# Koncno urejanje in shranjevanje
cap.release()
if writer:
    writer.release()
cv2.destroyAllWindows()

# Zapisemo izhodni .csv
if csv_out:
    csv_out = Path(csv_out)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_out, "w", newline="") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(["frame", "point_id", "x", "y"])
        writer_csv.writerows(trajectory_rows)
    print(f"Saved {len(trajectory_rows)} rows to {csv_out}")

print("Done.")

