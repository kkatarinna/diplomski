import numpy as np
import cv2
from collections import Counter
from tqdm import tqdm
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie2000
import matplotlib.pyplot as plt
import os

monk_cv2_lab = [
    (240.238, 1.909, 6.886),
    (235.301, 2.617, 9.246),
    (237.382, 0.274, 18.04),
    (223.311, 0.583, 22.54),
    (198.65, 4.408, 29.383),
    (140.612, 9.884, 33.96),
    (108.299, 15.653, 26.073),
    (78.229, 14.817, 16.935),
    (53.726, 3.416, 7.574),
    (37.255, 1.882, 4.477)
]

def lab_to_rgb(lab_color):
    L, a, b = lab_color

    lab = np.zeros((1, 1, 3), dtype=np.uint8)
    lab[0, 0, 0] = np.clip(L, 0, 255)
    lab[0, 0, 1] = np.clip(a + 128, 0, 255)
    lab[0, 0, 2] = np.clip(b + 128, 0, 255)

    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)[0, 0]

    return tuple(rgb / 255.0)

# Konverzija u RGB
rgb_colors = [lab_to_rgb(lab) for lab in monk_cv2_lab]

# Plot
fig, ax = plt.subplots(figsize=(8, 2))
for i, rgb in enumerate(rgb_colors):
    ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=rgb))

ax.set_xlim(0, len(rgb_colors))
ax.set_ylim(0, 1)
ax.set_xticks(range(len(rgb_colors)))
ax.set_yticks([])
ax.set_title("Monk Skin Tone Colors (LAB cv2 → RGB)")

plt.show()