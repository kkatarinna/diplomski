import numpy as np
import cv2
from collections import Counter
from tqdm import tqdm
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie2000


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

### POMOCNE FUNKCIJE

def brightest_color(color1, color2, color3, color4, color5):
    def brightness(rgb):
        r, g, b = rgb
        return 0.299 * r + 0.587 * g + 0.114 * b #standardna formula za svetlinu
    colors = [color1, color2, color3, color4, color5]
    return max(colors, key=brightness)

def rgb_to_lab(rgb_color):
    rgb = np.array([[rgb_color]], dtype=np.uint8)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    return tuple(map(float, lab[0][0]))

def get_closest_monk_tone(rgb_color):
    input_lab = rgb_to_lab(rgb_color)
    input_lab_color = LabColor(input_lab[0], input_lab[1], input_lab[2])
    min_diff = float('inf')
    closest_tone_index = -1
    for idx, (monk_l, monk_a, monk_b) in enumerate(monk_cv2_lab):
        monk_lab_color = LabColor(monk_l, monk_a, monk_b)
        diff = delta_e_cie2000(input_lab_color, monk_lab_color)
        if diff < min_diff:
            min_diff = diff
            closest_tone_index = idx
    return closest_tone_index + 1, monk_cv2_lab[closest_tone_index]

def colorparallel(image):
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)

    cv2.setNumThreads(4)
    slic = cv2.ximgproc.createSuperpixelSLIC(image, algorithm=cv2.ximgproc.SLICO, region_size=70, ruler=30)
    slic.iterate(5)
    labels = slic.getLabels()
    num_labels = slic.getNumberOfSuperpixels()
    color_counts = Counter()
    for label in range(num_labels):
        mask = (labels == label)
        mean_color = np.mean(image[mask], axis=0)
        color_counts[tuple(mean_color)] += np.sum(mask)
    dominant_color = max(color_counts, key=color_counts.get)
    top_5_colors = [color for color, _ in color_counts.most_common(5)]
    return dominant_color, top_5_colors
