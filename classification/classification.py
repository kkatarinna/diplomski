import numpy as np
import cv2
from collections import Counter
from tqdm import tqdm
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie2000
import matplotlib.pyplot as plt
import os

## PARAMETRI
REGION_SIZE = 70
RULER = 30


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

# def colorparallel(image):
#     image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
#     cv2.setNumThreads(4)
#     slic = cv2.ximgproc.createSuperpixelSLIC(image, algorithm=cv2.ximgproc.SLICO, region_size=70, ruler=30)
#     slic.iterate(5)
#     labels = slic.getLabels()
#     num_labels = slic.getNumberOfSuperpixels()
#     color_counts = Counter()
#     for label in range(num_labels):
#         mask = (labels == label)
#         mean_color = np.mean(image[mask], axis=0)
#         color_counts[tuple(mean_color)] += np.sum(mask)
#     dominant_color = max(color_counts, key=color_counts.get)
#     top_5_colors = [color for color, _ in color_counts.most_common(5)]
#     return dominant_color, top_5_colors

def lab_to_rgb(lab_color):
    L, a, b = lab_color

    lab = np.zeros((1, 1, 3), dtype=np.uint8)
    lab[0, 0, 0] = np.clip(L, 0, 255)
    lab[0, 0, 1] = np.clip(a + 128, 0, 255)
    lab[0, 0, 2] = np.clip(b + 128, 0, 255)

    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)[0, 0]

    return rgb

def colorparallel(image, region_size=70, ruler=30, resize_dim=256):
    image = cv2.resize(image, (resize_dim, resize_dim), interpolation=cv2.INTER_AREA)

    slic = cv2.ximgproc.createSuperpixelSLIC(
        image,
        algorithm=cv2.ximgproc.SLICO,
        region_size=region_size,
        ruler=ruler
    )
    slic.iterate(5)

    labels = slic.getLabels()
    num_labels = slic.getNumberOfSuperpixels()

    color_counts = Counter()
    superpixel_mean_image = np.zeros_like(image, dtype=np.uint8)

    brightest_color_global = None
    max_brightness = -1

    for label in range(num_labels):
        mask = (labels == label)
        mean_color = np.mean(image[mask], axis=0)
        mean_color_uint8 = mean_color.astype(np.uint8)

        color_counts[tuple(mean_color)] += np.sum(mask)
        superpixel_mean_image[mask] = mean_color_uint8

        r, g, b = mean_color
        brightness = 0.299 * r + 0.587 * g + 0.114 * b

        if brightness > max_brightness:
            max_brightness = brightness
            brightest_color_global = tuple(mean_color)

    dominant_color = max(color_counts, key=color_counts.get)
    top_5_colors = [color for color, _ in color_counts.most_common(5)]

    return dominant_color, top_5_colors, brightest_color_global, superpixel_mean_image


def save_visualization(
    image_rgb,
    superpixel_image,
    brightest_color,
    monk_lab,
    monk_index,
    save_path,
    original_size=244,
    patch_size=60
):
    # Resize slike
    original_resized = cv2.resize(image_rgb, (original_size, original_size))
    superpixel_resized = cv2.resize(superpixel_image, (original_size, original_size))

    # Brightest color patch
    brightest_patch = np.ones(
        (patch_size, patch_size, 3), dtype=np.uint8
    ) * np.array(brightest_color, dtype=np.uint8)

    # Monk LAB → RGB (ISPRAVNO)
    monk_rgb = lab_to_rgb(monk_lab)

    monk_patch = np.ones(
        (patch_size, patch_size, 3), dtype=np.uint8
    ) * np.array(monk_rgb, dtype=np.uint8)

    # Padding da sve bude iste visine
    def pad_patch(patch):
        pad_h = original_size - patch.shape[0]
        return np.pad(
            patch,
            ((pad_h // 2, pad_h - pad_h // 2), (0, 0), (0, 0)),
            mode="constant",
            constant_values=255
        )

    brightest_patch = pad_patch(brightest_patch)
    monk_patch = pad_patch(monk_patch)

    # Spajanje
    combined = np.hstack([
        original_resized,
        superpixel_resized,
        brightest_patch,
        monk_patch
    ])

    # ⬇️ Tekst: Monk index
    cv2.putText(
        combined,
        f"Monk {monk_index}",
        (original_size * 3 + 5, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2,
        cv2.LINE_AA
    )

    cv2.imwrite(save_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))



def process_folder_with_multiple_params(
    input_folder,
    base_output_dir,
    superpixel_params
):
    image_paths = load_images_from_folder(input_folder)

    for region_size, ruler in superpixel_params:
        # ⬇️ folder za konkretne parametre
        param_output_dir = make_param_folder(
            base_output_dir,
            region_size,
            ruler
        )

        for img_path in tqdm(image_paths, desc=f"RS={region_size}, R={ruler}"):
            image_name = os.path.splitext(os.path.basename(img_path))[0]

            image_bgr = cv2.imread(img_path)
            if image_bgr is None:
                continue

            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            dominant_color, top_colors, brightest_color_global, sp_image = colorparallel(
                image_rgb,
                region_size=region_size,
                ruler=ruler
            )

            monk_index, monk_lab = get_closest_monk_tone(brightest_color_global)
            save_name = f"{image_name}_monk_{monk_index}.png"
            save_path = os.path.join(param_output_dir, save_name)

            save_visualization(
                image_rgb,
                sp_image,
                brightest_color_global,
                monk_lab,
                monk_index,
                save_path
            )


def make_param_folder(base_dir, region_size, ruler):
    folder_name = f"RS_{region_size}_R_{ruler}"
    folder_path = os.path.join(base_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

def load_images_from_folder(folder_path, extensions=(".jpg", ".png", ".jpeg")):
    return [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(extensions)
    ]


if __name__ == "__main__":
    ## TESTIRANJE
    input_folder = "skin_test/final test7H/V_-140"   # folder sa ~20 slika
    base_output_dir = "classification/superpixel_test/results_augmented7H/V-140"

    superpixel_params = [
        (40, 20),
        (70, 30),
        (100, 40),
        (20, 60)
    ]

    process_folder_with_multiple_params(
        input_folder,
        base_output_dir,
        superpixel_params
    )