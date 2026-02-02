import numpy as np
import cv2
import os
from classification.classification import colorparallel, load_images_from_folder
from tqdm import tqdm
import matplotlib.pyplot as plt

# mean
# skin tone - 19
# min
# min obrnuto - max
# mean = 19
# min ()
monk_cv2_hsv = [
    (26.7/2, 12.3*255/100, 14.3*255/100),
    (26.3/2, 16*255/100, 19.6*255/100),
    (17.7/2, 29.7*255/100, 29*255/100),
    (23.8/2, 32*255/100, 38.6*255/100),
    (32.4/2, 30.1*255/100, 48.2*255/100),
    (36/2, 44.8*255/100, 71.6*255/100),
    (40/2, 53.3*255/100, 82.4*255/100),
    (40/2, 70.9*255/100, 89.2*255/100),
    (30/2, 50*255/100, 90.6*255/100),
    (30/2, 50*255/100, 92.9*255/100)
]


def rgb_to_hsv(rgb_color):
    rgb = np.array([[rgb_color]], dtype=np.uint8)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return tuple(map(float, hsv[0][0]))

def hsv_to_rgb(hsv_color):
    hsv = np.array([[hsv_color]], dtype=np.uint8)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return tuple(map(int, rgb[0][0]))

def change_image_hsv(image, dH, dS, dV):
    img = image.copy().astype(np.int16)

    # H kanal (0–180)
    img[:, :, 0] = np.clip(img[:, :, 0] + dH, 0, 180)

    # S i V kanali (0–255)
    img[:, :, 1] = np.clip(img[:, :, 1] + dS, 0, 255)
    img[:, :, 2] = np.clip(img[:, :, 2] + dV, 0, 255)

    return img.astype(np.uint8)

def get_image_features_hsv(image_hsv):
    H_channel = image_hsv[:, :, 0]
    S_channel = image_hsv[:, :, 1]
    V_channel = image_hsv[:, :, 2]

    features = {
        'H_mean': np.mean(H_channel),
        'H_std': np.std(H_channel),
        'S_mean': np.mean(S_channel),
        'S_std': np.std(S_channel),
        'V_mean': np.mean(V_channel),
        'V_std': np.std(V_channel),
    }
    return features

def get_skin_color(image_bgr):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    dominant_color, top_colors, brightest_color, sp_image = colorparallel(
        image_rgb,
        region_size=70,
        ruler=30
    )

    return rgb_to_hsv(brightest_color)


def change_color_based_on_skin(image_hsv, skin_hsv, target_hsv):
    dH = skin_hsv[0] - target_hsv[0]
    dS = skin_hsv[1] - target_hsv[1]
    dV = skin_hsv[2] - target_hsv[2]

    modified_image_hsv = change_image_hsv(image_hsv, -dH, -dS, -dV)
    return modified_image_hsv

def change_color_based_on_mean(image_hsv, mean_hsv, target_hsv):
    dH = mean_hsv[0] - target_hsv[0]
    dS = mean_hsv[1] - target_hsv[1]
    dV = mean_hsv[2] - target_hsv[2]

    modified_image_hsv = change_image_hsv(image_hsv, -dH, -dS, -dV)
    return modified_image_hsv

if __name__ == "__main__":
    input_images_path = "data/trening"
    output_image_path = "skin_test/images_hsv_variations"

    image_paths = load_images_from_folder(input_images_path)
    for img_path in tqdm(image_paths, desc=f"sledeca slika"):
        image_name = os.path.splitext(os.path.basename(img_path))[0]

        image_bgr = cv2.resize(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2HSV), (300, 300))
        
        if image_bgr is None:
            continue

        image_features = get_image_features_hsv(image_bgr)
        skin_color = (image_features['H_mean'], image_features['S_mean'], image_features['V_mean'])
        print(skin_color)
        for mst in monk_cv2_hsv:
            if not os.path.exists(f"{output_image_path}/{image_name}"):
                os.mkdir(f"{output_image_path}/{image_name}")
            modified_image_bgr = change_color_based_on_mean(image_bgr, skin_color, mst)
            cv2.imwrite(f"{output_image_path}/{image_name}/monk_{monk_cv2_hsv.index(mst)}.png", cv2.cvtColor(modified_image_bgr, cv2.COLOR_HSV2BGR))
        
    # ds = [100,110,120,130,140,150,160,170,180,190,200]
    # dv = [-100,-110,-120,-130,-140,-150]

    # for v in dv:
    #     if not os.path.exists(f"{output_image_path}/V_{v}"):
    #         os.mkdir(f"{output_image_path}/V_{v}")
    #     for s in ds:
    #         modified_image_hsv = change_image_hsv(image_hsv, dH=-60, dS=s, dV=v)
    #         img_rgb = cv2.cvtColor(modified_image_hsv, cv2.COLOR_HSV2BGR)
    #         cv2.imwrite(f"{output_image_path}/V_{v}/SV_{s}.png", img_rgb)