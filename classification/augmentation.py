import albumentations as A

def get_transforms(image_size):
    transforms_train = A.Compose([
        A.Transpose(p=0.5),
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=-0.3, contrast_limit=0.4, p=0.5),

        A.OneOf([
            A.MotionBlur(blur_limit=5),
            A.MedianBlur(blur_limit=5),
            A.GaussianBlur(blur_limit=5),
        ], p=0.7),

        A.OneOf([
            A.OpticalDistortion(distort_limit=1.0),
            A.GridDistortion(num_steps=5, distort_limit=1.0),
            A.ElasticTransform(alpha=3),
        ], p=0.7),

        A.CLAHE(clip_limit=4.0, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),

        A.Affine(
                scale=(0.8, 1.2),      # Zoom in/out by 80-120%
                rotate=(-15, 15),      # Rotate by -15 to +15 degrees
                p=0.7
            ),
            A.Resize(image_size, image_size),
            #A.Normalize() #Ovo zakomentarisi ukoliko zelis da izdvojis boju koze
        ])

    transforms_val = A.Compose([
        A.Resize(image_size, image_size),
        #A.Normalize()
    ])

    return transforms_train, transforms_val