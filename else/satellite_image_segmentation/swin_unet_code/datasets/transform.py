import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose([
    # reszie
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.OneOf([
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.2, brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2)),
        #A.HueSaturationValue(p=0.2, hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2),
        A.ShiftScaleRotate(p=0.2, shift_limit=0.0625, scale_limit=0.2, rotate_limit=20),
        A.CoarseDropout(p=0.2),
        A.Transpose(p=0.5)
    ]),
    A.Normalize(mean=(0.5, 0.5, 0.5,0.5), std=(0.5, 0.5, 0.5,0.5)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.5, 0.5, 0.5,0.5), std=(0.5, 0.5, 0.5,0.5)),
    ToTensorV2(),
])

infer_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.5, 0.5, 0.5,0.5), std=(0.5, 0.5,0.5,0.5)),
    ToTensorV2(),
])