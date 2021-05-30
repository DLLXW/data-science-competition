import albumentations
from albumentations.pytorch.transforms import ToTensorV2


def get_train_transforms(img_size=512):
    return albumentations.Compose([
        albumentations.Resize(img_size, img_size, always_apply=True),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        #albumentations.Rotate(limit=120, p=0.8),
        albumentations.RandomBrightness(limit=(0.09, 0.6), p=0.5),
        albumentations.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
        ),
        ToTensorV2(p=1.0)
    ])

def get_valid_transforms(img_size=512):

    return albumentations.Compose([
        albumentations.Resize(img_size, img_size, always_apply=True),
        albumentations.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
        ),
        ToTensorV2(p=1.0)
    ])