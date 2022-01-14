import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose([
    # reszie
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.OneOf([
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(p=0.2, shift_limit=0.0625, scale_limit=0.2, rotate_limit=45),
        A.Transpose(p=0.5)
    ]),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=1023.0, p=1.0),
    ToTensorV2(),
])
# train_transform = A.Compose([
#     # 
#     A.ToFloat(max_value=65535.0),
#     A.Resize(224, 224),
#     A.HorizontalFlip(p=0.5),
#     A.RandomBrightnessContrast(p=0.5),
#     A.OneOf([
#         A.Transpose(p=0.5),
#         A.VerticalFlip(p=0.5),
#         A.RandomRotate90(p=0.5),
#         A.ShiftScaleRotate(p=0.5,rotate_limit=30),
#     ],p=1.0),
#     A.HueSaturationValue(p=0.5),
#     A.OneOf([
#         A.MotionBlur(p=0.2),
#         A.MedianBlur(blur_limit=3, p=0.1),
#         A.Blur(blur_limit=3, p=0.1),
#     ], p=0.2),
#     A.OneOf([
#         A.OpticalDistortion(p=0.3),
#         A.GridDistortion(p=0.1),
#     ], p=0.25),
#     A.FromFloat(max_value=65535.0),
# ])

val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=1023.0, p=1.0),
    ToTensorV2(p=1.0),
],p=1.)

infer_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=1023.0, p=1.0),
    ToTensorV2(p=1.0),
],p=1.)