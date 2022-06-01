import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose([
    # reszie
    A.Resize(512, 512),
    A.HorizontalFlip(p=0.5),
    A.OneOf([
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.5),
        A.ShiftScaleRotate(p=0.5, shift_limit=0.0625, scale_limit=0.2, rotate_limit=20),
        #A.CoarseDropout(p=0.2),
        A.Transpose(p=0.5)
    ]),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2(),
])

# train_transform = A.Compose([
#     # reszie
#     # A.Resize(int(704 * 1.25), int(1280 * 1.25)),
#     A.Resize(704, 1280),
#     # A.RandomResizedCrop(704, 1280, scale=(0.5, 1.0)),
#     A.OneOf([
#         # A.VerticalFlip(p=0.5),
#         A.HorizontalFlip(p=0.5),
#         # A.RandomRotate90(p=0.5),
#         # A.Transpose(p=0.5),
#         A.RandomBrightnessContrast(p=0.2, brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2)),
#         A.HueSaturationValue(p=0.2, hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2),
#         A.ShiftScaleRotate(p=0.2, shift_limit=0.0625, scale_limit=0.2, rotate_limit=20),
#         A.CoarseDropout(p=0.2),
#         A.Cutout(p=0.2, max_h_size=16, max_w_size=16, fill_value=(0., 0., 0.), num_holes=16),
#     ]),
#     A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#     ToTensorV2(),
# ])


# train_transform = A.Compose([
#     #color transforms
#     # A.OneOf([
#     #     A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.2,p=1),
#     #     A.RandomGamma(gamma_limit=(70, 130),p=1),
#     #     A.ChannelShuffle(p=0.2),  
#     # ],
#     # p=0.8, 
#     # ),
#     # #distortion
#     # A.OneOf(
#     #     [
#     #     A.ElasticTransform(p=1),
#     #     A.OpticalDistortion(p=1),
#     #     A.GridDistortion(p=1),
#     #     A.IAAPerspective(p=1),      
#     #     ],
#     #     p=0.2, 
#     # ),
#     # #noise transforms
#     # A.OneOf([
#     #     A.GaussNoise(p=1),
#     #     A.IAASharpen(p=1),
#     #     A.MultiplicativeNoise(p=1),
#     #     A.GaussianBlur(p=1),  
#     # ],
#     # p=0.2,  
#     # ),
#     A.Resize(800, 800),
#     # 
#     A.OneOf([
#         # A.VerticalFlip(p=0.5),
#         A.HorizontalFlip(p=0.5),
#         # A.RandomRotate90(p=0.5),
#         A.Transpose(p=0.5)
#     ]),

#     A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#     ToTensorV2(),
# ])

val_transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2(),
])

infer_transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5,0.5)),
    ToTensorV2(),
])