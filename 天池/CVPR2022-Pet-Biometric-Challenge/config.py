
import torch
#
class CFG:
    seed = 54
    img_size = 224
    classes = 6000
    scale = 30
    margin = 0.5
    fc_dim = 512
    epochs = 15
    batch_size = 16
    num_workers = 4
    model_name = 'convnext_small'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scheduler_params = {
        "lr_start": 1e-5,
        "lr_max": 1e-5 * batch_size,     # 1e-5 * 32 (if batch_size(=32) is different then)
        "lr_min": 1e-6,
        "lr_ramp_ep": 5,
        "lr_sus_ep": 0,
        "lr_decay": 0.8,
    }
    model_path='./arcface_224x224_convnext_small.pt'#训练好的模型