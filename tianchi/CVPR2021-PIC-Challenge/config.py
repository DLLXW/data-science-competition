
#
class Config():
    seed = 2021
    input_size = 256
    landmarks = 318
    max_epoch = 70
    batch_size = 8
    num_workers = 4
    print_interval=20
    save_interval=4
    backbone = 'tf_efficientnet_b3'#'swin_base_patch4_window12_384'#tf_efficientnet_b2 pfld
    device = 'cuda'
    root='/home/limzero/qyl/3Dface/data_crop/'
    train_list_dir='dataset/train_crop.txt'
    val_list_dir='dataset/val_crop.txt'
    checkpoints_dir='ckpt/'
    optimizer='adam'
    scheduler_params = {
        "lr_start": 3e-4,
        "lr_max": 1e-5 * batch_size,     # 1e-5 * 32 (if batch_size(=32) is different then)
        "lr_min": 1e-6,
        "lr_ramp_ep": 5,
        "lr_sus_ep": 0,
        "lr_decay": 0.8,
    }