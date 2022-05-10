class Config(object):
    backbone = 'resnext101_32x4d'#resnet50, resnext101_32x4d, pnasnet5large
    num_classes = 3097#
    pooling = 'GeM'#One of ('SPoC', 'MAC', 'RMAC', 'GeM', 'Rpool', 'Flatten', 'CompactBilinearPooling')
    loss_module = 'arcface'# One of ('arcface', 'cosface', 'softmax')
    criterion = 'CrossEntropyLoss'#focal_loss/CrossEntropyLoss
    fc_dim=512
    #
    train_batch_size = 32 # batch size
    val_batch_size = 8
    test_batch_size = 8
    optimizer = 'sgd'
    device = "cuda"  # cuda  or32pu
    gpu_id = [0,1]
    num_workers = 8  # how many workers for loading data
    max_epoch = 100
    #
    momentum = 0.9
    lr = 1.05e-3  # adam 0.00001
    lr_step=10
    lr_decay_epoch = 5
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 5e-4
    
    lr_scheduler='cosine'#cosine or stepLR
    #
    input_size = 320
    crop_size=224
    val_interval = 1
    print_interval = 100
    save_interval = 5
    #
    data_root = '/opt/data/private/qyl/train_data'
    train_list = '/opt/data/private/qyl/xceptionForRetri/dataset/train.txt'
    val_list = '/opt/data/private/qyl/xceptionForRetri/dataset/val.txt'

    gallery_dir = '/opt/data/private/qyl/test_data_A/gallery'
    query_dir = '/opt/data/private/qyl/test_data_A/query'

    checkpoints_dir = 'checkpoints'
    test_model_dir = 'checkpoints/xception/xception_4.pth'
