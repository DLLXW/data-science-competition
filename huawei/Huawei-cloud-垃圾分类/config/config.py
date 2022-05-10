class Config(object):
    backbone = 'res50'
    num_classes = 43 #
    loss = 'CrossEntropyLoss'#focal_loss/CrossEntropyLoss
    #
    input_size = 224
    train_batch_size = 96  # batch size
    val_batch_size = 128
    test_batch_size = 1
    optimizer = 'sgd'
    lr = 1e-3  # adam 0.00001
    MOMENTUM = 0.9
    device = "cuda"  # cuda  or cpu
    gpu_id = [0,1]
    num_workers = 4  # how many workers for loading data
    max_epoch = 20
    lr_decay_epoch = 10
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 5e-4
    val_interval = 1
    print_interval = 100
    save_interval = 2
    min_save_epoch=2
    #
    log_dir = 'log/'
    train_val_data = '/media/ssd/qyl/rubbish/data/train_data'
    raw_json = '/media/ssd/qyl/rubbish/garbage_classify/garbage_classify_rule.json'
    train_list='/media/ssd/qyl/rubbish/dataset/train.txt'
    val_list='/media/ssd/qyl/rubbish/dataset/val.txt'
    #
    checkpoints_dir = 'ckpt/'
