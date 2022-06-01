class Config(object):
    backbone = 'None'#None mobilenetv3_small_075
    model_name='SpatailTimeNN'#
    use_lstm_decoder=False
    mode='all_data'#使用SODA微调模型
    num_classes = 24 #
    loss = 'L1'#MSE
    use_frt=[True,True,True,True]#['sst','t300','ua','va']
    scale_factor = 5
    train_batch_size = 32  # batch size
    val_batch_size = 32
    optimizer = 'adamw'#adamw
    scheduler='cosine'#cosine/step
    T_0=3
    T_mult=2
    lr = 1e-4  # adam 0.00001
    momentum = 0.9
    device = "cuda"  # cuda  or cpu
    gpu_id = [0,1]
    num_workers = 4  # how many workers for loading data
    max_epoch = 21
    lr_decay_epoch = 10
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 5e-4
    val_interval = 1
    print_interval=100
    min_save_epoch=0
    save_epoch=1
    #
    p=0.8
    extend=1.
    #
    root='../enso_round1_train_20210201/'
    checkpoints_dir = 'ckpt/'
    pretrain=None#'docker_submit_cnnlstm/cv331_lb337/SpatailTimeNN_best.pth'#
#