class Config(object):
    backbone = 'xception'
    num_classes = 4 #
    use_arcLoss = False
    metric = 'arc_margin'
    easy_margin = False
    loss = 'focal_loss'#focal_loss/CrossEntropyLoss
    feature_dimension=128
    #
    input_size = 800
    train_batch_size = 6  # batch size
    val_batch_size = 2
    test_batch_size = 1
    optimizer = 'sgd'
    lr_scheduler='cosine'
    lr = 0.5e-3  # adam 0.00001
    MOMENTUM = 0.9
    device = "cuda"  # cuda  or cpu
    gpu_id = [0,1]
    num_workers = 4  # how many workers for loading data
    max_epoch = 30
    lr_decay_epoch = 10
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 5e-4
    val_interval = 1
    print_interval = 100
    save_interval = 2
    min_save_epoch=5
    #
    log_dir = 'log/'
    raw_data_dir = '/home/admins/qyl/gaode_classify/dataset/amap_traffic_final_train_data'
    raw_json = '/home/admins/qyl/gaode_classify/dataset/amap_traffic_final_train_0906.json'
    train_list='/home/admins/qyl/gaode_classify/dataset/train_concat.txt'
    val_list='/home/admins/qyl/gaode_classify/dataset/val_concat.txt'
    #
    raw_test_dir=''
    raw_test_json='/home/admins/qyl/gaode_classify/dataset/amap_traffic_annotations_b_test_0828.json'
    testConcat_dir='/home/admins/qyl/gaode_classify/dataset/testBConcat'
    result_json='/home/admins/qyl/gaode_classify/prediction_result/result.json'
    #
    concat_data_dir='/home/admins/qyl/gaode_classify/dataset/amap_traffic_final_train_data_concat'
    trainValConcat_dir= '/home/admins/qyl/gaode_classify/dataset/trainValConcat'
    trainValSingle_dir='/home/admins/qyl/gaode_classify/dataset/trainValSingle'
    checkpoints_dir = 'checkpoints/'
    test_model_dir = 'checkpoints/xception/xception_24.pth'
