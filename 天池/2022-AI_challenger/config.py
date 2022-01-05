args_resnet = {
    'epochs': 185,#{2:[5,13,29,61],3:[8,20,44,92]}
    'optimizer_name': 'SGD',#'AdamW',#'SGD
    'optimizer_hyperparameters': {
        'lr': 0.01,#0.001,
        'momentum': 0.9,
        'weight_decay': 5e-4
    },
    'scheduler_name': 'CosineAnnealingWarmRestarts',
    'scheduler_hyperparameters': {
        #'T_max': 200
        'T_0':3,
        'T_mult':2,
        'eta_min':1e-5,
    },
    'batch_size': 128,
}
args_densenet = {
    'epochs': 185,
    'optimizer_name': 'SGD',#'AdamW',#'SGD',
    'optimizer_hyperparameters': {
        'lr': 0.01,#0.001,
        'momentum': 0.9,
        'weight_decay': 5e-4
    },
    'scheduler_name': 'CosineAnnealingWarmRestarts',
    'scheduler_hyperparameters': {
        'T_0':3,
        'T_mult':2,
        'eta_min':1e-5,
    },
    'batch_size': 128,
}