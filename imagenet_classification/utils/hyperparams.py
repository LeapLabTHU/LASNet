
def get_hyperparams(args, test_code=0):
    if not test_code:
        if args.hyperparams_set_index == 1: 
            args.epochs = 100
            args.start_eval_epoch = 90
            args.batch_size = 256 * 1
            ### data transform
            args.autoaugment = False
            args.colorjitter = False
            args.change_light = False
            ### optimizer
            args.optimizer = 'SGD'
            args.lr = 0.02 * args.batch_size / 256
            args.momentum = 0.9
            args.weigh_decay_apply_on_all = True 
            args.weight_decay = 5e-5
            args.nesterov = True
            ### criterion
            args.labelsmooth = 0.0
            ### lr scheduler
            args.scheduler = 'cosine'
            args.warmup_epoch = 0
            args.warmup_lr = args.lr * 0.1
            ### Mixup
            args.mixup = 0.0
            return args

        elif args.hyperparams_set_index == 2: 
            args.epochs = 100
            args.start_eval_epoch = 90
            args.batch_size = 256 * 2
            ### data transform
            args.autoaugment = False
            args.colorjitter = False
            args.change_light = False
            ### optimizer
            args.optimizer = 'SGD'
            args.lr = 0.02 * args.batch_size / 256
            args.momentum = 0.9
            args.weigh_decay_apply_on_all = True 
            args.weight_decay = 5e-5
            args.nesterov = True
            ### criterion
            args.labelsmooth = 0.0
            ### lr scheduler
            args.scheduler = 'cosine'
            args.warmup_epoch = 0
            args.warmup_lr = args.lr * 0.1
            ### Mixup
            args.mixup = 0.0
            return args

        elif args.hyperparams_set_index == 3: 
            args.epochs = 100
            args.start_eval_epoch = 90
            args.batch_size = 256 * 4
            ### data transform
            args.autoaugment = False
            args.colorjitter = False
            args.change_light = False
            ### optimizer
            args.optimizer = 'SGD'
            args.lr = 0.02 * args.batch_size / 256
            args.momentum = 0.9
            args.weigh_decay_apply_on_all = True 
            args.weight_decay = 5e-5
            args.nesterov = True
            ### criterion
            args.labelsmooth = 0.0
            ### lr scheduler
            args.scheduler = 'cosine'
            args.warmup_epoch = 0
            args.warmup_lr = args.lr * 0.1
            ### Mixup
            args.mixup = 0.0
            return args

        elif args.hyperparams_set_index == 4: 
            args.epochs = 100
            args.start_eval_epoch = 90
            args.batch_size = 256 * 8
            ### data transform
            args.autoaugment = False
            args.colorjitter = False
            args.change_light = False
            ### optimizer
            args.optimizer = 'SGD'
            args.lr = 0.02 * args.batch_size / 256
            args.momentum = 0.9
            args.weigh_decay_apply_on_all = True 
            args.weight_decay = 5e-5
            args.nesterov = True
            ### criterion
            args.labelsmooth = 0.0
            ### lr scheduler
            args.scheduler = 'cosine'
            args.warmup_epoch = 0
            args.warmup_lr = args.lr * 0.1
            ### Mixup
            args.mixup = 0.0
            return args
    else:
        args.epochs = 90
        args.start_eval_epoch = 0
        args.batch_size = 128
        ### data transform
        args.autoaugment = False
        args.colorjitter = True
        args.change_light = True
        ### optimizer
        args.optimizer = 'SGD'
        args.lr = 0.05
        args.momentum = 0.9
        args.weigh_decay_apply_on_all = False 
        args.weight_decay = 1e-4
        args.nesterov = True
        ### criterion
        args.labelsmooth = 0
        ### lr scheduler
        args.scheduler = 'multistep'
        args.lr_decay_rate = 0.1
        args.lr_decay_step = 30
        ### Mixup
        args.mixup = 0.0

        return args