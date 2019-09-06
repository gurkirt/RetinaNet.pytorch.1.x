import os
import shutil
import socket
import getpass
def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")
    
def get_class_names(dataset):
    classes = {
        'coco':['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'],
        'voc':['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'],
    }
    return classes[dataset]
    
def copy_source(source_dir):
    if not os.path.isdir(source_dir):
        os.system('mkdir -p ' + source_dir)
    
    for dirpath, dirs, files in os.walk('./', topdown=True):
        for file in files:
            if file.endswith('.py'): #fnmatch.filter(files, filepattern):
                shutil.copy2(os.path.join(dirpath, file), source_dir)

def set_args(args, iftest='train'):

    if iftest == 'test':
        args.eval_iters = [int(val) for val in args.eval_iters.split(',')]
    else:
        args.milestones = [int(val) for val in args.milestones.split(',')]
        args.gammas = [float(val) for val in args.gammas.split(',')]

    args.dataset = args.dataset.lower()
    args.basenet = args.basenet.lower()

    if args.dataset == 'coco':
        args.train_sets = ['train2017']
        args.val_sets = ['val2017']
    else:
        args.train_sets = ['train2007', 'val2007', 'train2012', 'val2012']
        args.val_sets = ['test2007']

    args.means =[0.485, 0.456, 0.406]
    args.stds = [0.229, 0.224, 0.225]

    username = getpass.getuser()
    hostname = socket.gethostname()
    args.hostname = hostname
    args.user = username

    print('\n\n ', username, ' is using ', hostname, '\n\n')
    if username == 'gurkirt':
        args.model_dir = '/mnt/mars-gamma/global-models/pytorch-imagenet/'
        if hostname == 'mars':
            args.data_root = '/mnt/mars-fast/datasets/'
            args.save_root = '/mnt/venus-alpha/'
            args.vis_port = 8097
        elif hostname == 'venus':
            args.data_root = '/mnt/venus-fast/datasets/'
            args.save_root = '/mnt/venus-alpha/'
            args.vis_port = 8095
        elif hostname in ['sun','jupiter']:
            args.data_root = '/mnt/mercury-fast/datasets/'
            args.save_root = '/mnt/mars-gamma/'
            if hostname in ['sun']:
                args.vis_port = 8096
            else:
                args.vis_port = 8095
        elif hostname == 'mercury':
            args.data_root = '/mnt/mercury-fast/datasets/'
            args.save_root = '/mnt/venus-alpha/'
            args.vis_port = 8098
        else:
            raise('ERROR!!!!!!!! Specify directories')
    
    print('Your working directories are', args.data_root, args.save_root)
    return args

def create_exp_name(args):
    return 'FPN{:d}x{:d}-{:01d}-{:s}-{:s}-hl{:01d}s{:01d}-bn{:d}f{:d}b{:d}-bs{:02d}-{:s}-lr{:06d}-{:s}'.format(
                                            args.min_size, args.max_size, int(args.multi_scale), args.dataset, args.basenet,
                                            args.num_head_layers, args.shared_heads, int(args.fbn), args.freezeupto, int(args.use_bias),
                                            args.batch_size, args.optim, int(args.lr * 1000000), args.loss_type)
# Freeze batch normlisation layers
def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') > -1:
        m.eval()
        if m.affine:
            m.weight.requires_grad = False
            m.bias.requires_grad = False