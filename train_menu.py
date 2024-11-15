import argparse
from models.create_detection_model import create_model


def parse_opt(default_config=None):
    """
    Parse command line arguments.

    Args:
        default_config (str, optional): default configuration file path.
            Defaults to None.

    Returns:
        dict: parsed command line arguments.
    """

    # Construct the argument parser.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--model',
        default='fasterrcnn_resnet50_fpn_v2',
        choices=create_model.keys(),
        help='name of the model'
    )
    parser.add_argument(
        '--config',
        default=default_config,
        help='path to the data config file'
    )
    parser.add_argument(
        '-d', '--device',
        default='cuda',
        help='computation/training device, default is GPU if GPU present'
    )
    parser.add_argument(
        '-e', '--epochs',
        default=10,
        type=int,
        help='number of epochs to train for'
    )
    parser.add_argument(
        '-j', '--workers',
        default=4,
        type=int,
        help='number of workers for data processing/transforms/augmentations'
    )
    parser.add_argument(
        '-b', '--batch',
        default=4,
        type=int,
        help='batch size to load the data'
    )
    parser.add_argument(
        '--lr',
        default=0.001,
        help='learning rate for the optimizer',
        type=float
    )
    parser.add_argument(
        '-ims', '--imgsz',
        default=640,
        type=int,
        help='image size to feed to the network'
    )
    parser.add_argument(
        '-n', '--name',
        default=None,
        type=str,
        help='training result dir name in outputs/training/, (default res_#)'
    )
    parser.add_argument(
        '-vt', '--vis-transformed',
        dest='vis_transformed',
        action='store_true',
        help='visualize transformed images fed to the network'
    )
    parser.add_argument(
        '--mosaic',
        default=0.0,
        type=float,
        help='probability of applying mosaic, (default, always apply)'
    )
    parser.add_argument(
        '-uta', '--use-train-aug',
        dest='use_train_aug',
        action='store_true',
        help='whether to use train augmentation, blur, gray, \
              brightness contrast, color jitter, random gamma \
              all at once'
    )
    parser.add_argument(
        '-w', '--weights',
        default=None,
        type=str,
        help='path to model weights if using pretrained weights'
    )
    parser.add_argument(
        '-r', '--resume-training',
        dest='resume_training',
        action='store_true',
        help='whether to resume training, if true, \
            loads previous training plots and epochs \
            and also loads the optimizer state dictionary'
    )
    parser.add_argument(
        '-st', '--square-training',
        dest='square_training',
        action='store_true',
        help='Resize images to square shape instead of aspect ratio resizing \
              for single image training. For mosaic training, this resizes \
              single images to square shape first then puts them on a \
              square canvas.'
    )
    parser.add_argument(
        '--world-size',
        default=1,
        type=int,
        help='number of distributed processes'
    )
    parser.add_argument(
        '--dist-url',
        default='env://',
        type=str,
        help='url used to set up the distributed training'
    )
    # Wandb
    parser.add_argument(
        '-we', '--enable-wandb',
        dest="disable_wandb",
        action='store_false',
        help='use wandb'
    )
    parser.add_argument(
        '-wd', '--disable-wandb',
        dest="disable_wandb",
        action='store_true',
        help='do not use wandb'
    )
    parser.set_defaults(disable_wandb=False)

    parser.add_argument(
        '--sync-bn',
        dest='sync_bn',
        help='use sync batch norm',
        action='store_true'
    )
    parser.add_argument(
        '--amp',
        action='store_true',
        help='use automatic mixed precision'
    )
    parser.add_argument(
        '--patience',
        default=10,
        help='number of epochs to wait for when mAP does not increase to \
              trigger early stopping',
        type=int
    )
    parser.add_argument(
        '--optimizer',
        default="adam",
        choices=["adam", "sgd"],
        help='type of optimizer',
        type=str,
    )
    parser.add_argument(
        '--momentum',
        default=0.9,
        help='optimizer momentum',
        type=float,
    )    
    parser.add_argument(
        '--weight-decay',
        default=0,
        help='optimizer weight decay (L2 penalty)',
        type=float,
    )    
    
    parser.add_argument(
        '-ca', '--cosine-annealing',
        dest='cosine_annealing',
        action='store_true',
        help='use cosine annealing warm restarts'
    )

    parser.add_argument(
        '--seed',
        default=0,
        type=int,
        help='global seed for training'
    )
    parser.add_argument(
        '--project-dir',
        dest='project_dir',
        default=None,
        help='save resutls to custom dir instead of `outputs` directory, \
              --project-dir will be named if not already present',
        type=str
    )

    args = vars(parser.parse_args())
    return args
