import sys, os, pdb, time 
import numpy as np
import scipy.misc
from optimize import optimize
from argparse import ArgumentParser
from utils import save_img, get_img, exists, list_files

CONTENT_WEIGHT = 7.5e0
STYLE_WEIGHT = 1e2
TV_WEIGHT = 2e2

LEARNING_RATE = 1e-3
NUM_EPOCHS = 2
CHECKPOINT_DIR = 'checkpoints'
CHECKPOINT_ITERATIONS = 2000
VGG_PATH = 'data/imagenet-vgg-verydeep-19.mat'
TRAIN_PATH = 'data/train2014'
BATCH_SIZE = 4
DEVICE = '/gpu:0'
FRAC_GPU = 1 #?

STYLE_PATH = 'data/style_targets/horses.jpg'

def build_parser():
    parser = ArgumentParser()
    
    parser.add_argument('--style', type=str, dest='style', 
            help='style images path', metavar='STYLE', required = True)

    parser.add_argument('--train-path', type=str, dest='train_path',
            help='path to training images folder', metavar='TRAIN_PATH', 
            default=TRAIN_PATH)

    parser.add_argument('--checkpoint-dir', type=str, dest='checkpoint_dir',
            help='dir to save checkpoint in', metavar='CHECKPOINT_DIR',
            default=CHECKPOINT_DIR)

    parser.add_argument('--test', type=str, dest='test', 
            help='test image path', metavar='TEST', default=False)

    parser.add_argument('--test-dir', type=str, dest='test_dir', 
            help='test image save dir', metavar='TEST_DIR', default=False)
    
    parser.add_argument('--epochs', type=int, dest='epochs', 
            help='num epochs', metavar='EPOCHS', default=NUM_EPOCHS)

    parser.add_argument('--batch-size', type=int, dest='batch_size',
            help='batch_size', metavar='BATCH_SIZE', default=BATCH_SIZE)

    parser.add_argument('--checkpoint-iterations', type=int, 
            dest='checkpoint_iterations', help='checkpoint frequency',
            metavar='CHECKPOINT_ITERATIONS', default=CHECKPOINT_ITERATIONS) 

    parser.add_argument('--vgg-path', type=str, dest='vgg_path',
            help='path to VGG19 network (default %(default)s)', 
            metavar='VGG_PATH', default=VGG_PATH)

    parser.add_argument('--content-weight', type=float, dest='content_weight',
            help='content weight (default %(default)s)',
            metavar='CONTENT_WEIGHT', default=CONTENT_WEIGHT)

    parser.add_argument('--style-weight', type=float, dest='style_weight',
            help='style weight (default %(default)s)',
            metavar='STYLE_WEIGHT', default=STYLE_WEIGHT)

    parser.add_argument('--tv-weight', type=float, dest='tv_weight',
            help='total variation regularization weight (default %(default)s)',
            metavar='TV_WEIGHT', default=TV_WEIGHT)
    
    parser.add_argument('--learning-rate', type=float, dest='learning_rate',
            help='learning rate (default %(default)s)',
            metavar='LEARNING_RATE', default=LEARNING_RATE)
    return parser 

def check_opts(opts):
    exists(opts.checkpoint_dir, "checkpoint dir not found!")
    exists(opts.style, "style path not found!")
    exists(opts.train_path, "train path not found!")
    if opts.test or opts.test_dir:
        exists(opts.test, "test img not found!")
        exists(opts.test_dir, "test directory not found!")
    exists(opts.vgg_path, "vgg network not found!")
    assert opts.epochs > 0
    assert opts.batch_size > 0
    assert opts.checkpoint_iterations > 0
    assert opts.content_weight >= 0
    assert opts.style_weight >= 0
    assert opts.tv_weight >= 0
    assert opts.learning_rate >= 0
"""
def _get_files(img_dir):
    files = list_files(img_dir)
    return [os.path.join(img_dir, x) for x in files]
"""
def main():
    parser = build_parser()
    options = parser.parse_args()
    check_opts(options)

    style_target = get_img(options.style)
    content_targets = list_files(options.train_path)
    kwargs = {
            "epochs":options.epochs,
            "print_iterations":options.checkpoint_iterations,
            "batch_size":options.batch_size,
            "save_path":os.path.join(options.checkpoint_dir,'fns.ckpt'),
            "learning_rate":options.learning_rate
            }
    args = [
            content_targets,
            style_target,
            options.content_weight,
            options.style_weight,
            options.tv_weight,
            options.vgg_path
            ]
    start_time = time.time()
    for preds, losses, i, epoch in optimize(*args, **kwargs):
        style_loss, content_loss, tv_loss, loss = losses
        print('{0} ---------- Epoch: {1}, Iteration: {2}----------'.format(time.ctime(), epoch, i))
        print('Total loss: {0}, Style loss: {1}, Content loss: {2}, TV loss: {3}'
                .format(loss, style_loss, content_loss, tv_loss))
    print("Training complete! Total training time is {0} s".format(time.time() - start_time))

if __name__ == '__main__':
    main()
