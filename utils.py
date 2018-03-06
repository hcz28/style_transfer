import scipy.misc
import numpy as np
import os, sys

def save_img(save_path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    scipy.misc.imsave(save_path, img)

def scale_img(path, scale):
    scale = float(scale)
    o0, o1, o2 = scipy.misc.imread(path, mode='RGB').shape
    new_shape = (int(o0 * scale), int(o1 * scale), o2)
    style_target = get_img(path, img_size=new_shape)
    return style_target

def get_img(path, img_size=False):
    img = scipy.misc.imread(path, mode='RGB')
    if not (len(img.shape) == 3 and img.shape[2] == 3):
        img = np.dstack((img, img, img))
    if img_size != False:
        img = scipy.misc.imresize(img, img_size)
    return img

def exists(path, msg):
    assert os.path.exists(path), msg

def list_files(path):
    files = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        files.extend(filenames)
        break
    return files
