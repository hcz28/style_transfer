import tensorflow as tf
import numpy as np
import scipy.io
import pdb

# image normalization before processed by vgg
MEAN_PIXEL = np.array([123.68, 116.779, 103.939])

def net(vgg_path, input_image):
    """
    Calculate the output of the layers of VGG19 network.

    Args:
        vgg_path
        input_image

    Returns:
        A dictionary. Key is the name of VGG19 layers, value is the output of layers.
    """
    layers = (
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

            'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

            'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
            'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

            'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
            'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

            'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
            'relu5_3', 'conv5_4', 'relu5_4' 
            )
    vgg = scipy.io.loadmat(vgg_path)
    weights = vgg['layers'][0]

    net = {}
    current = input_image
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels = weights[i][0][0][2][0][0] # difference
            bias = weights[i][0][0][2][0][1]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = np.transpose(kernels, (1, 0, 2, 3))
            bias = bias.reshape(-1)
            current = _conv_layer(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current)
        elif kind == 'pool':
            current = _pool_layer(current)
        net[name] = current
    
    assert len(net) == len(layers)
    return net


def _conv_layer(x, kernels, bias):
    conv = tf.nn.conv2d(x, tf.constant(kernels), strides = (1, 1, 1, 1),
            padding = 'SAME')
    return tf.nn.bias_add(conv, bias)

def _pool_layer(x):
    return tf.nn.max_pool(x, ksize = (1, 2, 2, 1), strides = (1, 2, 2, 1),
            padding = 'SAME')

def preprocess(image):
    return image - MEAN_PIXEL

def unprocess(image):
    return image + MEAN_PIXEL
