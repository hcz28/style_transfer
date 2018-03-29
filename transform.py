import tensorflow as tf
import pdb # python debugger

def net(image):
    """
    Compute the transformed image.

    Args:
        image: The imput 4-D(batch, width, height, channels) image batch. Note that the input image should be normalized outside this
        function. 

    Returns:
        The transformed image, which has the same size with input image.
    """
    conv1 = _conv2d(image, 32, 9, 1)
    conv2 = _conv2d(conv1, 64, 3, 2)
    conv3 = _conv2d(conv2, 128, 3, 2)
    res1 = _residual_block(conv3, 3)
    res2 = _residual_block(res1, 3)
    res3 = _residual_block(res2, 3)
    res4 = _residual_block(res3, 3)
    res5 = _residual_block(res4, 3)
    conv_t1 = _conv2d_transpose(res5, 64, 3, 2)
    conv_t2 = _conv2d_transpose(conv_t1, 32, 3, 2)
    conv_t3 = _conv2d(conv_t2, 3, 9, 1, relu = False)
    preds = tf.nn.tanh(conv_t3) * 127.5 + 255./2
    return preds


def _conv2d(x, out_channels, filter_size, stride_size, relu = True):
    """
    Define the convolutional layer.

    Args:
        x: The input 4-D batch.
        out_channels: The number of filters/out put channels of this layer.
        filter_size: The filter size of this layer.
        stride_size: The stride size.
        relu: Whether use relu as activation function.

    Returns:
        A 4-D Tensor.
    """
    weights_init = _weights(x, out_channels, filter_size)
    stride_shape = [1, stride_size, stride_size, 1]
    x = tf.nn.conv2d(x, weights_init, stride_shape, padding = 'SAME')
    x = _instance_norm(x)
    if relu:
        x = tf.nn.relu(x)
    return x

def _residual_block(x, filter_size = 3):
    """
    Define the residual block.

    Args:
        x: The input 4-D batch.
        filter_size: Filter size of the residual block.

    Returns:
        A 4-D Tensor.
    """
    tmp = _conv2d(x, 128, filter_size, 1)
    tmp = _conv2d(tmp, 128, filter_size, 1, relu = False) # there is no relu
    return x + tmp

def _conv2d_transpose(x, out_channels, filter_size, stride_size):
    """
    Define the transpose convolutional layer.

    Args:
        x: The input 4-D batch.
        out_channels: The number of filters/output channels.
        filter_size: The size of the filter.
        stride_size: The size of the stride.

    Returns:
        A 4-D Tensor.
    """
    weights_init = _weights(x, out_channels, filter_size, transpose = True)
    batch_size, rows, cols, in_channels = [i.value for i in x.shape]
    new_rows, new_cols = int(rows * stride_size), int(cols * stride_size)
    new_shape = [batch_size, new_rows, new_cols, out_channels]
    output_shape = tf.stack(new_shape)
    stride_shape = [1, stride_size, stride_size, 1]
    x = tf.nn.conv2d_transpose(x, weights_init, output_shape, stride_shape, padding = 'SAME')
    x = _instance_norm(x)
    return tf.nn.relu(x)


def _instance_norm(x):
    """
    Instance normalization.

    Args:
        x: The input 4-D Tensor.

    Returns:
        A 4-D Tensor.
    """
    in_channels = x.shape[3].value 
    mu, var = tf.nn.moments(x, [1, 2], keep_dims = True)
    scale = tf.Variable(tf.ones([in_channels]))
    shift = tf.Variable(tf.zeros([in_channels]))
    epsilon = 1e-3
    norm = (x - mu) / (var + epsilon)**(.5)
    return scale * norm + shift # broadcast

def _weights(x, out_channels, filter_size, transpose = False):
    """
    Weights initialization for the conv2d layer and conv2d transpose layer.
    
    Args:
        x: The input 4-D Tensor.
        out_channels: The number of filters/output channels.
        filter_size: The size of filter.
        transpose: Whether the weights are initialized for a transpose layer or not.

    Returns:
        The initialization weights.
    """
    _, row, col, in_channels = [i.value for i in x.shape]
    if not transpose:
        weights_shape = [filter_size, filter_size, in_channels, out_channels]
    else:
        weights_shape = [filter_size, filter_size, out_channels, in_channels]
    weights_init = tf.Variable(tf.truncated_normal(weights_shape, stddev = .1,
        seed = 1), dtype = tf.float32)
    return weights_init

