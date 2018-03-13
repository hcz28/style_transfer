import pdb, transform 
import tensorflow as tf
import numpy as np
from argparse import ArgumentParser 
from utils import exists, list_files, get_img, save_img 

OUTPUT_PATH = 'results'

def build_parser():
    parser = ArgumentParser()

    parser.add_argument('--style', type=str, dest='style',
            help='style model path', metavar='STYLE', required=True)

    parser.add_argument('--content', type=str, dest='content',
            help='content image path', metavar='CONTENT', required=True)

    parser.add_argument('--output-path', type=str, dest='output_path',
            help='output path', metavar='OUTPUT_PATH', default=OUTPUT_PATH)
    return parser

def check_opts(opts):
    exists(opts.style, "style model not found!")
    exists(opts.content, "content image not found!")

def main():
    parser = build_parser()
    options = parser.parse_args()
    check_opts(options)
    content_img = get_img(options.content, (256, 256, 3)).astype(np.float32)
    content_img = np.reshape(content_img, (1,) + content_img.shape)
    prediction = ffwd(content_img, options.style)
    save_img(options.output_path, prediction)
    print('Image saved to {}'.format(options.output_path))

def ffwd(content, network_path):
    with tf.Session() as sess:
        content_placeholder = tf.placeholder(tf.float32, 
                shape=content.shape, name='content_placeholder')
        network = transform.net(content_placeholder/255.0)

        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(network_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
        prediction = sess.run(network, feed_dict={content_placeholder : content})
    return prediction[0]

if __name__ == '__main__':
    main()

    


