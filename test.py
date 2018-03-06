from transform import net
import tensorflow as tf 

image = tf.Variable(tf.truncated_normal([4,256,256,3], stddev = .1, seed = 1))
preds = net(image)
