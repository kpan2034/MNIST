import tensorflow as tf
import numpy as np
import 'mnist.py'
# Get data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
sess = tf.Session()
loader = tf.train.Saver()
loader.restore(sess, model_path)
print ("Loading Session...")
with