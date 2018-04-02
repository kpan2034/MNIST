import tensorflow as tf

sess = tf.Session()

#Print a string...you know which

hello = tf.constant("Hello, world!")
print(sess.run(hello))