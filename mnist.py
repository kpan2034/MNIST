import tensorflow as tf
import numpy as np

# Get data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Create the model

## Tensor for array of pixels
x = tf.placeholder(tf.float32, [None, 784])

## W - weight, b - bias
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

## model
y = tf.matmul(x, W) + b
y_ = tf.placeholder(tf.float32, [None, 10])

## Loss function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

## Optimizer function
learning_rate = 0.5
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_step = optimizer.minimize(cross_entropy)

# Start session
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# Train

for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs , y_: batch_ys})
    curr_x, curr_y, curr_loss = sess.run([x, y, cross_entropy], feed_dict={x:batch_xs, y_:batch_ys}) 
    print ("W: %s\nb: %s\nL: %s" % (curr_x, curr_y, curr_loss))

# Test trained model

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_:mnist.test.labels}))

# Save trained model
model_path = "/home/kpan/tf/"
saver = tf.train.Saver()
save_path = saver.save(sess, "my_model")


