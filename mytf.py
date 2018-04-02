import numpy as np
import tensorflow as tf

# Model we want is A * x + B
# First we set up variable parameters A and B

A = tf.Variable([0.3], tf.float32)
B = tf.Variable([-0.3], tf.float32)

# Now we write the definitions for input and output
x = tf.placeholder(tf.float32)
linear_model = A * x + B

# y is used for loss calculation
y = tf.placeholder(tf.float32)

# Loss function
loss = tf.reduce_sum(tf.square(linear_model - y)) # Sum of squares

# Optimizer function
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(loss)

# Setting up the training data
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

# Training loop
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init) # reset values to incorrect defaults

# Now we train our model
for i in range(1000):
    sess.run(train, feed_dict={x:x_train, y:y_train})
    curr_x, curr_y, curr_loss = sess.run([A, B, loss], feed_dict={x:x_train, y:y_train})
    print("A: %s\nB: %s\nloss: %s \n" % (curr_x, curr_y, curr_loss))

#Evaluate training accuracy
final_A, final_B, final_loss = sess.run([A, B, loss], feed_dict={x:x_train, y:y_train})
print ("A: %s\nB: %s\nloss: %s \n" % (final_A, final_B, final_loss))
