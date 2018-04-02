import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt  
import matplotlib.animation as animation

num_house = 160
np.random.seed(42)
house_size = np.random.randint(low=1000, high=3500, size=num_house)

np.random.seed(42)
house_price = house_size * 100.0 + np.random.randint(low=20000, high=70000, size=num_house)

plt.plot(house_size, house_price, "bx")
plt.ylabel("Price")
plt.xlabel("Size")
plt.show()

# we normalize the data
def normalize(array):
    return (array - array.mean()) / array.std()

# create 70-30 divide
num_train_samples = math.floor(num_house * 0.7)

# training data as follows:
# This is an array.
# We then proceed to normalize it.

train_house_size = np.asarray(house_size[:num_train_samples])
train_price = np.asarray(house_size[:num_train_samples])

train_house_size_norm = normalize(train_house_size)
train_price_norm = normalize(train_price)

# test data as follows:
# Same thing as before - it is an array and we normalize it
test_house_size = np.asarray(house_size[num_train_samples:])
test_price = np.asarray(house_price[num_train_samples:])

test_house_norm = normalize(test_house_size)
test_price_norm = normalize(test_price)


# Note that we normalized AFTER the split

# setting up tensors
# They will be provided values from the data so they are merely placeholders.

tf_house_size = tf.placeholder("float", name="house_size")
tf_price = tf.placeholder("float", name="house_price")

# Define size factor and offset as variables
tf_size_factor = tf.Variable(np.random.randn(), name="size_factor")
tf_price_offset = tf.Variable(np.random.randn(), name="price_offset")

# inference function - the stuff we want
# be sure to use tf and not np functions for add() and multiply()
tf_price_pred = tf.add(tf.multiply(tf_size_factor, tf_house_size), tf_price_offset) # apparently we could've used '+' and '*' instead

#loss function
tf_cost = tf.reduce_sum(tf.pow(tf_price_pred - tf_price, 2))/(2*num_train_samples)

# learning rate i.e. gradient step size
learning_rate = 0.1

# Gradient Descent optimizer to minimize loss in 'cost' operation
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_cost)



# initialze stuff so they exist
init = tf.global_variables_initializer()

# Launch the graph in session
with tf.Session() as sess:
    sess.run(init)

    # how often we want to display training progress and number of iterations

    display_every = 2
    num_training_iter = 50

    # keep iterating the training data

    for iteration in range(num_training_iter):

        # fit all the training data
        for (x, y) in zip(train_house_size_norm, train_price_norm):
            sess.run(optimizer, feed_dict={tf_house_size: x, tf_price: y})

        # display current status
        if ((iteration + 1) % display_every == 0):
            c = sess.run(tf_cost, feed_dict={tf_house_size: train_house_size_norm, tf_price: train_price_norm})
            print("iteration #:", '%04d' % (iteration + 1), "cost=", "{:.9f}".format(c), "size_factor=",sess.run(tf_size_factor), "price_offset=", sess.run(tf_price_offset))


    print("Optimization finished:")
    training_cost = sess.run(tf_cost, feed_dict={tf_house_size: train_house_size_norm, tf_price: train_price_norm})
    print("Trained Cost=", training_cost, "size_factor=", sess.run(tf_size_factor), "price_offset=", sess.run(tf_price_offset), '\n')


    #Plot of training and test data, and learned regression:

    #Get values used to normalize data so we can denormalize it:
    train_house_size_mean = train_house_size.mean()
    train_house_size_std = train_house_size.std()

    train_price_mean = train_price.mean()
    train_price_std = train_price.std()


    #Plot the graph:
    plt.rcParams["figure.figsize"] = (10, 8)
    plt.figure()
    plt.ylabel("Price")
    plt.xlabel("Size sq. ft")
    plt.plot(train_house_size, train_price, 'go', label='trainig data')
    plt.plot(test_house_size, test_price, 'mo', label='testing data')
    plt.plot(train_house_size_norm * train_house_size_std + train_house_size_mean,\
    (sess.run(tf_size_factor) * train_house_size_norm + sess.run(tf_price_offset)) \
    *train_price_std + train_price_mean, label='Learned Regression')

    plt.legend('upper left')
    plt.show()

