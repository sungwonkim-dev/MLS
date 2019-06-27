import tensorflow as tf
import matplotlib.pyplot as plt
import random
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../../res/data/mnist", one_hot=True)

number_classes = 10
width = 28
height = 28
learning_rate = 0.01

X = tf.compat.v1.placeholder(tf.float32, [None, width * height])
Y = tf.compat.v1.placeholder(tf.float32, [None, number_classes])

W = tf.compat.v1.Variable(tf.random.normal([width * height, number_classes]))
b = tf.compat.v1.Variable(tf.random.normal([number_classes]))

hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

limit_epochs = 10
batch_size = 100

with tf.compat.v1.Session() as session:
    session.run(tf.global_variables_initializer())

    for epoch in range(limit_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)
        print("total batch :", total_batch)
        for index in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            cost_val, _ = session.run([cost, optimizer], feed_dict={X: batch_x, Y: batch_y})

            avg_cost = (avg_cost + cost_val) / total_batch

            print("epoch : ", epoch, "avg cost : ", avg_cost)
    print("test acc : ", session.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))

    random_index = random.randint(0, mnist.test.num_examples - 1)
    print(session.run(tf.argmax(mnist.test.labels[random_index:random_index + 1], 1)))
    print(session.run(tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[random_index:random_index + 1]}))
    plt.imshow(mnist.test.images[random_index:random_index + 1].reshape(width, height))
    plt.show()
