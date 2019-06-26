import tensorflow as tf
import numpy as np

df = np.loadtxt('../../res/data/example/data-03-diabetes.csv', delimiter=',', dtype=np.float32)

x = df[:, 0:-1]
y = df[:, [-1]]

X = tf.placeholder(tf.float32, shape=[None, x.shape[1]])
Y = tf.placeholder(tf.float32, shape=[None, y.shape[1]])

W = tf.Variable(tf.random_normal([x.shape[1], y.shape[1]]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(X, W) + b)
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis)
                                     + (1 - Y) * tf.log(1 - hypothesis)))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
acc = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    for step in range(10001):
        cost_val, W_val, b_val, _ = session.run([cost, W, b, train], feed_dict={X: x, Y: y})

        if step % 200 == 0:
            print(step, cost_val, W_val, b_val)

    h, c, a = session.run([hypothesis, predicted, acc], feed_dict={X: x, Y: y})
    print(h, c, a)
