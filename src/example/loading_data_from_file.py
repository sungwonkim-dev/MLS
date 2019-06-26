import tensorflow as tf
import numpy as np

df = np.loadtxt('../../res/data/example/data-01-test-score.csv', delimiter=',', dtype=np.float32)

print(df)

x = df[:, 0: -1]
y = df[:, [-1]]

X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(X, W) + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

session = tf.Session()
session.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, hy_val, _ = session.run(
        [cost, hypothesis, train], feed_dict={X: x, Y: y})
    if step % 10 == 0:
        print(step, cost_val, hy_val)

print("test set : 100 70 101")
print(session.run(hypothesis, feed_dict={X: [[100, 70, 101]]}))
