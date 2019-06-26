import tensorflow as tf

x = [1, 2, 3]
y = [1, 2, 3]

W = tf.Variable(tf.random_normal([1]), name='weight')
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = X * W

cost = tf.reduce_mean(tf.square(hypothesis - Y))

'''
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.005)
train = optimizer.minimize(cost)
'''
learning_rate = 0.01
gradient = tf.reduce_mean((W * X - Y) * X)
descent = W - learning_rate * gradient
update = W.assign(descent)

session = tf.Session()
session.run(tf.global_variables_initializer())

for step in range(100):
    session.run(update, feed_dict={X: x, Y : y})
    print(step, session.run(cost, feed_dict={X: x, Y : y}), session.run(W))
