import tensorflow as tf

x = [1, 4, 9]
y = [3, 9, 19]

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = x * W + b

cost = tf.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.005)
train = optimizer.minimize(cost)

session = tf.Session()
session.run(tf.global_variables_initializer())

for step in range(1001):
    session.run(train)
    if step % 10 == 0:
        print(step, session.run(cost), session.run(W), session.run(b))

x = tf.placeholder(tf.float32, shape=[None])
y = tf.placeholder(tf.float32, shape=[None])

for step in range(1001):
    cost_val, W_val, b_val, train_val = session.run([cost, W, b, train], feed_dict={x: [1, 4, 9], y: [3, 9, 19]})
    if step % 10 == 0:
        print(step, cost_val, W_val, b_val)

print(session.run(hypothesis, feed_dict={x: [1, 4, 9]}))
