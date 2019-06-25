import tensorflow as tf
import matplotlib.pyplot as plt

x = [1, 4, 9]
y = [3, 12, 27]
W = tf.placeholder(tf.float32)

hypothesis = x * W
cost = tf.reduce_mean(tf.square(hypothesis - y))

session = tf.Session()
session.run(tf.global_variables_initializer())

cost_arr = []
W_arr = []

# -3 ~ 5 0.1 씩
for i in range(-30, 50):
    feed_W = i * 0.1
    curr_cost, curr_W = session.run([cost, W], feed_dict={W: feed_W})
    cost_arr.append(curr_cost)
    W_arr.append(curr_W)

plt.plot(W_arr, cost_arr)
plt.show()
