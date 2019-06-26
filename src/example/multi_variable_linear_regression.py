import tensorflow as tf

'''
math = [73., 88., 12., 32., 76.]
eng = [45., 88., 100., 32., 76.]
sci = [92., 43., 54., 12., 77.]
grade = [152., 178., 123., 88., 149]

m = tf.placeholder(tf.float32)
e = tf.placeholder(tf.float32)
s = tf.placeholder(tf.float32)

Y = tf.placeholder(tf.float32)

w_m = tf.Variable(tf.random_normal([1]), name='weight_math')
w_e = tf.Variable(tf.random_normal([1]), name='weight_english')
w_s = tf.Variable(tf.random_normal([1]), name='science')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = w_m * m + w_e * e + w_s * s

cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

session = tf.Session()
session.run(tf.global_variables_initializer())

for step in range(1001):
    cost_val, hypothesis_val, _ = session.run([cost, hypothesis, train],
                                              feed_dict={m: math, e: eng, s: sci, Y: grade})
    if step % 10 == 0:
        print(step, cost_val, hypothesis_val)

'''

x = [[73., 80., 75.],
     [93., 88., 93.],
     [89., 91., 90.],
     [96., 98., 100.],
     [73., 66., 70.]]
y = [[152.],
     [185.],
     [180.],
     [196.],
     [142.]]


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
    cost_val, hypothesis_val, _ = session.run(
        [cost, hypothesis, train], feed_dict={X: x, Y: y})
    if step % 10 == 0:
        print(step, cost_val, hypothesis_val)
