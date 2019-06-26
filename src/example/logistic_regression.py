import tensorflow as tf

x = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y = [[0], [0], [0], [1], [1], [1]]

X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])

# W = tf.Variable(tf.random_normal([a, b]), name='weight')
# a = input 개수,  b = output 개수
W = tf.Variable(tf.random_normal([2, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(X, W) + b)
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis)
                                     + (1 - Y) * tf.log(1 - hypothesis)))

# cost가 최소가 되도록 minimize 함수 설정
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# 정확도 측정
# 0.5보다 크면  predicted = 1 아니면 0
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
acc = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    for step in range(10001):
        cost_val, W_val, b_val, _ = session.run([cost, W, b, train], feed_dict={X: x, Y: y})

        if step % 200 == 0:
            print(step, cost_val, W_val, b_val)

    print('finished train')
    h, c, a = session.run([hypothesis, predicted, acc], feed_dict={X: x, Y: y})
    print(h, c, a)
