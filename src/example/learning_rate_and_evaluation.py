import tensorflow as tf

# Normalized inputs
# from sklearn.preprocessing import MinMaxScaler
# x = MinMaxScaler().fit_transform(x)

x = [[1, 2, 1], [1, 3, 2], [1, 3, 4], [1, 5, 5], [1, 7, 5], [1, 2, 5], [1, 6, 6], [1, 7, 7]]
y = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]]

x_t = [[2, 1, 1], [3, 1, 2], [3, 3, 4]]
y_t = [[0, 0, 1], [0, 0, 1], [0, 0, 1]]

X = tf.compat.v1.placeholder("float", [None, 3])
Y = tf.compat.v1.placeholder("float", [None, 3])

W = tf.Variable(tf.random.normal([3, 3]))
b = tf.Variable(tf.random.normal([3]))

hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.math.log(hypothesis), axis=1))
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

prediction = tf.argmax(hypothesis, 1)
is_correct = tf.equal(prediction, tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

with tf.compat.v1.Session() as session:
    session.run(tf.compat.v1.global_variables_initializer())

    for step in range(2001):
        cost_val, W_val, _ = session.run([cost, W, optimizer], feed_dict={X: x, Y: y})

        if step % 100 == 0:
            print(step, cost_val, W_val)

    print("prediction : ", session.run(prediction, feed_dict={X: x_t, Y: y_t}))
    print("acc : ", session.run(accuracy, feed_dict={X: x_t, Y: y_t}))
