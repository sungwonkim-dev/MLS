import tensorflow as tf

x = [[1, 2, 1, 1],
     [2, 1, 3, 2],
     [3, 1, 3, 4],
     [4, 1, 5, 5],
     [1, 7, 5, 5],
     [1, 2, 5, 6],
     [1, 6, 6, 6],
     [1, 7, 7, 7]]
y = [[0, 0, 1],
     [0, 0, 1],
     [0, 0, 1],
     [0, 1, 0],
     [0, 1, 0],
     [0, 1, 0],
     [1, 0, 0],
     [1, 0, 0]]

X = tf.placeholder("float", [None, 4])
Y = tf.placeholder("float", [None, 3])
nb_classes = 3

W = tf.Variable(tf.random_normal([4, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    for step in range(2001):
        _, cost_val = session.run([optimizer, cost], feed_dict={X: x, Y: y})

        if step % 200 == 0:
            print(step, cost_val)

    result = session.run(hypothesis, feed_dict={X: [[1, 11, 7, 9]]})
    print(result, session.run(tf.arg_max(result, 1)))
