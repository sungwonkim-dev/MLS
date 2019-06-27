import tensorflow as tf
import numpy as np

df = np.loadtxt('../../res/data/example/data-04-zoo.csv', delimiter=',', dtype=np.float32)

x = df[:, 0:-1]
y = df[:, [-1]]

number_classes = 7

X = tf.placeholder(tf.float32, [None, x.shape[1]])
Y = tf.placeholder(tf.int32, [None, 1])
Y_one_hot = tf.one_hot(Y, number_classes)
Y_one_hot = tf.reshape(Y_one_hot, [-1, number_classes])

W = tf.Variable(tf.random_normal([x.shape[1], number_classes]), name="weight")
b = tf.Variable(tf.random_normal([number_classes]), name="bias")

logits = tf.matmul(X, W) + b

hypothesis = tf.nn.softmax(logits)
cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot)

cost = tf.reduce_mean(cost_i)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    # loss = cost
    for step in range(2001):
        _, loss, acc = session.run([optimizer, cost, accuracy], feed_dict={X: x, Y: y})

        if step % 100 == 0:
            print("Step: {:5}\tCost: {:.3f}\tAcc: {:.2%}".format(step, loss, acc))

    pred = session.run(prediction, feed_dict={X: x})
    for p, y in zip(pred, y.flatten()):
        print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))