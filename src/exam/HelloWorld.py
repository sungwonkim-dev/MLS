import tensorflow as tf

comment = 'Hello World!'
hello = tf.constant(comment)

session = tf.Session()

print(session.run(hello))