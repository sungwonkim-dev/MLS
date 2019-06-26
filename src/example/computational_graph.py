import tensorflow as tf

l_node = tf.constant (3.0, tf.float32)
r_node = tf.constant (8.0)

root_node = tf.add(l_node, r_node)

session = tf.Session()

print(session.run(root_node))