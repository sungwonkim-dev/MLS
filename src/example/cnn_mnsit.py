import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../../res/data/mnist", one_hot=True)

image = mnist.train.images[0].reshape(28, 28)

print(image)

plt.imshow(image, cmap='gray')

session = tf.InteractiveSession()

# -1 -> 숫자 제한없음
image = image.reshape(-1, 28, 28, 1)
# 3x3 필터, 1개의 채널, 5개의 필터
weight = tf.Variable(tf.random_normal([3, 3, 1, 5], stddev=0.01))
# 2x2
conv2d = tf.nn.conv2d(image, weight, strides=[1, 2, 2, 1], padding='SAME')

print(conv2d)

session.run(tf.global_variables_initializer())
conv2d_image = conv2d.eval()
conv2d_image = np.swapaxes(conv2d_image, 0, 3)
for i, one_img in enumerate(conv2d_image):
    plt.subplot(1, 5, i + 1), plt.imshow(one_img.reshape(14, 14), cmap='gray')

pool = tf.nn.max_pool(conv2d, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
session.run(tf.global_variables_initializer())

pool_image = pool.eval()
pool_image = np.swapaxes(pool_image, 0, 3)

for i, one_img in enumerate(pool_image):
    plt.subplot(1, 5, i + 1), plt.imshow(one_img.reshape(7, 7), cmap='gray')
