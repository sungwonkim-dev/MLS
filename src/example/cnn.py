# image : (num,height,width,channel)
# filter (height, width, channel, filter num)
# stride (ㅁ,mH, mW,ㅁ)

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

session = tf.InteractiveSession()
image = np.array([[[[1], [2], [3]],
                   [[4], [5], [6]],
                   [[7], [8], [9]]]], dtype=np.float32)

weight = tf.constant([[[[1.]], [[1.]]],
                      [[[1.]], [[1.]]]])

#pool = tf.nn.max_pool(image, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')

print(image.shape)
print(weight.shape)

# padding = "SAME" -> 원본과 같은 크기로 만들어짐
conv2d = tf.nn.conv2d(image, weight, strides=[1, 1, 1, 1], padding='VALID')
conv2d_image = conv2d.eval()

for i, one_img in enumerate(conv2d_image):
    print(one_img.reshape(2, 2))
    plt.subplot(1, 2, i + 1), plt.imshow(one_img.reshape(2, 2), cmap='gray')
