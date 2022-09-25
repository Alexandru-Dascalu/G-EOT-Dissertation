import tensorflow as tf

def normalize(std_images, adv_images):
    std_images_minimums = tf.reduce_min(std_images, axis=[1, 2, 3], keepdims=True)
    adv_images_minimums = tf.reduce_min(adv_images, axis=[1, 2, 3], keepdims=True)

    std_images_maximums = tf.reduce_max(std_images, axis=[1, 2, 3], keepdims=True)
    adv_images_maximums = tf.reduce_max(adv_images, axis=[1, 2, 3], keepdims=True)

    minimum = tf.minimum(std_images_minimums, adv_images_minimums)
    maximum = tf.maximum(std_images_maximums, adv_images_maximums)

    minimum = tf.minimum(minimum, 0)
    maximum = tf.maximum(maximum, 1)

    return (std_images - minimum) / (maximum - minimum), (adv_images - minimum) / (maximum - minimum)