import tensorflow as tf
import config

def add_background(std_images, adv_images, uv_maps):
    # compute a mask with True values for each pixel which represents the object, and False for background pixels.
    mask = tf.reduce_all(tf.not_equal(uv_maps, 0.0), axis=3, keepdims=True)
    # generate random background colour for each image in batch
    color = tf.random.uniform([config.batch_size, 1, 1, 3], config.background_min, config.background_max)

    new_std_images = set_background(std_images, mask, color)
    new_adv_images = set_background(adv_images, mask, color)
    return std_images, adv_images

def set_background(images, mask, colours):
    """
    images: A 4-D tensor with shape [batch_size, height, size, 3].
    mask: boolean mask with shape [batch_size, height, width, 1]
    colours: tensor with shape [batch_size, 1, 1, 3].
    """
    # repeat mask for each colour channel
    mask = tf.tile(mask, [1, 1, 1, 3])
    inverse_mask = tf.logical_not(mask)

    background = tf.cast(inverse_mask, tf.float32) * colours
    object_cutout = tf.cast(mask, tf.float32) * image
    return object_cutout + background