import tensorflow as tf
import config

def repeat(x):
    """
    Args: x: A 3-D tensor with shape [height, width, 3]
    Returns: A 4-D tensor with shape [batch_size, height, size, 3]
    """
    return tf.tile(tf.expand_dims(x, axis=0), [config.batch_size, 1, 1, 1])