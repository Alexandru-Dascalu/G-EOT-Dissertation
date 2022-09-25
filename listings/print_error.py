import tensorflow as tf
import config

def apply_print_error(std_textures, adv_textures):
    multiplier = tf.random.uniform(
        [config.batch_size, 1, 1, 3],
        config.channel_mult_min,
        config.channel_mult_max
    )
    addend = tf.random.uniform(
        [config.batch_size, 1, 1, 3],
        config.channel_add_min,
        config.channel_add_max
    )
    std_textures = std_textures * multiplier + addend
    adv_textures = adv_textures * multiplier + addend

    return std_textures, adv_textures