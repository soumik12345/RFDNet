import tensorflow as tf


def horizontal_flips(lr, hr):
    condition = tf.cast(
        tf.random.uniform(
            [], maxval=2, dtype=tf.int32
        ), tf.bool
    )
    lr, hr = tf.cond(
        condition, lambda: tf.image.flip_left_right(lr),
        lambda: tf.image.flip_left_right(hr)
    )
    return lr, hr


def rotate_90(lr, hr):
    condition = tf.cast(
        tf.random.uniform(
            [], maxval=2, dtype=tf.int32
        ), tf.bool
    )
    lr, hr = tf.cond(
        condition, lambda: tf.image.rot90(lr, k=1),
        lambda: tf.image.rot90(hr, k=1)
    )
    return lr, hr
