import tensorflow as tf


def shallow_residual_block(input_tensor, filters):
    x = tf.keras.layers.Conv2D(
        filters, kernel_size=(3, 3), padding='same'
    )(input_tensor)
    x = tf.nn.relu(x)
    return input_tensor + x


def residual_feature_distillation_block(input_tensor):
    filter_left = int(list(input_tensor.shape)[-1] / 2)
    filter_right = int(list(input_tensor.shape)[-1])
    left_1 = tf.keras.layers.Conv2D(filter_left, kernel_size=(1, 1))(input_tensor)
    right_1 = shallow_residual_block(input_tensor, filter_right)
    left_2 = tf.keras.layers.Conv2D(filter_left, kernel_size=(1, 1))(right_1)
    right_2 = shallow_residual_block(right_1, filter_right)
    left_3 = tf.keras.layers.Conv2D(filter_left, kernel_size=(1, 1))(right_2)
    right_3 = shallow_residual_block(right_2, filter_right)
    right_final = tf.keras.layers.Conv2D(filter_left, kernel_size=(3, 3), padding='same')(right_3)
    concat = tf.keras.layers.Concatenate(axis=-1)([left_1, left_2, left_3, right_final])
    concat_1 = tf.keras.layers.Conv2D(filter_right, kernel_size=(1, 1))(concat)
    return concat_1 + input_tensor
