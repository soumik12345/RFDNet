import tensorflow as tf
from .blocks import residual_feature_distillation_block


def RFDNet(features=64, filters=64, scale_factor=3):
    input_tensor = tf.keras.Input(shape=[None, None, 3])
    x_1 = tf.keras.layers.Conv2D(
        features, kernel_size=(3, 3), padding='same'
    )(input_tensor)
    b_1 = residual_feature_distillation_block(x_1)
    b_2 = residual_feature_distillation_block(b_1)
    b_3 = residual_feature_distillation_block(b_2)
    b_4 = residual_feature_distillation_block(b_3)
    concat = tf.keras.layers.Concatenate(axis=-1)([b_1, b_3, b_3, b_4])
    concat_1 = tf.keras.layers.Conv2D(
        features, kernel_size=(1, 1), activation='relu'
    )(concat)
    lr = tf.keras.layers.Conv2D(
        features, kernel_size=(3, 3), padding='same'
    )(concat_1) + x_1
    x_up = tf.keras.layers.Conv2D(
        filters * (scale_factor ** 2), 3, padding='same'
    )(lr)
    out = tf.nn.depth_to_space(x_up, scale_factor)
    output_tensor = tf.keras.layers.Conv2D(3, kernel_size=(1, 1))(out)
    return tf.keras.Model(input_tensor, output_tensor)
