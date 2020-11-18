import tensorflow as tf
from rfdnet import RFDNet


def test_model_output_shape():
    model = RFDNet()
    for upscale_factor in range(1, 5):
        x = tf.ones((1, 100, 100, upscale_factor))
        y = model(x)
        assert y.shape[1] == x.shape[1] * upscale_factor
        assert y.shape[2] == x.shape[2] * upscale_factor
