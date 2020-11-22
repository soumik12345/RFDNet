import numpy as np
from PIL import Image
import tensorflow as tf
from .model import RFDNet


class Inferer:

    def __init__(self):
        self.model = None

    def load_model(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def _build_model(self, features, filters, scale_factor):
        self.model = RFDNet(
            features=features, filters=filters,
            scale_factor=scale_factor
        )

    def load_weights(self, features=64, filters=64, scale_factor=3, weights_path=None):
        self._build_model(features, filters, scale_factor)
        self.model.load_weights(weights_path)

    def infer(self, image_path):
        original_image = Image.open(image_path)
        image = tf.keras.preprocessing.image.img_to_array(original_image)
        image = image.astype('float32') / 255.0
        image = np.expand_dims(image, axis=0)
        output = self.model.predict(image)
        output_image = output[0] * 255.0
        output_image = output_image.clip(0, 255)
        output_image = output_image.reshape(
            (np.shape(output_image)[0], np.shape(output_image)[1], 3)
        )
        output_image = Image.fromarray(np.uint8(output_image))
        original_image = Image.fromarray(np.uint8(original_image))
        original_image_bilinear = original_image.resize(output_image.size, Image.BICUBIC)
        return output_image, original_image_bilinear
