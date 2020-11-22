import os
import tensorflow as tf
from .model import RFDNet
from .dataloader import SRDataLoader
from wandb.keras import WandbCallback


class Trainer:

    def __init__(self):
        self.model = None
        self.train_dataset = None
        self.loss_function = None
        self.optimizer = None

    def build_dataset(
            self, dataset_url=None, crop_size=300, image_limiter=800,
            downsample_factor=3, batch_size=8, buffer_size=1024):
        self.train_dataset = SRDataLoader(
            dataset_url=dataset_url, crop_size=crop_size,
            downsample_factor=downsample_factor, image_limiter=image_limiter,
            batch_size=batch_size, buffer_size=buffer_size
        ).make_dataset()

    def build_model(self, features=64, filters=64, scale_factor=3):
        self.model = RFDNet(
            features=features, filters=filters,
            scale_factor=scale_factor
        )

    def compile(self, learning_rate=1e-3):
        self.build_model()
        self.loss_function = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate, epsilon=1e-8)
        self.model.compile(optimizer=self.optimizer, loss=self.loss_function)

    def train(self, epochs=100, steps_per_epoch=20, checkpoint_path='./checkpoints'):
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='loss', patience=10
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(
                    checkpoint_path, 'rfdnet_best.h5'
                ), monitor='loss', mode='min', period=1,
                save_best_only=True
            ), WandbCallback()
        ]
        self.model.fit(
            self.train_dataset, epochs=epochs,
            callbacks=callbacks, steps_per_epoch=steps_per_epoch
        )
