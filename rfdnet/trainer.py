import os
import tensorflow as tf
from .model import RFDNet
from .utils import lr_scheduler
from .dataloader import SRDataLoader
from wandb.keras import WandbCallback


class Trainer:

    def __init__(self):
        self.model = None
        self.train_dataset = None
        self.dataset_length = None
        self.loss_function = None
        self.optimizer = None
        self.batch_size = None
        self.strategy = tf.distribute.OneDeviceStrategy("GPU:0")
        if len(tf.config.list_physical_devices('GPU')) > 1:
            self.strategy = tf.distribute.MirroredStrategy()

    def build_dataset(
            self, dataset_url=None, crop_size=64, image_limiter=None,
            downsample_factor=2, batch_size=16, buffer_size=1024):
        with self.strategy.scope():
            train_dataloader = SRDataLoader(
                dataset_url=dataset_url, crop_size=crop_size,
                downsample_factor=downsample_factor, image_limiter=image_limiter,
                batch_size=batch_size, buffer_size=buffer_size
            )
            self.train_dataset = train_dataloader.make_dataset()
            self.dataset_length = len(train_dataloader)
            self.batch_size = batch_size
            print('Number of Images:', self.dataset_length)

    def build_model(self, features=64, filters=64, scale_factor=2):
        self.model = RFDNet(
            features=features, filters=filters,
            scale_factor=scale_factor
        )

    def compile(self, features=64, filters=64, scale_factor=2, learning_rate=1e-3):
        with self.strategy.scope():
            self.build_model(features, filters, scale_factor)
            self.loss_function = tf.keras.losses.MeanSquaredError()
            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=learning_rate, epsilon=1e-8)
            self.model.compile(optimizer=self.optimizer, loss=self.loss_function)

    def train(
            self, epochs=100, steps_per_epoch=1e5,
            checkpoint_path='./checkpoints', checkpoint_name='rfdnet_best.h5'):
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='loss', patience=10
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(
                    checkpoint_path, checkpoint_name
                ), monitor='loss', mode='min', save_freq=1,
                save_best_only=True, save_weights_only=True
            ), WandbCallback(),
            tf.keras.callbacks.LearningRateScheduler(lr_scheduler),
            # tf.keras.callbacks.ReduceLROnPlateau(
            #     monitor='loss', factor=0.5, patience=5
            # )
        ]
        self.model.fit(
            self.train_dataset, epochs=epochs,
            callbacks=callbacks, steps_per_epoch=steps_per_epoch
        )
