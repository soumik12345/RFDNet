from glob import glob
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE


class SRDataLoader:

    def __init__(
            self, dataset_url=None, image_limiter=100,
            crop_size=300, downsample_factor=3, batch_size=8, buffer_size=1024):
        self.image_files = self.download_dataset(dataset_url, image_limiter)
        self.crop_size = crop_size
        self.downsample_factor = downsample_factor
        self.batch_size = batch_size
        self.buffer_size = buffer_size

    def __len__(self):
        return len(self.image_files)

    @staticmethod
    def download_dataset(dataset_url, image_limiter):
        file_name = dataset_url.split('/')[-1]
        dataset_path = tf.keras.utils.get_file(
            file_name, dataset_url, extract=True
        )
        image_files = glob(
            '/'.join([i for i in dataset_path.split('/')[:-1]]) + '/{}/*'.format(
                file_name.split('.')[0]
            )
        )
        image_files = image_files[:image_limiter] if image_limiter is not None else image_files
        return image_files

    @staticmethod
    def read_image(image_file):
        image = tf.io.read_file(image_file)
        image = tf.image.decode_image(image, channels=3)
        image.set_shape([None, None, 3])
        image = tf.cast(image, dtype=tf.float32)
        return image

    def map_function(self, image_file):
        image = self.read_image(image_file)
        hr = tf.image.random_crop(
            image, [self.crop_size, self.crop_size, 3]
        )
        lr = tf.image.resize(
            hr, [
                self.crop_size // self.downsample_factor,
                self.crop_size // self.downsample_factor
            ], 'bicubic'
        )
        hr = hr / 255.0
        lr = lr / 255.0
        return lr, hr

    def make_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices(self.image_files)
        dataset = dataset.shuffle(self.buffer_size)
        dataset = dataset.map(
            self.map_function,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        dataset = dataset.repeat()
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset
