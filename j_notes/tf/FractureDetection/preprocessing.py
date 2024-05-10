import tensorflow as tf
import os
import pathlib
from utils import Config


def get_ds(type:str = 'train'):
    config = Config()
    data_dir = config.options["data_dir"]
    if type == 'valid':
        img_paths = pathlib.Path(data_dir, 'MURA-v1.1/valid_image_paths.csv')
    else:
        img_paths = pathlib.Path(data_dir, 'MURA-v1.1/train_image_paths.csv')


    path_types = [tf.string]
    img_ds = tf.data.experimental.CsvDataset(str(img_paths), path_types)
    dir_label_types = [tf.string, tf.int32]

    def process_path(img_path):
        label = tf.strings.split(img_path, os.sep)[-2]
        label = tf.strings.split(label, '_')[-1]
        if label == b'positive':
            label = 1
        else:
            label = 0
        
        image = tf.io.read_file(data_dir+'/'+img_path)
        image = tf.io.decode_png(image)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [128, 128])
        return image, label


    labeled_ds = img_ds.map(process_path)
    return labeled_ds

    # for image_raw, label_text in labeled_ds.take(1):
    #     print(repr(image_raw.numpy()[:100]))
    #     print()
    #     print(label_text.numpy())


if __name__ == "__main__":
    get_ds()
