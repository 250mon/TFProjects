import keras.callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import os
from utils import Config
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    config = Config()
    data_dir = config.options["data_dir"]

    train_dir = os.path.join(data_dir, "train")
    valid_dir = os.path.join(data_dir, "valid")
    # fractured_dir = os.path.join(train_dir, "FRACTURED")
    # unfractured_dir = os.path.join(train_dir, "UNFRACTURED")
    # fractured_img = os.listdir(fractured_dir)
    # print(fractured_img[:5])

    batch_size = 54
    target_size = (100, 100)

    train_datagen = ImageDataGenerator(rescale=1/255)
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        classes=["FRACTURED", "UNFRACTURED"],
        class_mode='binary'
    )

    valid_datagen = ImageDataGenerator(rescale=1/255)
    valid_generator = valid_datagen.flow_from_directory(
        valid_dir,
        target_size=target_size,
        batch_size=batch_size,
        classes=["FRACTURED", "UNFRACTURED"],
        class_mode='binary'
    )

    model_path = "muras_frac_detect_model.keras"
    try:
        model = keras.models.load_model(model_path)
    except:
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(16, (3, 3), activation="tanh", input_shape=(100, 100, 3)),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Conv2D(32, (3, 3), activation="tanh"),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation="tanh"),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="tanh"),
            tf.keras.layers.Dense(1, activation="sigmoid")
        ])
        model.compile(
            loss="binary_crossentropy",
            optimizer="adam",
            metrics=["accuracy",
                     tf.keras.metrics.Precision(name='precision'),
                     tf.keras.metrics.Recall(name='recall'),
                     ]
        )

    model.summary()
    total_sample = train_generator.n
    num_epoch = 100
    checkpoint_cb = keras.callbacks.ModelCheckpoint(model_path, save_best_only=True)
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    history = model.fit(
        train_generator,
        steps_per_epoch=int(total_sample/batch_size),
        epochs=num_epoch,
        validation_data=valid_generator,
        validation_steps=200,
        callbacks=[checkpoint_cb, early_stopping_cb]
    )

    return history


if __name__ == "__main__":
    history = main()

    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.show()

