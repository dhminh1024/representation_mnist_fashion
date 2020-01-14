import tensorflow as tf
import generator


callbacks = [
    ModelCheckpoint(
            filepath='checkpoint_weights.hdf5',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            verbose=1),
        EarlyStopping(
            monitor='val_loss',
            min_delta=1e-8,
            patience=10,
            restore_best_weights=True,
            verbose=1),
        ReduceLROnPlateau(
            monitor='val_loss',
            min_delta=1e-8,
            factor=0.2,
            patience=5,
            verbose=1)
]

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
          input_shape=(IMG_SIZE, IMG_SIZE, IMG_CHANNELS)),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(rate=0.3),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(rate=0.3),
        tf.keras.layers.Dense(generator.NUM_CLASSES, activation='softmax')
    ])

    return model