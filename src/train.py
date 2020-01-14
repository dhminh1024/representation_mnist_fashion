import tensorflow as tf
import model as mb
import generator
import config as c
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

def train_model(train_ds, test_ds, learning_rate=0.001):
    model = mb.build_model(learning_rate=learning_rate)

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

    model.fit(train_ds,
           epochs=c.NUM_EPOCHS,
           steps_per_epoch=int(c.NUM_TRAIN_EXAMPLES/c.BATCH_SIZE),
           validation_data=test_ds, 
           validation_steps=int(c.NUM_TEST_EXAMPLES/c.BATCH_SIZE),
           callbacks=callbacks)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default=False)
    args = parser.parse_args()

    if args.train:
        train_ds, test_ds, ds_info = generator.build_dataset()

        model = train_model(train_ds, test_ds)