import tensorflow as tf
import model as mb
import generator
import config as c
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

import datetime
import argparse
import os
import pathlib
import sys

CHECKPOINT_PATH = os.path.join('.', 'evaluation', 'checkpoint_weights.hdf5')

def train_model(train_ds, test_ds, learning_rate=0.001):
    model = mb.build_model(learning_rate=learning_rate)
    model.summary()

    callbacks = [
        ModelCheckpoint(
            filepath=CHECKPOINT_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=True,
            verbose=1),
        EarlyStopping(
            monitor='val_accuracy',
            min_delta=1e-8,
            patience=10,
            restore_best_weights=True,
            verbose=1),
        ReduceLROnPlateau(
            monitor='val_accuracy',
            min_delta=1e-8,
            factor=0.2,
            patience=5,
            verbose=1)
    ]

    history = model.fit(train_ds,
           epochs=c.NUM_EPOCHS,
           steps_per_epoch=int(c.NUM_TRAIN_EXAMPLES/c.BATCH_SIZE),
           validation_data=test_ds, 
           validation_steps=int(c.NUM_TEST_EXAMPLES/c.BATCH_SIZE),
           callbacks=callbacks)

    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default=False)
    args = parser.parse_args()

    os.makedirs('evaluation', exist_ok=True)

    if args.train:
        train_ds, test_ds, ds_info = generator.build_dataset()

        start_time = datetime.datetime.now()

        model, h = train_model(train_ds, test_ds)

        total_time = datetime.datetime.now() - start_time

        loss = h.history['loss']
        val_loss = h.history['val_loss']

        min_val_loss = min(val_loss)
        min_val_loss_i = val_loss.index(min_val_loss)
        
        accuracy = h.history['accuracy']
        val_accuracy = h.history['val_accuracy']
        
        max_val_acc = max(val_accuracy)
        max_val_acc_i = val_accuracy.index(min_val_acc)

        time_epoch = (total_time / len(loss))

        t_corpus = "\n".join([
            "Batch:                   {}\n".format(c.BATCH_SIZE),
            "Time per epoch:          {}".format(time_epoch),
            "Total epochs:            {}".format(len(loss)),
            "Best epoch               {}\n".format(min_val_loss_i + 1),
            "Training loss:           {}".format(loss[min_val_loss_i]),
            "Validation loss:         {}".format(min_val_loss),
            "Training accuracy:           {}".format(accuracy[max_val_acc_i]),
            "Validation accuracy:         {}".format(max_val_acc),
        ])

        with open(os.path.join('.', 'evaluation', 'train.txt'), "w") as f:
            f.write(t_corpus)
            print(t_corpus)