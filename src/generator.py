# coding: utf-8

import tensorflow as tf
import tensorflow_datasets as tfds
import config as c

tfds.disable_progress_bar()
dataset_name = 'fashion_mnist'
labels_text = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def preprocess(ds):
    img = tf.image.resize_with_pad(ds['image'], c.IMG_H, c.IMG_W)
    img = tf.cast(img, tf.float32)
    img = (img/127.5) -1
    return img, ds['label']

def augmentation(image, label):
    image = tf.image.random_brightness(image, .1)
    image = tf.image.random_contrast(image, lower=0.0, upper=1.0)
    image = tf.image.random_flip_left_right(image)
    return image, label

def build_dataset(dataset_name=dataset_name):
    fashion_mnist = tfds.image.FashionMNIST()
    fashion_mnist.download_and_prepare()
    ds_info = fashion_mnist.info
    ds = fashion_mnist.as_dataset()
    train_ds, test_ds = ds['train'], ds['test']
    train_ds = train_ds.map(preprocess).cache().repeat().shuffle(c.SHUFFLE_BUFFER_SIZE).batch(c.BATCH_SIZE)
    # train_ds = train_ds.map(augmentation)
    train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.map(preprocess).cache().repeat().batch(c.BATCH_SIZE)
    test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)

    return train_ds, test_ds, ds_info

if __name__ == "__main__":
    train_ds, test_ds = build_dataset()
    for image_batch, label_batch in train_ds:
        print("Image batch shape: ", image_batch.shape)
        print("Label batch shape: ", label_batch.shape)
        break