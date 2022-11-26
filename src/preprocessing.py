import numpy as np
import tensorflow as tf


def preprocess(img):
    preprocessed_img = tf.Variable(tf.image.convert_image_dtype(img, tf.float32))

    # TODO: do other stuff here if necessary

    return preprocessed_img