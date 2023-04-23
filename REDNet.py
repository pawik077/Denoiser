import tensorflow as tf
import sklearn.model_selection as sk

import augmentation
from helpers import *

def REDNet_model():
    input = tf.keras.layers.Input(shape=(None, None, 3), name='input')
    conv1 = tf.keras.layers.Conv2D(filters=256, kernel_size=2, padding='same', name='conv1')(input)
    conv2 = tf.keras.layers.Conv2D(filters=256, kernel_size=2, padding='same', name='conv2')(conv1)
    conv3 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', name='conv3')(conv2)
    conv4 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', name='conv4')(conv3)
    conv5 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', name='conv5')(conv4)

    deconv5 = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=2, padding='same', name='deconv5')(conv5)
    deconv5 = tf.keras.layers.Add(name='add1')([conv4, deconv5])
    deconv4 = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=2, padding='same', name='deconv4')(deconv5)
    deconv3 = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=3, padding='same', name='deconv3')(deconv4)
    deconv3 = tf.keras.layers.Add(name='add2')([conv2, deconv3])
    deconv2 = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=3, padding='same', name='deconv2')(deconv3)
    deconv1 = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=3, padding='same', name='deconv1')(deconv2)
    output = tf.keras.layers.Add(name='add3')([input, deconv1])

    model = tf.keras.Model(inputs=input, outputs=output)
    return model

if __name__ == '__main__':
    model = REDNet_model()
    model.summary()
