import tensorflow as tf
import numpy as np

# methods used for data augmentation

def up_down_flip(image, label):
	'''Flips the image and label vertically.'''
	image = tf.image.flip_up_down(image)
	label = tf.image.flip_up_down(label)
	return image, label

def left_right_flip(image, label):
	'''Flips the image and label horizontally.'''
	image = tf.image.flip_left_right(image)
	label = tf.image.flip_left_right(label)
	return image, label

def rotate(image, label):
	'''Rotates the image and label by a random angle, 0, 90, 180, or 270 degrees.'''
	angle = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
	image = tf.image.rot90(image, angle)
	label = tf.image.rot90(label, angle)
	return image, label

def adjust_hue(image, label):
	'''Adjusts the hue of the image and label by a random amount.'''
	image, label = tf.image.random_hue([image, label], 0.5)
	return image, label

def adjust_brightness(image, label):
	'''Adjusts the brightness of the image and label by a random amount.'''
	image = tf.image.random_brightness([image, label], 0.3)
	return image, label

def adjust_contrast(image, label):
	'''Adjusts the contrast of the image and label by a random amount.'''
	image = tf.image.random_contrast([image, label], 1, 3)
	return image, label

def adjust_saturation(image, label):
	'''Adjusts the saturation of the image and label by a random amount.'''
	image = tf.image.random_saturation([image, label], 1, 5)
	return image, label