import tensorflow as tf
import numpy as np

# methods used for data augmentation

def up_down_flip(image, label):
	image = tf.image.flip_up_down(image)
	label = tf.image.flip_up_down(label)
	return image, label

def left_right_flip(image, label):
	image = tf.image.flip_left_right(image)
	label = tf.image.flip_left_right(label)
	return image, label

def rotate(image, label):
	angle = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
	image = tf.image.rot90(image, angle)
	label = tf.image.rot90(label, angle)
	return image, label

def adjust_hue(image, label):
	image, label = tf.image.random_hue([image, label], 0.5)
	return image, label

def adjust_brightness(image, label):
	image = tf.image.random_brightness([image, label], 0.3)
	return image, label

def adjust_contrast(image, label):
	image = tf.image.random_contrast([image, label], 1, 3)
	return image, label

def adjust_saturation(image, label):
	image = tf.image.random_saturation([image, label], 1, 5)
	return image, label

def dataset_generator(x, y, batch_size=32, augmentations=None):
	# x: noisy images
	# y: clean images
	# augmentations: list of augmentation functions
	# returns: dataset with augmentations
	dataset = tf.data.Dataset.from_tensor_slices((x, y))
	if augmentations:
		for f in augmentations:
			if np.random.uniform() < 0.5:
				dataset = dataset.map(f, num_parallel_calls=tf.data.experimental.AUTOTUNE)

	dataset = dataset.repeat()
	dataset = dataset.batch(batch_size, drop_remainder=True)
	dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
	return dataset