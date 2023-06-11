import tensorflow as tf
import sklearn.model_selection as sk

from helpers import *
import augmentation
from REDNet import REDNet_model

def train(model: tf.keras.Model, datasets, augmentations, filename: str, batch_size: int = 1, epochs: int = 200):
    '''Trains the model on the given datasets and saves the model to the given filename
    Args:
        model: the model to train
        datasets: a list of datasets to train on
        augmentations: a list of augmentations to apply to the training data
        filename: the filename to save the model to
        batch_size: the batch size to use for training
        epochs: the number of epochs to train for'''
    train_dataset, test_dataset, train_length, test_length = generate_dataset(datasets, batch_size=batch_size, augmentations=augmentations)

    #train model
    model_path = f'./models/{filename}'
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, min_delta=0.0001, patience=10),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.8, patience=3, verbose=1, min_lr=1e-6, min_delta=0.0001),
        tf.keras.callbacks.ModelCheckpoint(filepath=model_path, monitor='val_loss', mode='min', save_best_only=True, save_weights_only=False, frequency='epoch'),      
    ]
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss=tf.keras.losses.MeanSquaredError(), metrics=[tf.keras.metrics.MeanSquaredError()])
    model.fit(train_dataset, validation_data=test_dataset, epochs=epochs, steps_per_epoch=np.ceil(train_length/batch_size), validation_steps=np.ceil(test_length/batch_size), verbose=1, callbacks=callbacks)
    #save model
    model.save(model_path)
    remove_links()

if __name__ == '__main__':
    # for testing purposes
    train(REDNet_model(), ['SIDD', 'RENOIR', 'NIND'], [augmentation.up_down_flip, augmentation.left_right_flip, augmentation.rotate], 'REDNet.h5')
