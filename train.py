import tensorflow as tf
import sklearn.model_selection as sk

from helpers import *
import augmentation
from REDNet import REDNet_model

def train(model: tf.keras.Model, datasets, augmentations,  filename: str, epochs: int = 200):
    # load data
    gt_paths, noisy_paths = get_img_paths(datasets)

    #split data
    gt_paths_train, gt_paths_test, noisy_paths_train, noisy_paths_test = sk.train_test_split(gt_paths, noisy_paths, test_size=0.2)
    
    #load data
    gt_train_imgs = load_images(gt_paths_train)
    noisy_train_imgs = load_images(noisy_paths_train)
    gt_test_imgs = load_images(gt_paths_test)
    noisy_test_imgs = load_images(noisy_paths_test)

    #augment data
    train_set = augmentation.dataset_generator(x=noisy_train_imgs, y=gt_train_imgs, batch_size=4, augmentations=augmentations)
    test_set = augmentation.dataset_generator(x=noisy_test_imgs, y=gt_test_imgs, batch_size=4, augmentations=None)

    #train model
    model_path = f'./models/{filename}.h5'
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, min_delta=0.0001, patience=10),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.8, patience=3, verbose=1, min_lr=1e-6, min_delta=0.0001),
        tf.keras.callbacks.ModelCheckpoint(filepath=model_path, monitor='val_loss', mode='min', save_best_only=True, save_weights_only=False, frequency='epoch'),      
    ]
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss=tf.keras.losses.MeanSquaredError(), metrics=[tf.keras.metrics.MeanSquaredError()])
    model.fit(train_set, validation_data=test_set, epochs=epochs, steps_per_epoch=len(noisy_train_imgs), validation_steps=len(noisy_test_imgs), verbose=1, callbacks=callbacks)

    #save model
    model.save(model_path)

if __name__ == '__main__':
    # for testing purposes
    train(REDNet_model(), ['SIDD', 'RENOIR', 'NIND'], [augmentation.up_down_flip, augmentation.left_right_flip, augmentation.rotate], 'REDNet')
