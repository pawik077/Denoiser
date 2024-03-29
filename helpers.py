import pathlib
import re
import numpy as np
import os
import shutil
import sklearn.model_selection as sk
# import cv2 as cv
import tensorflow as tf

def get_img_paths(datasets):
    '''Returns a list of paths to ground truth and noisy images from the specified datasets
    Args:
        datasets: list of strings, each string is the name of a dataset'''
    gts = []
    noisy = []
    sidd_dir = './datasets_full/SIDD'
    renoir_dir = './datasets_full/RENOIR'
    nind_dir = './datasets_full/NIND'
    polyu_dir = './datasets_full/PolyU'

    # SIDD
    if 'SIDD' in datasets:
        sidd_dir = pathlib.Path(sidd_dir)
        sidd_dir_data = sidd_dir / 'Data'
        pairs = [list(x.iterdir()) for x in sidd_dir_data.iterdir()]
        gts.extend([str(x) for pair in pairs for x in pair if 'GT' in x.name])
        noisy.extend([str(x) for pair in pairs for x in pair if 'NOISY' in x.name])

    # Renoir
    if 'RENOIR' in datasets:
        renoir_dir = pathlib.Path(renoir_dir)
        sources = list(renoir_dir.iterdir())
        batches = [batch for source in sources for batch in source.iterdir() if batch.is_dir()]
        gts.extend([str(x) for batch in batches for x in batch.iterdir() if 'Reference' in x.name])
        noisy.extend([str(n[np.argmax([int(re.search('(?<=IMG_)(.*)(?=N)', x.name)[0]) for x in n])]) for n in [list(batch.glob('*Noisy.bmp')) for batch in batches]]) # insert skull emoji here

    # #NIND
    if 'NIND' in datasets:
        nind_dir = pathlib.Path(nind_dir)
        for batch in nind_dir.iterdir():
            img_paths = list(batch.glob('*.png'))
            img_paths.extend(list(batch.glob('*.jpg')))
            img_paths = [str(path) for path in img_paths]
            isos = [re.search('(?<=ISO).*?(?=[\.-])', x)[0] for x in img_paths]
            ref = np.argmin([int(x) if x.isdigit() else 9999999 for x in isos])
            if 'H4' in isos:
                noisiest = isos.index('H4')
            elif 'H3' in isos:
                noisiest = isos.index('H3')
            elif 'H2' in isos:
                noisiest = isos.index('H2')
            elif 'H1' in isos:
                noisiest = isos.index('H1')
            else:
                noisiest = np.argmax([int(x) for x in isos])
            gts.append(img_paths[ref])
            noisy.append(img_paths[noisiest])

    # PolyU
    if 'PolyU' in datasets:
        polyu_dir = pathlib.Path(polyu_dir)
        groups = list(set([re.search('(.*)(?=_)', a.name).group(0) for a in list(polyu_dir.iterdir())]))
        pairs = [list(polyu_dir.glob(f'*{group}*')) for group in groups]
        gts.extend([str(x) for pair in pairs for x in pair if 'mean' in x.name])
        noisy.extend([str(x) for pair in pairs for x in pair if 'Real' in x.name])

    
    gts_array = np.asarray(gts)
    noisy_array = np.asarray(noisy)
    return gts_array, noisy_array

def link_imgs(datasets):
    '''Creates symlinks to the images in the specified datasets
    Args:
        datasets: list of strings, each string is the name of a dataset
    Returns:
        int, number of images in the training set
        int, number of images in the test set'''
    gt_paths, noisy_paths = get_img_paths(datasets)
    gt_paths_train, gt_paths_test, noisy_paths_train, noisy_paths_test = sk.train_test_split(gt_paths, noisy_paths, test_size=0.2)
    os.makedirs('./datasets/gts', exist_ok=True)
    os.makedirs('./datasets/noisy', exist_ok=True)
    os.makedirs('./datasets/gts_test', exist_ok=True)
    os.makedirs('./datasets/noisy_test', exist_ok=True)
    for i, (gt, n) in enumerate(zip(gt_paths_train, noisy_paths_train)):
        os.symlink(os.path.abspath(gt), os.path.join('./datasets/gts', f'{i}.{gt.split(".")[-1]}'))
        os.symlink(os.path.abspath(n), os.path.join('./datasets/noisy', f'{i}.{n.split(".")[-1]}'))
    for i, (gt, n) in enumerate(zip(gt_paths_test, noisy_paths_test)):
        os.symlink(os.path.abspath(gt), os.path.join('./datasets/gts_test', f'{i}.{gt.split(".")[-1]}'))
        os.symlink(os.path.abspath(n), os.path.join('./datasets/noisy_test', f'{i}.{n.split(".")[-1]}'))
    return len(gt_paths_train), len(gt_paths_test)

def remove_links():
    '''Removes the symlinks created by link_imgs'''
    shutil.rmtree('./datasets/')

def generate_dataset(datasets, batch_size=1, augmentations=None):
    '''Generates a tf.data.Dataset from the specified datasets
    Args:
        datasets: list of strings, each string is the name of a dataset
        batch_size: int, batch size
        augmentations: list of functions, each function is an augmentation to be applied to the dataset
    Returns:
        train_dataset: tf.data.Dataset, training dataset
        test_dataset: tf.data.Dataset, testing dataset
        train_length: int, number of training images
        test_length: int, number of testing images'''
    train_length, test_length =  link_imgs(datasets)
    train_dataset_gts = tf.keras.utils.image_dataset_from_directory('datasets/gts', label_mode=None, color_mode='rgb', batch_size=batch_size, image_size=(512, 512), shuffle=False, interpolation='bilinear', follow_links=True)
    train_dataset_noisy = tf.keras.utils.image_dataset_from_directory('datasets/noisy', label_mode=None, color_mode='rgb', batch_size=batch_size, image_size=(512, 512), shuffle=False, interpolation='bilinear', follow_links=True)
    train_dataset = tf.data.Dataset.zip((train_dataset_noisy, train_dataset_gts))
    test_dataset_gts = tf.keras.utils.image_dataset_from_directory('datasets/gts_test', label_mode=None, color_mode='rgb', batch_size=batch_size, image_size=(512, 512), shuffle=False, interpolation='bilinear', follow_links=True)
    test_dataset_noisy = tf.keras.utils.image_dataset_from_directory('datasets/noisy_test', label_mode=None, color_mode='rgb', batch_size=batch_size, image_size=(512, 512), shuffle=False, interpolation='bilinear', follow_links=True)
    test_dataset = tf.data.Dataset.zip((test_dataset_noisy, test_dataset_gts))
    if augmentations:
        for f in augmentations:
            if np.random.uniform() < 0.5:
                train_dataset = train_dataset.map(f, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    train_dataset = train_dataset.repeat()
    test_dataset = test_dataset.repeat()
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return train_dataset, test_dataset, train_length, test_length

if __name__ == '__main__':
    datasets = ['SIDD', 'RENOIR', 'NIND', 'PolyU']
    train_dataset, test_dataset, train_length, test_length = generate_dataset(datasets)
    print(train_length)
    print(test_length)

    remove_links()
