import pathlib
import re
import numpy as np
import os
import shutil
import sklearn.model_selection as sk
# import cv2 as cv
import tensorflow as tf

def get_img_paths(datasets):
    gts = []
    noisy = []
    sidd_dir = './datasets_full/SIDD'
    renoir_dir = './datasets_full/RENOIR'
    nind_dir = './datasets_full/NIND'
    polyu_dir = './datasets_full/PolyU'

    # SIDD
    if 'SIDD' in datasets:
        sidd_dir = pathlib.Path(sidd_dir)
        sidd_img_paths = list(sidd_dir.rglob('*.PNG'))
        sidd_img_paths = [str(path) for path in sidd_img_paths]
        for img in sidd_img_paths:
            img_type = img.split(os.path.sep)[-1].split('_')[0]
            if img_type == 'GT':
                gts.append(img)
            elif img_type == 'NOISY':
                noisy.append(img)

    # Renoir
    if 'RENOIR' in datasets:
        renoir_dir = pathlib.Path(renoir_dir)
        for set in renoir_dir.iterdir():
            for batch in set.iterdir():
                if '.txt' in batch.name:
                    continue
                img_paths = list(batch.glob('*.bmp'))
                img_paths = [str(path) for path in img_paths]
                noisies = [x for x in img_paths if 'Noisy' in x.split(os.path.sep)[-1]]
                gts.append([x for x in img_paths if 'Reference' in x.split(os.path.sep)[-1]][0])
                noisy.append(noisies[np.argmax([int(re.search('(?<=IMG_)(.*)(?=N)', x)[0]) for x in noisies])]) # insert skull emoji here

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
        for img in polyu_dir.iterdir():
            if 'mean' in img.name: # someday I'll be living in a big old city and all you're ever gonna be is mean (with apologies to Taylor Swift)
                gts.append(str(img))
            elif 'Real' in img.name:
                noisy.append(str(img))
    
    gts_array = np.asarray(gts)
    noisy_array = np.asarray(noisy)
    return gts_array, noisy_array

def link_imgs(datasets):
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
    shutil.rmtree('./datasets/')

def generate_dataset(datasets, batch_size=1, augmentations=None):
    train_length, test_length =  link_imgs(datasets)
    train_dataset_gts = tf.keras.utils.image_dataset_from_directory('datasets/gts', label_mode=None, color_mode='rgb', batch_size=batch_size, image_size=(256, 256), shuffle=False, interpolation='bilinear', follow_links=True)
    train_dataset_noisy = tf.keras.utils.image_dataset_from_directory('datasets/noisy', label_mode=None, color_mode='rgb', batch_size=batch_size, image_size=(256, 256), shuffle=False, interpolation='bilinear', follow_links=True)
    train_dataset = tf.data.Dataset.zip((train_dataset_noisy, train_dataset_gts))
    test_dataset_gts = tf.keras.utils.image_dataset_from_directory('datasets/gts_test', label_mode=None, color_mode='rgb', batch_size=batch_size, image_size=(256, 256), shuffle=False, interpolation='bilinear', follow_links=True)
    test_dataset_noisy = tf.keras.utils.image_dataset_from_directory('datasets/noisy_test', label_mode=None, color_mode='rgb', batch_size=batch_size, image_size=(256, 256), shuffle=False, interpolation='bilinear', follow_links=True)
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
    datasets = ['RENOIR']
    train_dataset, test_dataset, train_length, test_length = generate_dataset(datasets)
    print(train_length)
    print(test_length)

    remove_links()