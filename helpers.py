import pathlib
import re
import numpy as np
import os
import cv2 as cv

def get_img_paths(datasets):
    gts = []
    noisy = []
    sidd_dir = './datasets/SIDD'
    renoir_dir = './datasets/RENOIR'
    nind_dir = './datasets/NIND'

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
                noisy.append(noisies[np.argmax([int(re.search(r"(?<=IMG_)(.*)(?=N)", x)[0]) for x in noisies])]) # insert skull emoji here

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
    
    gts_array = np.asarray(gts)
    noisy_array = np.asarray(noisy)
    return gts_array, noisy_array

def load_images(img_paths):
    images = []
    for path in img_paths:
        img = cv.imread(path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB) # openCV reads images in BGR format, convert to RGB for nn and plotting
        img = cv.resize(img, (256, 256))
        images.append(img)
    return np.asarray(images)
