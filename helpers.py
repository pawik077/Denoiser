import pathlib
import re
import numpy as np
import os
import cv2 as cv

def get_img_paths(sidd_dir, renoir_dir, nind_dir):
    gts = []
    noisy = []
    
    # SIDD
    sidd_dir = pathlib.Path(sidd_dir)
    sidd_img_paths = list(sidd_dir.rglob('*.png'))
    sidd_img_paths = [str(path) for path in sidd_img_paths]
    for img in sidd_img_paths:
        img_type = img.split(os.path.sep)[-1].split('_')[0]
        if img_type == 'GT':
            gts.append(img)
        elif img_type == 'NOISY':
            noisy.append(img)

    # Renoir
    renoir_dir = pathlib.Path(renoir_dir)
    for set in renoir_dir.iterdir():
        for batch in set.iterdir():
            if '.txt' in batch.name:
                continue
            img_paths = list(batch.glob('*.bmp'))
            img_paths = [str(path) for path in img_paths]
            ref = [x for x in img_paths if 'Reference' in x.split(os.path.sep)[-1].split('.')[0]][0]
            noisies = 0
            for img in img_paths:
                if 'Noisy' in img.split(os.path.sep)[-1].split('.')[0]:
                    noisy.append(img)
                    noisies += 1
            for _ in range(noisies):
                gts.append(ref)

    # #NIND
    # nind_dir = pathlib.Path(nind_dir)
    # nind_img_paths = list(nind_dir.rglob('*.png'))
    # nind_img_paths.extend(list(nind_dir.rglob('*.jpg')))
    # nind_img_paths = [str(path) for path in nind_img_paths]
    # for img in nind_img_paths:
    #     img_name = img.split(os.path.sep)[-1]
    #     iso = re.search('(?<=ISO).*?(?=[\.-])', img_name)[0]
    #     if 'H' in iso:
    #         noisy.append(img)
    #     elif int(iso) > 4000:
    #         noisy.append(img) 
    #     else:
    #         gts.append(img)

    # #NIND
    # nind_dir = pathlib.Path(nind_dir)
    # for batch in nind_dir.iterdir():
    #     img_paths = list(batch.glob('*.png'))
    #     img_paths.extend(list(batch.glob('*.jpg')))
    #     img_paths = [str(path) for path in img_paths]
    # to be continued
        
    gts_array = np.asarray(gts)
    noisy_array = np.asarray(noisy)
    return gts_array, noisy_array

def load_images(img_paths):
    images = []
    for path in img_paths:
        img = cv.imread(path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = cv.resize(img, (256, 256))
        images.append(img)
    return np.asarray(images)




if __name__ == '__main__':
    sidd_dir = './datasets/SIDD'
    renoir_dir = './datasets/RENOIR'
    nind_dir = './datasets/NIND'
    # get_img_paths(sidd_dir, renoir_dir, nind_dir)