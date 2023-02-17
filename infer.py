import tensorflow as tf
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import random

from helpers import *

def infer_one(model: tf.keras.Model, noisy_img: np.ndarray):
    input = np.expand_dims(noisy_img, axis=0)
    denoised_img = model.predict(input)
    denoised_img = (255.0*(denoised_img[0] - np.min(denoised_img[0]))/np.ptp(denoised_img[0])).astype(np.uint8)
    return denoised_img[0]

def infer(model: tf.keras.Model, noisy_imgs: np.ndarray):
    denoised_imgs = model.predict(noisy_imgs, batch_size=4)
    denoised_imgs = np.asarray([(255.0*(x - np.min(x))/np.ptp(x)).astype(np.uint8) for x in denoised_imgs])
    return denoised_imgs

def visualize(noisy_img: np.ndarray, denoised_img: np.ndarray, clean_img: np.ndarray = None, name: str = None):
    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(noisy_img)
    axs[0].set_title('Noisy')
    axs[1].imshow(denoised_img)
    axs[1].set_title('Denoised')
    if clean_img is not None:
        axs[2].imshow(clean_img)
        axs[2].set_title('Clean')
    # axs[2].imshow(denoised_img - noisy_img)
    # axs[2].set_title('Denoised - Noisy')
    if name != None:
        plt.savefig(f'./results/{name}.png')
    else:
        plt.show()

if __name__ == '__main__':
    # load model
    model_name = 'REDNet'
    model = tf.keras.models.load_model(f'./models/{model_name}.h5')

    # load data
    sidd_dir = './datasets/SIDD'
    renoir_dir = './datasets/RENOIR'
    nind_dir = './datasets/NIND'
    gt_paths, noisy_paths = get_img_paths(sidd_dir, renoir_dir, nind_dir)

    # load data
    choices = random.choices(range(len(gt_paths)), k=10)
    gt_imgs = load_images([gt_paths[i] for i in choices])
    noisy_imgs = load_images([noisy_paths[i] for i in choices])

    # infer
    denoised_imgs = infer(model, noisy_imgs)
    print(f'Paths: \n{[noisy_paths[i] for i in choices]}')
    # visualize
    for i in range(10):
        visualize(noisy_imgs[i], denoised_imgs[i], gt_imgs[i], f'{model_name}_{i}_plot')
        denoised_imgs[i] = cv.cvtColor(denoised_imgs[i], cv.COLOR_RGB2BGR) # openCV uses BGR, so we need to convert back
        print(cv.imwrite(f'./results/{model_name}_{i}.png', denoised_imgs[i]))