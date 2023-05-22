import tensorflow as tf
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import random
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
import sys
import getopt

from helpers import *
from MWCNN import DWT_downsampling, IWT_upsampling, Conv_block
from PRIDNet import Convolution_block, Channel_attention, Avg_pool_Unet_Upsample_msfe, Multi_scale_feature_extraction, Kernel_selecting_module

def load_images(paths):
    imgs = []
    for path in paths:
        img = cv.imread(path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = cv.resize(img, (256, 256))
        imgs.append(img)
    return np.asarray(imgs, dtype=np.uint8)

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
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'm:rs:n:f:', ['model=', 'random', 'seed=', 'number=', 'file='])
    except getopt.GetoptError as err:
        print(err)
        exit(2)
    
    model_name = None
    rand = False
    seed = None
    number = None
    file = None

    for o, a in opts:
        if o in ('-m', '--model'):
            model_name = a
        elif o in ('-r', '--random'):
            rand = True
        elif o in ('-s', '--seed'):
            seed = int(a)
        elif o in ('-n', '--number'):
            number = int(a)
        elif o in ('-f', '--file'):
            file = a

    if model_name == None:
        raise ValueError('Model name not specified')
    if rand == False and file == None:
        raise ValueError('No input specified')
    if rand == True and file != None:
        raise ValueError('Cannot specify both random and file')
    if rand == False and seed != None:
        raise ValueError('Cannot specify seed without random')
    if rand == False and number != None:
        raise ValueError('Cannot specify number without random')
    if rand == True and number == None:
        raise ValueError('Number not specified')
    
    # load model
    if 'REDNet' in model_name:
        model = tf.keras.models.load_model(f'./models/{model_name}.h5')
    elif 'MWCNN' in model_name:
        model = tf.keras.models.load_model(f'./models/{model_name}.h5', custom_objects={'DWT_downsampling': DWT_downsampling, 'IWT_upsampling': IWT_upsampling, 'Conv_block': Conv_block})
    elif 'PRIDNet' in model_name:
        model = tf.keras.models.load_model(f'./models/{model_name}.h5', custom_objects={'Convolution_block': Convolution_block, 'Channel_attention': Channel_attention, 'Avg_pool_Unet_Upsample_msfe': Avg_pool_Unet_Upsample_msfe, 'Multi_scale_feature_extraction': Multi_scale_feature_extraction, 'Kernel_selecting_module': Kernel_selecting_module})
    elif model_name == 'nlm':
        model = None
    else:
        raise ValueError('Model not recognized')
    
    if rand == True:
        gt_paths, noisy_paths = get_img_paths(['SIDD', 'RENOIR', 'NIND'])
        choices = random.choices(range(len(gt_paths)), k=number)
        gt_images = load_images([gt_paths[i] for i in choices])
        noisy_images = load_images([noisy_paths[i] for i in choices])
    else:
        number = 1
        noisy_images = load_images([file])
    
    if model_name != 'nlm':
        denoised_images = infer(model, noisy_images)
    else:
        denoised_images = np.asarray([cv.fastNlMeansDenoisingColored(noisy_images[i], None, 10, 10, 7, 21) for i in range(len(noisy_images))])

    if rand:
        psnr_gt_mean = np.mean([psnr(gt_images[i], noisy_images[i]) for i in range(len(gt_images))])
        psnr_denoised_mean = np.mean([psnr(gt_images[i], denoised_images[i]) for i in range(len(gt_images))])
        ssim_gt_mean = np.mean([ssim(gt_images[i], noisy_images[i], channel_axis=-1, data_range=noisy_images[i].max() - noisy_images[i].min()) for i in range(len(gt_images))])
        ssim_denoised_mean = np.mean([ssim(gt_images[i], denoised_images[i], channel_axis=-1, data_range=denoised_images[i].max() - denoised_images[i].min()) for i in range(len(gt_images))])
        mse_gt_mean = np.mean([mse(gt_images[i], noisy_images[i]) for i in range(len(gt_images))])
        mse_denoised_mean = np.mean([mse(gt_images[i], denoised_images[i]) for i in range(len(gt_images))])
        print(f'PSNR (GT): {psnr_gt_mean}')
        print(f'PSNR (Denoised): {psnr_denoised_mean}')
        print(f'SSIM (GT): {ssim_gt_mean}')
        print(f'SSIM (Denoised): {ssim_denoised_mean}')
    # else:
    #     psnr_denoised_mean = psnr(gt_images[0], denoised_images[0])
    #     ssim_denoised_mean = ssim(gt_images[0], denoised_images[0], channel_axis=-1, data_range=denoised_images[0].max() - denoised_images[0].min())
    #     print(f'PSNR (Denoised): {psnr_denoised_mean}')
    #     print(f'SSIM (Denoised): {ssim_denoised_mean}')
    for i in range(number):
        visualize(noisy_images[i], denoised_images[i], gt_images[i], f'{model_name}_{i}_plot')
        denoised_images[i] = cv.cvtColor(denoised_images[i], cv.COLOR_RGB2BGR)
        is_saved = cv.imwrite(f'./results/{model_name}_{i}.png', denoised_images[i])
        if not is_saved:
            print(f'Failed to save {model_name}_{i}.png')
