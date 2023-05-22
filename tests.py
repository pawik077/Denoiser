import tensorflow as tf
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse

import sys
import getopt

from helpers import *
from infer import load_images, infer
from MWCNN import DWT_downsampling, IWT_upsampling, Conv_block
from PRIDNet import Convolution_block, Channel_attention, Avg_pool_Unet_Upsample_msfe, Multi_scale_feature_extraction, Kernel_selecting_module

def full_test():
    rednet_model = tf.keras.models.load_model('./models/REDNet.h5')
    mwcnn_model = tf.keras.models.load_model(f'./models/MWCNN.h5', custom_objects={'DWT_downsampling': DWT_downsampling, 'IWT_upsampling': IWT_upsampling, 'Conv_block': Conv_block})
    pridnet_model = tf.keras.models.load_model(f'./models/PRIDNet.h5', custom_objects={'Convolution_block': Convolution_block, 'Channel_attention': Channel_attention, 'Avg_pool_Unet_Upsample_msfe': Avg_pool_Unet_Upsample_msfe, 'Multi_scale_feature_extraction': Multi_scale_feature_extraction, 'Kernel_selecting_module': Kernel_selecting_module})

    gt_paths, noisy_paths = get_img_paths(['PolyU'])
    gt_imgs = load_images(gt_paths)
    noisy_imgs = load_images(noisy_paths)
    
    denoised_imgs_rednet = infer(rednet_model, noisy_imgs)
    denoised_imgs_mwcnn = infer(mwcnn_model, noisy_imgs)
    denoised_imgs_pridnet = infer(pridnet_model, noisy_imgs)
    denoised_imgs_nlm = np.asarray([cv.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21) for img in noisy_imgs])

    psnr_gt_mean = np.mean([psnr(gt_imgs[i], noisy_imgs[i]) for i in range(len(gt_imgs))])
    psnr_rednet_mean = np.mean([psnr(gt_imgs[i], denoised_imgs_rednet[i]) for i in range(len(gt_imgs))])
    psnr_mwcnn_mean = np.mean([psnr(gt_imgs[i], denoised_imgs_mwcnn[i]) for i in range(len(gt_imgs))])
    psnr_pridnet_mean = np.mean([psnr(gt_imgs[i], denoised_imgs_pridnet[i]) for i in range(len(gt_imgs))])
    psnr_nlm_mean = np.mean([psnr(gt_imgs[i], denoised_imgs_nlm[i]) for i in range(len(gt_imgs))])

    ssim_gt_mean = np.mean([ssim(gt_imgs[i], noisy_imgs[i], multichannel=True) for i in range(len(gt_imgs))])
    ssim_rednet_mean = np.mean([ssim(gt_imgs[i], denoised_imgs_rednet[i], multichannel=True) for i in range(len(gt_imgs))])
    ssim_mwcnn_mean = np.mean([ssim(gt_imgs[i], denoised_imgs_mwcnn[i], multichannel=True) for i in range(len(gt_imgs))])
    ssim_pridnet_mean = np.mean([ssim(gt_imgs[i], denoised_imgs_pridnet[i], multichannel=True) for i in range(len(gt_imgs))])
    ssim_nlm_mean = np.mean([ssim(gt_imgs[i], denoised_imgs_nlm[i], multichannel=True) for i in range(len(gt_imgs))])

    mse_gt_mean = np.mean([mse(gt_imgs[i], noisy_imgs[i]) for i in range(len(gt_imgs))])
    mse_rednet_mean = np.mean([mse(gt_imgs[i], denoised_imgs_rednet[i]) for i in range(len(gt_imgs))])
    mse_mwcnn_mean = np.mean([mse(gt_imgs[i], denoised_imgs_mwcnn[i]) for i in range(len(gt_imgs))])
    mse_pridnet_mean = np.mean([mse(gt_imgs[i], denoised_imgs_pridnet[i]) for i in range(len(gt_imgs))])
    mse_nlm_mean = np.mean([mse(gt_imgs[i], denoised_imgs_nlm[i]) for i in range(len(gt_imgs))])
    # insert data output here

def single_test(gt_path, noisy_path):
    rednet_model = tf.keras.models.load_model('./models/REDNet.h5')
    mwcnn_model = tf.keras.models.load_model(f'./models/MWCNN.h5', custom_objects={'DWT_downsampling': DWT_downsampling, 'IWT_upsampling': IWT_upsampling, 'Conv_block': Conv_block})
    pridnet_model = tf.keras.models.load_model(f'./models/PRIDNet.h5', custom_objects={'Convolution_block': Convolution_block, 'Channel_attention': Channel_attention, 'Avg_pool_Unet_Upsample_msfe': Avg_pool_Unet_Upsample_msfe, 'Multi_scale_feature_extraction': Multi_scale_feature_extraction, 'Kernel_selecting_module': Kernel_selecting_module})

    gt_image = load_images([gt_path])
    noisy_image = load_images([noisy_path])

    denoised_image_rednet = infer(rednet_model, noisy_image)[0]
    denoised_image_mwcnn = infer(mwcnn_model, noisy_image)[0]
    denoised_image_pridnet = infer(pridnet_model, noisy_image)[0]
    denoised_image_nlm = cv.fastNlMeansDenoisingColored(noisy_image[i], None, 10, 10, 7, 21)

    psnr_gt = psnr(gt_image, noisy_image)
    psnr_rednet = psnr(gt_image, denoised_image_rednet)
    psnr_mwcnn = psnr(gt_image, denoised_image_mwcnn)
    psnr_pridnet = psnr(gt_image, denoised_image_pridnet)
    psnr_nlm = psnr(gt_image, denoised_image_nlm)

    ssim_gt = ssim(gt_image, noisy_image, multichannel=True)
    ssim_rednet = ssim(gt_image, denoised_image_rednet, multichannel=True)
    ssim_mwcnn = ssim(gt_image, denoised_image_mwcnn, multichannel=True)
    ssim_pridnet = ssim(gt_image, denoised_image_pridnet, multichannel=True)
    ssim_nlm = ssim(gt_image, denoised_image_nlm, multichannel=True)

    mse_gt = mse(gt_image, noisy_image)
    mse_rednet = mse(gt_image, denoised_image_rednet)
    mse_mwcnn = mse(gt_image, denoised_image_mwcnn)
    mse_pridnet = mse(gt_image, denoised_image_pridnet)
    mse_nlm = mse(gt_image, denoised_image_nlm)
    # insert data output here

if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'asf:', ['all', 'single', 'file='])
    except getopt.GetoptError as err:
        print(err)
        exit(2)
    
    all = False
    single = False
    file = None

    for o,a in opts,args:
        if o in ('a', '--all'):
            all = True
        elif o in ('s', '--single'):
            single = True
        elif o in ('-f', '--file'):
            file = a

    if all and single:
        raise Exception('Cannot run all and single at the same time')
    if not all and not single:
        raise Exception('Must specify either all or single')
    if single and file == None:
        raise Exception('Must specify file for single')
    
    if all:
        full_test()
    else:
        single_test(file)
