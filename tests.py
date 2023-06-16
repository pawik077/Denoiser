import tensorflow as tf
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse

import sys
import getopt
import random

from helpers import *
from infer import load_images, infer
from MWCNN import DWT_downsampling, IWT_upsampling, Conv_block
from PRIDNet import Convolution_block, Channel_attention, Avg_pool_Unet_Upsample_msfe, Multi_scale_feature_extraction, Kernel_selecting_module

def full_test(datasets):
    rednet_model = tf.keras.models.load_model('./models/REDNet.h5')
    dncnn_model = tf.keras.models.load_model('./models/DnCNN.h5')
    mwcnn_model = tf.keras.models.load_model(f'./models/MWCNN.h5', custom_objects={'DWT_downsampling': DWT_downsampling, 'IWT_upsampling': IWT_upsampling, 'Conv_block': Conv_block})
    pridnet_model = tf.keras.models.load_model(f'./models/PRIDNet.h5', custom_objects={'Convolution_block': Convolution_block, 'Channel_attention': Channel_attention, 'Avg_pool_Unet_Upsample_msfe': Avg_pool_Unet_Upsample_msfe, 'Multi_scale_feature_extraction': Multi_scale_feature_extraction, 'Kernel_selecting_module': Kernel_selecting_module})

    gt_paths, noisy_paths = get_img_paths(datasets)
    gt_imgs = load_images(gt_paths)
    noisy_imgs = load_images(noisy_paths)
    
    denoised_imgs_rednet = infer(rednet_model, noisy_imgs)
    denoised_imgs_dncnn = infer(dncnn_model, noisy_imgs)
    denoised_imgs_mwcnn = infer(mwcnn_model, noisy_imgs)
    denoised_imgs_pridnet = infer(pridnet_model, noisy_imgs)
    denoised_imgs_nlm = np.asarray([cv.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21) for img in noisy_imgs])

    psnr_gt_mean = np.mean([psnr(gt_imgs[i], noisy_imgs[i]) for i in range(len(gt_imgs))])
    psnr_rednet_mean = np.mean([psnr(gt_imgs[i], denoised_imgs_rednet[i]) for i in range(len(gt_imgs))])
    psnr_dncnn_mean = np.mean([psnr(gt_imgs[i], denoised_imgs_dncnn[i]) for i in range(len(gt_imgs))])
    psnr_mwcnn_mean = np.mean([psnr(gt_imgs[i], denoised_imgs_mwcnn[i]) for i in range(len(gt_imgs))])
    psnr_pridnet_mean = np.mean([psnr(gt_imgs[i], denoised_imgs_pridnet[i]) for i in range(len(gt_imgs))])
    psnr_nlm_mean = np.mean([psnr(gt_imgs[i], denoised_imgs_nlm[i]) for i in range(len(gt_imgs))])

    ssim_gt_mean = np.mean([ssim(gt_imgs[i], noisy_imgs[i], channel_axis=-1, data_range=noisy_imgs[i].max() - noisy_imgs[i].min()) for i in range(len(gt_imgs))])
    ssim_rednet_mean = np.mean([ssim(gt_imgs[i], denoised_imgs_rednet[i], channel_axis=-1, data_range=denoised_imgs_rednet[i].max() - denoised_imgs_rednet[i].min()) for i in range(len(gt_imgs))])
    ssim_dncnn_mean = np.mean([ssim(gt_imgs[i], denoised_imgs_dncnn[i], channel_axis=-1, data_range=denoised_imgs_dncnn[i].max() - denoised_imgs_dncnn[i].min()) for i in range(len(gt_imgs))])
    ssim_mwcnn_mean = np.mean([ssim(gt_imgs[i], denoised_imgs_mwcnn[i], channel_axis=-1, data_range=denoised_imgs_mwcnn[i].max() - denoised_imgs_mwcnn[i].min()) for i in range(len(gt_imgs))])
    ssim_pridnet_mean = np.mean([ssim(gt_imgs[i], denoised_imgs_pridnet[i], channel_axis=-1, data_range=denoised_imgs_pridnet[i].max() - denoised_imgs_pridnet[i].min()) for i in range(len(gt_imgs))])
    ssim_nlm_mean = np.mean([ssim(gt_imgs[i], denoised_imgs_nlm[i], channel_axis=-1, data_range=denoised_imgs_nlm[i].max() - denoised_imgs_nlm[i].min()) for i in range(len(gt_imgs))])

    mse_gt_mean = np.mean([mse(gt_imgs[i], noisy_imgs[i]) for i in range(len(gt_imgs))])
    mse_rednet_mean = np.mean([mse(gt_imgs[i], denoised_imgs_rednet[i]) for i in range(len(gt_imgs))])
    mse_dncnn_mean = np.mean([mse(gt_imgs[i], denoised_imgs_dncnn[i]) for i in range(len(gt_imgs))])
    mse_mwcnn_mean = np.mean([mse(gt_imgs[i], denoised_imgs_mwcnn[i]) for i in range(len(gt_imgs))])
    mse_pridnet_mean = np.mean([mse(gt_imgs[i], denoised_imgs_pridnet[i]) for i in range(len(gt_imgs))])
    mse_nlm_mean = np.mean([mse(gt_imgs[i], denoised_imgs_nlm[i]) for i in range(len(gt_imgs))])
    # insert data output here

    print('Saving results...')
    testname = ''
    for dataset in datasets:
        testname += dataset + '_'

    plt.figure()
    plt.title('PSNR')
    plt.bar(['GT', 'REDNet', 'DnCNN', 'MWCNN', 'PRIDNet', 'NLM'], [psnr_gt_mean, psnr_rednet_mean, psnr_dncnn_mean, psnr_mwcnn_mean, psnr_pridnet_mean, psnr_nlm_mean])
    plt.savefig(f'./test/{testname}psnr.png')
    plt.figure()
    plt.title('SSIM')
    plt.bar(['GT', 'REDNet', 'DnCNN', 'MWCNN', 'PRIDNet', 'NLM'], [ssim_gt_mean, ssim_rednet_mean, ssim_dncnn_mean, ssim_mwcnn_mean, ssim_pridnet_mean, ssim_nlm_mean])
    plt.savefig(f'./test/{testname}ssim.png')
    plt.figure()
    plt.title('MSE')
    plt.bar(['GT', 'REDNet', 'DnCNN', 'MWCNN', 'PRIDNet', 'NLM'], [mse_gt_mean, mse_rednet_mean, mse_dncnn_mean, mse_mwcnn_mean, mse_pridnet_mean, mse_nlm_mean])
    plt.savefig(f'./test/{testname}mse.png')

    with open(f'./test/{testname}results.csv', 'w') as f:
        f.write('Model,PSNR,SSIM,MSE\n')
        f.write(f'GT,{psnr_gt_mean},{ssim_gt_mean},{mse_gt_mean}\n')
        f.write(f'REDNet,{psnr_rednet_mean},{ssim_rednet_mean},{mse_rednet_mean}\n')
        f.write(f'DnCNN,{psnr_dncnn_mean},{ssim_dncnn_mean},{mse_dncnn_mean}\n')
        f.write(f'MWCNN,{psnr_mwcnn_mean},{ssim_mwcnn_mean},{mse_mwcnn_mean}\n')
        f.write(f'PRIDNet,{psnr_pridnet_mean},{ssim_pridnet_mean},{mse_pridnet_mean}\n')
        f.write(f'NLM,{psnr_nlm_mean},{ssim_nlm_mean},{mse_nlm_mean}\n')

    choices = random.choices(range(len(gt_imgs)), k=5)
    for i in choices:
        gt_img = cv.cvtColor(gt_imgs[i], cv.COLOR_RGB2BGR)
        noisy_img = cv.cvtColor(noisy_imgs[i], cv.COLOR_RGB2BGR)
        denoised_img_rednet = cv.cvtColor(denoised_imgs_rednet[i], cv.COLOR_RGB2BGR)
        denoised_img_dncnn = cv.cvtColor(denoised_imgs_dncnn[i], cv.COLOR_RGB2BGR)
        denoised_img_mwcnn = cv.cvtColor(denoised_imgs_mwcnn[i], cv.COLOR_RGB2BGR)
        denoised_img_pridnet = cv.cvtColor(denoised_imgs_pridnet[i], cv.COLOR_RGB2BGR)
        denoised_img_nlm = cv.cvtColor(denoised_imgs_nlm[i], cv.COLOR_RGB2BGR)
        cv.imwrite(f'./test/{testname}img{i}_gt.png', gt_img)
        cv.imwrite(f'./test/{testname}img{i}_noisy.png', noisy_img)
        cv.imwrite(f'./test/{testname}img{i}_rednet.png', denoised_img_rednet)
        cv.imwrite(f'./test/{testname}img{i}_dncnn.png', denoised_img_dncnn)
        cv.imwrite(f'./test/{testname}img{i}_mwcnn.png', denoised_img_mwcnn)
        cv.imwrite(f'./test/{testname}img{i}_pridnet.png', denoised_img_pridnet)
        cv.imwrite(f'./test/{testname}img{i}_nlm.png', denoised_img_nlm)
    print('Done!')

def single_test(gt_path, noisy_path):
    rednet_model = tf.keras.models.load_model('./models/REDNet.h5')
    dncnn_model = tf.keras.models.load_model('./models/DnCNN.h5')
    mwcnn_model = tf.keras.models.load_model(f'./models/MWCNN.h5', custom_objects={'DWT_downsampling': DWT_downsampling, 'IWT_upsampling': IWT_upsampling, 'Conv_block': Conv_block})
    pridnet_model = tf.keras.models.load_model(f'./models/PRIDNet.h5', custom_objects={'Convolution_block': Convolution_block, 'Channel_attention': Channel_attention, 'Avg_pool_Unet_Upsample_msfe': Avg_pool_Unet_Upsample_msfe, 'Multi_scale_feature_extraction': Multi_scale_feature_extraction, 'Kernel_selecting_module': Kernel_selecting_module})

    gt_image = load_images([gt_path])[0]
    noisy_image = load_images([noisy_path])

    denoised_image_rednet = infer(rednet_model, noisy_image)[0]
    denoised_image_dncnn = infer(dncnn_model, noisy_image)[0]
    denoised_image_mwcnn = infer(mwcnn_model, noisy_image)[0]
    denoised_image_pridnet = infer(pridnet_model, noisy_image)[0]
    denoised_image_nlm = cv.fastNlMeansDenoisingColored(noisy_image[0], None, 10, 10, 7, 21)

    psnr_gt = psnr(gt_image, noisy_image[0])
    psnr_rednet = psnr(gt_image, denoised_image_rednet)
    psnr_dncnn = psnr(gt_image, denoised_image_dncnn)
    psnr_mwcnn = psnr(gt_image, denoised_image_mwcnn)
    psnr_pridnet = psnr(gt_image, denoised_image_pridnet)
    psnr_nlm = psnr(gt_image, denoised_image_nlm)

    ssim_gt = ssim(gt_image, noisy_image[0], channel_axis=-1, data_range=noisy_image[0].max() - noisy_image[0].min())
    ssim_rednet = ssim(gt_image, denoised_image_rednet, channel_axis=-1, data_range=denoised_image_rednet.max() - denoised_image_rednet.min())
    ssim_dncnn = ssim(gt_image, denoised_image_dncnn, channel_axis=-1, data_range=denoised_image_dncnn.max() - denoised_image_dncnn.min())
    ssim_mwcnn = ssim(gt_image, denoised_image_mwcnn, channel_axis=-1, data_range=denoised_image_mwcnn.max() - denoised_image_mwcnn.min())
    ssim_pridnet = ssim(gt_image, denoised_image_pridnet, channel_axis=-1, data_range=denoised_image_pridnet.max() - denoised_image_pridnet.min())
    ssim_nlm = ssim(gt_image, denoised_image_nlm, channel_axis=-1, data_range=denoised_image_nlm.max() - denoised_image_nlm.min())

    mse_gt = mse(gt_image, noisy_image[0])
    mse_rednet = mse(gt_image, denoised_image_rednet)
    mse_dncnn = mse(gt_image, denoised_image_dncnn)
    mse_mwcnn = mse(gt_image, denoised_image_mwcnn)
    mse_pridnet = mse(gt_image, denoised_image_pridnet)
    mse_nlm = mse(gt_image, denoised_image_nlm)
    # insert data output here

    print('Saving results...')
    plt.figure()
    plt.title('PSNR')
    plt.bar(['GT', 'REDNet', 'DnCNN', 'MWCNN', 'PRIDNet', 'NLM'], [psnr_gt, psnr_rednet, psnr_dncnn, psnr_mwcnn, psnr_pridnet, psnr_nlm])
    plt.savefig(f'./test/single_psnr.png')
    plt.figure()
    plt.title('SSIM')
    plt.bar(['GT', 'REDNet', 'DnCNN', 'MWCNN', 'PRIDNet', 'NLM'], [ssim_gt, ssim_rednet, ssim_dncnn, ssim_mwcnn, ssim_pridnet, ssim_nlm])
    plt.savefig(f'./test/single_ssim.png')
    plt.figure()
    plt.title('MSE')
    plt.bar(['GT', 'REDNet', 'DnCNN', 'MWCNN', 'PRIDNet', 'NLM'], [mse_gt, mse_rednet, mse_dncnn, mse_mwcnn, mse_pridnet, mse_nlm])
    plt.savefig(f'./test/single_mse.png')

    with open('./test/single_results.csv', 'w') as f:
        f.write('Model,PSNR,SSIM,MSE\n')
        f.write(f'GT,{psnr_gt},{ssim_gt},{mse_gt}\n')
        f.write(f'REDNet,{psnr_rednet},{ssim_rednet},{mse_rednet}\n')
        f.write(f'DnCNN,{psnr_dncnn},{ssim_dncnn},{mse_dncnn}\n')
        f.write(f'MWCNN,{psnr_mwcnn},{ssim_mwcnn},{mse_mwcnn}\n')
        f.write(f'PRIDNet,{psnr_pridnet},{ssim_pridnet},{mse_pridnet}\n')
        f.write(f'NLM,{psnr_nlm},{ssim_nlm},{mse_nlm}\n')

    denoised_image_rednet = cv.cvtColor(denoised_image_rednet, cv.COLOR_RGB2BGR)
    denoised_image_dncnn = cv.cvtColor(denoised_image_dncnn, cv.COLOR_RGB2BGR)
    denoised_image_mwcnn = cv.cvtColor(denoised_image_mwcnn, cv.COLOR_RGB2BGR)
    denoised_image_pridnet = cv.cvtColor(denoised_image_pridnet, cv.COLOR_RGB2BGR)
    denoised_image_nlm = cv.cvtColor(denoised_image_nlm, cv.COLOR_RGB2BGR)
    cv.imwrite('./test/single_rednet.png', denoised_image_rednet)
    cv.imwrite('./test/single_dncnn.png', denoised_image_dncnn)
    cv.imwrite('./test/single_mwcnn.png', denoised_image_mwcnn)
    cv.imwrite('./test/single_pridnet.png', denoised_image_pridnet)
    cv.imwrite('./test/single_nlm.png', denoised_image_nlm)
    print('Done!')

if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'asf:g:d:rp', ['all', 'single', 'file=', 'gt=', 'renoir', 'polyu'])
    except getopt.GetoptError as err:
        print(err)
        exit(2)
    
    all = False
    single = False
    file = None
    gt = None
    datasets = []

    for o,a in opts:
        if o in ('-a', '--all'):
            all = True
        elif o in ('-s', '--single'):
            single = True
        elif o in ('-f', '--file'):
            file = a
        elif o in ('-g', '--gt'):
            gt = a
        elif o in ('-r', '--renoir'):
            datasets.append('RENOIR')
        elif o in ('-p', '--polyu'):
            datasets.append('PolyU')

    if all and single:
        raise Exception('Cannot run all and single at the same time')
    if not all and not single:
        raise Exception('Must specify either all or single')
    if single and (file == None or gt == None):
        raise Exception('Must specify file for single')
    if all and len(datasets) == 0:
        raise Exception('Must specify at least one dataset for all')
    if single and len(datasets) > 0:
        raise Exception('Cannot specify dataset for single')
    
    if all:
        full_test(datasets)
    else:
        single_test(gt, file)
