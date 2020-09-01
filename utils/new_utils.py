"""
paper: GridDehazeNet: Attention-Based Multi-Scale Network for Image Dehazing
file: utils.py
about: all utilities
author: Xiaohong Liu
date: 01/08/19
"""

# --- Imports --- #
import time
import torch
import torch.nn.functional as F
import torchvision.utils as utils
from math import log10
from skimage import metrics
from noise_generation import generate_noise
from utils.utils import *
import cv2


def to_psnr(dehaze, gt):
    mse = F.mse_loss(dehaze, gt, reduction='none')
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]

    intensity_max = 2.0
    psnr_list = [20.0 * log10(intensity_max / mse) for mse in mse_list]
    return psnr_list


def to_ssim_skimage(dehaze, gt):
    dehaze_list = torch.split(dehaze, 1, dim=0)
    gt_list = torch.split(gt, 1, dim=0)

    dehaze_list_np = [dehaze_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(dehaze_list))]
    gt_list_np = [gt_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(dehaze_list))]
    ssim_list = [metrics.structural_similarity(dehaze_list_np[ind],  gt_list_np[ind], data_range=1, multichannel=True) for ind in range(len(dehaze_list))]

    return ssim_list


def validation(net, val_data_loader, device, hf_noise_str, lf_noise_str, gain_decay, save_tag=False):
    """
    :param net: GateDehazeNet
    :param val_data_loader: validation loader
    :param device: The GPU that loads the network
    :param save_tag: tag of saving image or not
    :return: average PSNR value
    """
    psnr_list = []

    for batch_id, val_data in enumerate(val_data_loader):

        with torch.no_grad():
            img = val_data
            gain, offset = generate_noise(img, hf_noise_str, lf_noise_str, gain_decay)
            imgn = gain*img + offset

            img = img.to(device)
            imgn = imgn.to(device)
            out = net(imgn)

        # --- Calculate the average PSNR --- #
        psnr_list.extend(to_psnr(out, img))

        # --- Save image --- #
        if save_tag:
            save_image(torch.cat([newlpt(imgn), newlpt(imgn - out), newlpt(out), newlpt(img)], dim=3),
                       image_folder='./val_results/', image_name="val_index_{}".format(batch_id))

    avr_psnr = sum(psnr_list) / len(psnr_list)
    return avr_psnr


def save_image(img, image_folder, image_name):
    dehaze_images = torch.split(img, 1, dim=0)
    batch_num = len(dehaze_images)

    for ind in range(batch_num):
        cv2.imwrite(image_folder + '{}'.format(image_name + '.png'), variable_to_cv2_image(dehaze_images[ind]))


def print_log(epoch, num_epochs, one_epoch_time, train_psnr, val_psnr):
    print('({0:.0f}s) Epoch [{1}/{2}], Train_PSNR:{3:.2f}, Val_PSNR:{4:.2f}'
          .format(one_epoch_time, epoch, num_epochs, train_psnr, val_psnr))

    # --- Write the training log --- #
    # with open('./training_log/{}_log.txt'.format(category), 'a') as f:
    #     print('Date: {0}s, Time_Cost: {1:.0f}s, Epoch: [{2}/{3}], Train_PSNR: {4:.2f}, Val_PSNR: {5:.2f}, Val_SSIM: {6:.4f}'
    #           .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    #                   one_epoch_time, epoch, num_epochs, train_psnr, val_psnr, val_ssim), file=f)


def adjust_learning_rate(optimizer, epoch, step=10, lr_decay=0.5):

    if not epoch % step and epoch > 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay
            print('Learning rate sets to {}.'.format(param_group['lr']))
    else:
        for param_group in optimizer.param_groups:
            print('Learning rate sets to {}.'.format(param_group['lr']))
