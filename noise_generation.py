import os
import cv2
import numpy as np
import torch
import torchvision
import scipy.io as sio
from utils.utils import newlp


def generate_noise(x, hf_noise_str, lf_noise_str, gain_decay=0.1):

    gain = torch.zeros(x.size())
    offset = torch.zeros(x.size())

    batch_num, features, hei, wid = x.size()
    hf_noise_tab = (hf_noise_str[1] - hf_noise_str[0]) * (np.random.rand(batch_num)) + hf_noise_str[0]
    lf_noise_tab = (lf_noise_str[1] - lf_noise_str[0]) * (np.random.rand(batch_num)) + lf_noise_str[0]

    for k in range(batch_num):
        gain[k, :, :, :], offset[k, :, :, :] \
            = frame_generate_noise(x[k:k+1, :, :, :], hf_noise_tab[k], hf_noise_tab[k], lf_noise_tab[k], gain_decay)

    return gain, offset


def frame_generate_noise(x, rnd_str, stp_str, lfnu_str, gain_decay):

    random_offset = torch.FloatTensor(x.size()).normal_(mean=0, std=rnd_str/255.)
    stripe_offset = torch.FloatTensor(torch.Size([1, 1, 1, x.size()[3]])).normal_(mean=0, std=stp_str/255.)
    stripe_offset = stripe_offset.repeat(1, 1, x.shape[2], 1)
    lowfrequency_offset = generate_lfnu(x, lfnu_str, disturbance=0.2)
    offset = random_offset + stripe_offset + lowfrequency_offset

    random_gain = torch.FloatTensor(x.size()).normal_(mean=0, std=gain_decay*rnd_str/255.)
    stripe_gain = torch.FloatTensor(torch.Size([1, 1, 1, x.size()[3]])).normal_(mean=0, std=gain_decay*stp_str/255.)
    stripe_gain = stripe_gain.repeat(1, 1, x.shape[2], 1)
    lowfrequency_gain = generate_lfnu(x, gain_decay*lfnu_str, disturbance=0.2)
    gain = random_gain + stripe_gain + lowfrequency_gain + 1
    
    return gain, offset
    

def generate_lfnu(img, lfnu_str, disturbance=0.05):

    img = img.numpy()
    OUT = np.zeros(img.shape)

    x_grid = np.arange(1, img.shape[3]+1)
    y_grid = np.arange(1, img.shape[2]+1)
    x, y = np.meshgrid(x_grid, y_grid)
    x = x.astype(np.float32)
    y = y.astype(np.float32)

    # --- Estimated 4-order polynomial model of Low-Frequency Non-Uniformity from blackbody images --- #
    matfn = './bias_seed.mat'
    bias_seed_mat = sio.loadmat(matfn)
    bias_seed = bias_seed_mat['bias_seed'].astype(np.float32)

    # --- Re-generate LFNU with disturbed parameters --- #
    for i in range(img.shape[0]):
        bias_disturbance = np.random.normal(1, disturbance, bias_seed.shape)
        bias_corrupted = bias_seed * bias_disturbance

        cnt = 0
        lf = np.zeros(x.shape)
        lf = lf.astype(np.float32)
        for p in range(1, 5):
            for q in range(0, p + 1):
                lf = lf + bias_corrupted[cnt] * pow(y, q) * pow(x, p - q)
                cnt += 1

        # cv2.imshow('lf', newlp(lf))
        # cv2.waitKey(0)

        mean_lf = lf.mean()
        std_lf = lf.std()
        lfnu = (lf - mean_lf) / std_lf * lfnu_str

        OUT[i, :, :, :] = lfnu

    return torch.FloatTensor(OUT)


# def generate_lfnu(img, lfnu_str, disturbance=0.7):
#     img = img.numpy()
#     OUT = np.zeros(img.shape)
#
#     x_grid = np.arange(1, img.shape[3] + 1)
#     y_grid = np.arange(1, img.shape[2] + 1)
#     x, y = np.meshgrid(x_grid, y_grid)
#
#     # --- Estimated 4-order polynomial model of low-frequency Nonuniformity from blackbody images --- #
#     bias_seed = np.array([6.26225735334367e-08,
#                           -6.29692596662619e-08,
#                           -6.87663828333308e-08,
#                           -4.71347026597265e-08,
#                           1.95539477810700e-05,
#                           2.75439220810853e-05,
#                           1.34216291876528e-05,
#                           -0.00340162133902158,
#                           -0.00218894348159150,
#                           0.871550929832063])
#
#     # --- Re-generate lfnu with disturbed parameters --- #
#     for i in range(img.shape[0]):
#         bias_randn = np.random.normal(0, 1, bias_seed.shape) * disturbance * bias_seed
#         bias_corrupted = bias_seed + bias_randn
#
#         lowFrequencyNoise = bias_corrupted[0] * x * x * x + bias_corrupted[1] * y * y * y \
#                             + bias_corrupted[2] * x * x * y + bias_corrupted[3] * x * y * y \
#                             + bias_corrupted[4] * x * x + bias_corrupted[5] * y * y + bias_corrupted[6] * x * y \
#                             + bias_corrupted[7] * x + bias_corrupted[8] * y + bias_corrupted[9] * np.ones(
#             (img.shape[2], img.shape[3]))
#
#         mean_LF = lowFrequencyNoise.mean()
#         std_LF = lowFrequencyNoise.std()
#         lfnu = (lowFrequencyNoise - mean_LF) / std_LF * lfnu_str
#
#         OUT[i, :, :, :] = lfnu
#
#     return torch.FloatTensor(OUT)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    