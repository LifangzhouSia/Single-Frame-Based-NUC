"""
file: train.py
about: training a single-frame-based network for NUC
author: Fangzhou Li
date: 02/05/20
"""

import time
import os
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from utils.dataset import prepare_data, Dataset
from utils.utils import *
from utils.new_utils import *
from noise_generation import generate_noise


# --- load model --- #
# from models.delfnu_models import rdp_nuc
# from models.nuc_models import rdp_nuc
# from models.model_easy import rdp_nuc
# from models.residual_nuc_models import rdp_nuc
# from models.dense_nuc_models import dense_nuc_model
# from models.new_models import AUNet
# from models.sfb_unet_models import SFB_DCUnet
# from models.sfb_dual_regular_models2 import DRNet
# from models.sfb_fpn_nuc_models import DRNet
from models.sfb_lfnu_nuc_models import DRNet

# --- Parse hyper-param eters  --- #
parser = argparse.ArgumentParser(description="Hyper-parameters for Single-frame-based Deep NUC")
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--batchSize_train", type=int, default=8, help="Training batch size")
parser.add_argument("--batchSize_val", type=int, default=1, help="Validation batch size")
parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--lr_step", type=float, default=30, help="Initial learning rate")
parser.add_argument("--outf", type=str, default="logs", help='path of log files')
parser.add_argument("--net_dir", type=str, default="net_records", help="path of net files")
parser.add_argument("--gain_decay", type=float, default="0.1", help="sigma_g = gain_decay * sigma_o")
opt = parser.parse_args()


# --- Main program  --- #
def main():

    # --- Load dataset --- #
    print('Loading dataset ...\n')
    dataset_train = Dataset(train=True)
    dataset_val = Dataset(train=False)
    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batchSize_train, shuffle=True)
    loader_val = DataLoader(dataset=dataset_val,   num_workers=4, batch_size=opt.batchSize_val,   shuffle=False)
    print("# of training samples: %d\n" % int(len(dataset_train)))

    # --- Gpu device --- #
    device_ids = [Id for Id in range(torch.cuda.device_count())]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Build model
    net = DRNet()

    # --- Move to GPU  --- #
    net = net.to(device)
    net = nn.DataParallel(net, device_ids=device_ids)

    # --- Optimizer  --- #
    optimizer = optim.Adam(net.parameters(), lr=opt.lr)

    # --- Load the network weight --- #
    try:
        checkpoint = torch.load(os.path.join(opt.net_dir, 'state'))
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        init_epoch = checkpoint['epoch']
        print('--- weight loaded ---\n'+'epoch: {}'.format(init_epoch))
    except:
        init_epoch = 0
        print('--- no weight loaded ---')

    # --- Calculate all trainable parameters in network --- #
    pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Total_params: {}".format(pytorch_total_params))

    # --- Previous PSNR and SSIM in testing --- #
    # old_val_psnr, old_val_ssim = validation(net, loader_val, device, rnd_str=opt.rnd_str, stp_str=opt.stp_str, lfnu_str=opt.lfnu_str)
    # print('old_val_psnr: {0:.2f}, old_val_ssim: {1:.4f}'.format(old_val_psnr, old_val_ssim))

    # --- Start training --- #
    for epoch in range(init_epoch, opt.epochs+1):

        psnr_list = []
        start_time = time.time()
        adjust_learning_rate(optimizer, epoch, step=opt.lr_step)

        hf_noise_str = [35, 55]
        lf_noise_str = [1, 2]
        # lf_noise_str = [1, 2.5]
        # --- Main training --- #
        for i, data in enumerate(loader_train, 0):

            # --- Zero the parameter gradients --- #
            net.zero_grad()
            optimizer.zero_grad()
            img_train = data

            gain, offset = generate_noise(img_train, hf_noise_str, lf_noise_str, opt.gain_decay)
            imgn_train = gain*img_train + offset
            img_train, imgn_train = Variable(img_train.cuda()), Variable(imgn_train.cuda())

            # --- Forward + Backward + Optimize --- #
            net.train()
            out_train = net(imgn_train)

            # --- Calculate loss --- #
            loss = F.mse_loss(out_train, img_train, reduction='mean')

            loss.backward()
            optimizer.step()

            # --- Calculate average PSNR --- #
            psnr_list.extend(to_psnr(out_train, img_train))

            # --- Calculate PSNR --- #
            net.eval()
            psnr_train = np.mean(to_psnr(out_train, img_train))
            print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" %
                  (epoch + 1, i + 1, len(loader_train), loss.item(), psnr_train))

            train_peephole = True
            if train_peephole:
                save_image(torch.cat(
                    [newlpt(imgn_train[0:1, :, :, :]), newlpt(imgn_train[0:1, :, :, :] - out_train[0:1, :, :, :]),
                     newlpt(out_train[0:1, :, :, :]), newlpt(img_train[0:1, :, :, :])], dim=3),
                           image_folder='./train_results/', image_name="train_index_{}".format(i))

        # --- Calculate the average training PSNR in one epoch --- #
        train_psnr = sum(psnr_list) / len(psnr_list)

        # --- Save net parameters --- #
        state = {'net': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        torch.save(state, os.path.join(opt.net_dir, 'state'))

        # --- Validation at the end of each epoch--- #
        net.eval()

        val_psnr = validation(net, loader_val, device, hf_noise_str, lf_noise_str, gain_decay=opt.gain_decay, save_tag=True)
        one_epoch_time = time.time() - start_time
        print_log(epoch + 1, opt.epochs, one_epoch_time, train_psnr, val_psnr)

        # --- Update the best net --- #
        # if val_psnr >= old_val_psnr:
        #     torch.save(net.state_dict(), os.path.join(opt.net_dir, 'best_net.pth'))
        #     old_val_psnr = val_psnr


if __name__ == "__main__":
    if opt.preprocess:
        prepare_data(data_path=os.path.join('data', 'MOT17Det'),
                     patch_size=[240, 320], patch_stride=[240, 320], frame_length=15, frame_stride=200,
                     val_patch_size=[240, 320], val_patch_stride=[480, 640], val_frame_length=300, val_frame_stride=300,
                     aug_times=1, only_val=False)
    main()
