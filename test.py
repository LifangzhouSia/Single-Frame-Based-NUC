
import cv2
import os
import time
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.dataset import prepare_data, Dataset
from torch.autograd import Variable
from utils.utils import *
from utils.new_utils import *


# --- load model --- #
# from models.lf_first_nuc_models import rdp_nuc
# from models.model_easy import rdp_nuc
# from models.residual_nuc_models import rdp_nuc
from models.sfb_lfnu_nuc_models import DRNet


# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description="DnCNN_Test")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--logdir", type=str, default="logs", help='path of log files')
parser.add_argument("--batchSize_val", type=int, default=1, help="Validation batch size")
parser.add_argument("--net_dir", type=str, default="net_records", help="path of net files")
parser.add_argument("--rnd_str", type=float, default="30", help="Random noise strength")
parser.add_argument("--stp_str", type=float, default="30", help="Stripe noise strength")
parser.add_argument("--lfnu_str", type=float, default="2", help="Low-frequency Non-Uniformity strength")
parser.add_argument("--preprocess", type=bool, default=True, help="Prepare data or not")
parser.add_argument("--gain_decay", type=float, default="0.1", help="sigma_g = gain_decay * sigma_o")
opt = parser.parse_args()


def main():
    # --- Gpu device --- #
    device_ids = [Id for Id in range(torch.cuda.device_count())]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # --- Validation data loader --- #
    dataset_val = Dataset(train=False)
    loader_val = DataLoader(dataset_val, num_workers=4, batch_size=opt.batchSize_val, shuffle=False)

    # Building net
    print('Loading net ...\n')
    net = DRNet()

    # --- Multi-GPU --- #
    net = net.to(device)
    net = nn.DataParallel(net, device_ids=device_ids)

    # --- Load the network weight --- #
    checkpoint = torch.load(os.path.join(opt.net_dir, 'state'))
    net.load_state_dict(checkpoint['net'])

    # --- Use the evaluation model in testing --- #
    net.eval()
    print('--- Testing starts! ---')
    start_time = time.time()
    hf_noise_str = [35, 55]
    lf_noise_str = [1.5, 2]
    val_psnr = validation(net, loader_val, device, hf_noise_str, lf_noise_str, gain_decay=opt.gain_decay, save_tag=True)
    end_time = time.time() - start_time
    print('val_psnr: {0:.2f}'.format(val_psnr))
    print('validation time is {0:.4f}s'.format(end_time))


if __name__ == "__main__":
    if opt.preprocess:
        prepare_data(data_path=os.path.join('data', 'MOT17Det'),
                     patch_size=[480, 640], patch_stride=[240, 320], frame_length=15, frame_stride=200,
                     val_patch_size=[480, 640], val_patch_stride=[480, 640], val_frame_length=200, val_frame_stride=300,
                     aug_times=1, only_val=True)
    main()
