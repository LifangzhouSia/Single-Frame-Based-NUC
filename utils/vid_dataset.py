import os
import os.path
import numpy as np
import random
import h5py
import torch
import cv2
import glob
import torch.utils.data as udata
from utils.utils import data_augmentation


def normalize(data):
    out = (data - data.mean()) / data.std()
    return out


def vid2cubes(vid, patch_size, patch_stride):
    # vid -> frame*hei*wid
    # Y   -> frame*patch_size*patch_size*Num_of_Patches
    k = 0
    frame_num, hei, wid = vid.shape
    patch_hei, patch_wid = patch_size
    stride_hei, stride_wid = patch_stride
    patch = vid[:, 0:hei - patch_hei + 0 + 1:stride_hei, 0:wid - patch_wid + 0 + 1:stride_wid]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([frame_num, patch_hei * patch_wid, TotalPatNum], np.float32)
    for i in range(patch_hei):
        for j in range(patch_wid):
            cubes = vid[:, i:hei - patch_hei + i + 1:stride_hei, j:wid - patch_wid + j + 1:stride_wid]
            Y[:, k, :] = np.array(cubes[:]).reshape(frame_num, TotalPatNum)
            k = k + 1
    return Y.reshape([frame_num, patch_hei, patch_wid, TotalPatNum])


def prepare_data(data_path, patch_size, patch_stride, frame_length, frame_stride,
                 val_patch_stride, val_frame_length, val_frame_stride, aug_times=1, only_val=False):

    if not only_val:

        # --- Training Data Preparation --- #
        print('Start processing training data.')
        h5f = h5py.File('train.h5', 'w')
        # scales = [1, 0.9, 0.8, 0.7, 0.6]
        scales = [1]

        train_vid_list = os.path.join(data_path, 'train', 'train_list.txt')
        with open(train_vid_list) as f:
            contents = f.readlines()
            vid_list = [i.strip() for i in contents]
        f.close()

        train_num = 0
        for vid_name in vid_list:

            files = glob.glob(os.path.join(data_path, 'train', vid_name, 'img1', '*.jpg'))
            files.sort()
            x = cv2.imread(files[0])
            hei, wid, chn = x.shape

            for s in range(len(scales)):

                for k in range(0, len(files) - frame_length, frame_stride):

                    scaled_hei = int(hei * scales[s])
                    scaled_wid = int(wid * scales[s])

                    vid = np.zeros([frame_length, scaled_hei, scaled_wid], np.float32)

                    cnt = 0
                    for g in range(k, k + frame_length):
                        img = cv2.imread(files[g])
                        img = cv2.resize(img[:, :, 0], (scaled_wid, scaled_hei), interpolation=cv2.INTER_CUBIC)
                        img = np.float32(normalize(img))
                        vid[cnt, :, :] = img
                        cnt += 1

                    cubes = vid2cubes(vid, patch_size, patch_stride)
                    print('frame: %s->%s scale: %.1f samples %d' % (
                    k, k + frame_length - 1, scales[s], cubes.shape[3] * aug_times))

                    for h in range(cubes.shape[3]):
                        data = cubes[:, :, :, h].copy()
                        h5f.create_dataset(str(train_num), data=data)
                        train_num += 1
                        for m in range(aug_times - 1):
                            data_aug = data_augmentation(data, np.random.randint(1, 8))
                            h5f.create_dataset(str(train_num) + '_aug_%d' % (m + 1), data=data_aug)
                            train_num += 1

            files.clear()
        h5f.close()
        print('Training set, # sample {}\n'.format(train_num))

    # --- Validation Data Preparation --- #
    print('Start validation processing')
    h5f = h5py.File('val.h5', 'w')

    val_vid_list = os.path.join(data_path, 'test', 'val_list.txt')
    with open(val_vid_list) as f:
        contents = f.readlines()
        vid_list = [i.strip() for i in contents]
    f.close()

    val_num = 0
    for vid_name in vid_list:

        files = glob.glob(os.path.join(data_path, 'test', vid_name, 'img1', '*.jpg'))
        files.sort()
        x = cv2.imread(files[0])
        hei, wid, chn = x.shape

        for k in range(0, len(files) - val_frame_length, val_frame_stride):

            vid = np.zeros([val_frame_length, hei, wid], np.float32)

            cnt = 0
            for g in range(k, k + val_frame_length):
                img = cv2.imread(files[g])
                img = np.float32(normalize(img[:, :, 0]))
                vid[cnt, :, :] = img
                cnt += 1

            cubes = vid2cubes(vid, patch_size, val_patch_stride)

            for h in range(cubes.shape[3]):
                data = cubes[:, :, :, h].copy()
                h5f.create_dataset(str(val_num), data=data)
                val_num += 1
                for m in range(aug_times - 1):
                    data_aug = data_augmentation(data, np.random.randint(1, 8))
                    h5f.create_dataset(str(val_num) + '_aug_%d' % (m + 1), data=data_aug)
                    val_num += 1

        files.clear()
    h5f.close()
    print('Validation set, # sample {}\n'.format(val_num))


class Dataset(udata.Dataset):
    def __init__(self, train=True):
        super(Dataset, self).__init__()
        self.train = train
        if self.train:
            h5f = h5py.File('train.h5', 'r')
        else:
            h5f = h5py.File('val.h5', 'r')
        self.keys = list(h5f.keys())
        random.shuffle(self.keys)
        h5f.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        if self.train:
            h5f = h5py.File('train.h5', 'r')
        else:
            h5f = h5py.File('val.h5', 'r')
        key = self.keys[index]
        data = np.array(h5f[key])
        h5f.close()
        return torch.Tensor(data)
