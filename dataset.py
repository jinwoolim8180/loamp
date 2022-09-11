# -*- coding: utf-8 -*-
from torch.utils.data import Dataset
import imageio
import os
import torch
import glob
import random
import numpy as np
from functools import lru_cache

ycbcr_from_rgb = torch.tensor([[    65.481,   128.553,    24.966],
                               [   -37.797,   -74.203,     112.0],
                               [     112.0,   -93.786,   -18.214]])
def rgb2ycbcr(rgb):
    arr = torch.mul(rgb.float() / 255.0, ycbcr_from_rgb.transpose(1,0))
    arr[..., 0] += 16
    arr[..., 1] += 128
    arr[..., 2] += 128
    return arr


class SlowDataset(Dataset):
    def __init__(self, args, data_dir, ext, training=True):
        super(SlowDataset, self).__init__()
        self.args = args
        self.training = training
        self.image_folder = os.path.join('.', data_dir)
        self.bin_image_folder = os.path.join('.', data_dir+'bin')
        if not os.path.exists(self.bin_image_folder): os.makedirs(self.bin_image_folder, exist_ok=True)
        self.ext = '/*%s' % ext
        self.file_names = glob.glob(self.image_folder + self.ext)
        self.ext_str = ext
        self.bin_file_names = list()
        self.prepare_cache()

    def prepare_cache(self):
        for fname in self.file_names:
            bin_fname = fname.replace(self.image_folder, self.bin_image_folder).replace(self.ext_str, '.npy')
            self.bin_file_names.append(bin_fname)
            if not os.path.exists(bin_fname):
                img = imageio.imread(fname)
                np.save(bin_fname, img)
                print('{} prepared!'.format(bin_fname))

    def __len__(self):
        if self.training:
            return len(self.file_names) * 200
        else:
            return len(self.file_names) * 900


    @lru_cache(maxsize=400)
    def get_ndarray(self, fname):
        return np.load(fname)

    def __getitem__(self, index):
        rgb_range = self.args.rgb_range
        n_channels = self.args.in_channels

        img = torch.Tensor(self.get_ndarray(self.bin_file_names[index % len(self.file_names)]))

        if img.numpy().ndim == 2:
            img = img.unsqueeze(2)

        c = img.shape[2]
        # input rgb image output y chanel
        if n_channels == 1 and c == 3:
            img = rgb2ycbcr(img)[:, :, 0].unsqueeze(2)
        elif n_channels == 3 and c == 1:
            img = img.repeat(1, 1, 3)

        w, h, _ = img.shape
        th = tw = self.args.patch_size

        if self.training:
            i = random.randint(0, w - tw)
            j = random.randint(0, h - th)
            img = img[i:i + tw, j:j + th, :]

            img_tensor = img.permute(2, 0, 1)
            img_tensor = img_tensor * rgb_range / 255.0

            img_tensor = self.augment(img_tensor)
        else:
            w_pad = tw - np.mod(w, tw)
            h_pad = th - np.mod(h, th)
            img = torch.cat([img, torch.zeros(w, h_pad)], dim=1)
            img = torch.cat([img, torch.zeros(w_pad, h + h_pad)], dim=0)

            img_tensor = img.permute(2, 0, 1)
            img_tensor = img_tensor * rgb_range / 255.0

        return img_tensor


    def augment(self, img, hflip=True, rot=True):

        hflip = hflip and random.random() < 0.5
        vflip = rot and random.random() < 0.5
        rot90 = rot and random.random() < 0.5

        if hflip:
            img = img.flip([1])
        if vflip:
            img = img.flip([0])
        if rot90:
            img = img.permute(0, 2, 1)

        return img

