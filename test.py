import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from argparse import ArgumentParser

from dataset import SlowDataset
from model.amp import LOAMP
from utils import SSIM, psnr


def load_model(args, model):
    model.load_state_dict(torch.load("./{}/best_model.pkl".format(args.model_dir)))


def main(args):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_list
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cs_channels = int(args.patch_size * args.patch_size * args.in_channels * args.cs_ratio)
    model = LOAMP(args.in_channels, cs_channels, args.n_channels, args.n_stage, args.patch_size)
    model = nn.DataParallel(model)
    model = model.to(device)
    load_model(args, model)
    model.eval()

    test_dataset = SlowDataset(args, '.tif', 'data/test/Set11')
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    ssim = SSIM(args.in_channels)

    with torch.no_grad():
        psnr_loss = 0
        ssim_loss = 0
        num = 0
        pbar = tqdm(test_dataloader)
        for i, data in enumerate(pbar):
            x = data.view(-1, args.in_channels, args.patch_size, args.patch_size)
            x = x.to(device)
            x_hat = model(x)

            psnr_loss += psnr(x, x_hat)
            ssim_loss += ssim(x, x_hat)
            num += 1

            pbar.set_description("PSNR: {0}, SSIM: {1}".format(psnr_loss, ssim_loss))

        psnr_loss /= num
        ssim_loss /= num
        print("PSNR: {0} / SSIM: {1}".format(psnr_loss, ssim_loss))


if __name__ == '__main__':
    parser = ArgumentParser(description='AMP with learned onsager term')
    parser.add_argument('--epoch', type=int, default=400, help='epoch number of training')
    parser.add_argument('--n_stage', type=int, default=20, help='number of stages of loamp')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers')
    parser.add_argument('--group_num', type=int, default=1, help='group number for training')
    parser.add_argument('--cs_ratio', type=float, default=0.1, help='sampling ratio')
    parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')
    parser.add_argument('--test_dir', type=str, default='data/test', help='training data directory')
    parser.add_argument('--rgb_range', type=int, default=1, help='value range 1 or 255')
    parser.add_argument('--in_channels', type=int, default=1, help='1 for gray, 3 for color')
    parser.add_argument('--n_channels', type=int, default=32, help='channels for deep learning')
    parser.add_argument('--patch_size', type=int, default=33, help='from {1, 4, 10, 25, 40, 50}')

    parser.add_argument('--model_dir', type=str, default='model', help='trained or pre-trained model directory')
    parser.add_argument('--ext', type=str, default='.png', help='file extension')
    parser.add_argument('--test_name', type=str, default='Set11', help='name of test set')

    args = parser.parse_args()
    main(args)