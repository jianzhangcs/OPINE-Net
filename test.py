
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy.io as sio
import numpy as np
import os
import sys
import glob
from time import time
from PIL import Image

import math
from torch.nn import init
import copy
import cv2
from skimage.measure import compare_ssim as ssim
import models
import option
from utils import *


def test():
    args = option.get_args_test()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_list

    epoch_num = args.test_epoch
    layer_num = args.layer_num
    group_num = args.group_num
    phi_weight = args.phi_weight
    CS_ratio = args.cs_ratio
    test_dir = args.test_dir
    learning_rate = args.learning_rate
    phi_index = args.phi_index
    share_flag = args.share_flag

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # our model
    cs_ratio_dic = {4: 43, 1: 10, 10: 109, 25: 272, 30: 327, 40: 436, 50: 545}

    n_input = cs_ratio_dic[CS_ratio]
    n_output = 1089
    # our model
    model = models.get_ISTANet(9, n_input, share_flag)

    model = nn.DataParallel(model)
    model = model.to(device)

    nrtrain = 88912
    batch_size = 64

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model_dir = "OPINE_Net_share_%d_layer_%d_group_%d_%d_Binary_Norm_%.2f_Single_%.6f" % (share_flag, layer_num, group_num, CS_ratio, phi_weight, learning_rate)
    output_file_name = "Log_output_%s.txt" % (model_dir)


    model.load_state_dict(torch.load('./%s/net_params_%d.pkl' % (model_dir, epoch_num)))
    # model.load_state_dict(torch.load('net_params_%d.pkl' % epoch_num, map_location=lambda storage, loc: storage))

    Test_Img = './' + test_dir

    filepaths = glob.glob(Test_Img + '/*.tif')

    ImgNum = len(filepaths)
    PSNR_All = np.zeros([1, ImgNum], dtype=np.float32)
    SSIM_All = np.zeros([1, ImgNum], dtype=np.float32)


    with torch.no_grad():
        for img_no in range(ImgNum):

            imgName = filepaths[img_no]
            Img = cv2.imread(imgName, 1)

            Img_yuv = cv2.cvtColor(Img, cv2.COLOR_BGR2YCrCb)
            Img_rec_yuv = Img_yuv.copy()

            Iorg_y = Img_yuv[:,:,0]

            [Iorg, row, col, Ipad, row_new, col_new] = imread_CS_py(Iorg_y)
            Icol = img2col_py(Ipad, 33).transpose()/255.0

            Img_output = Icol

            start = time()

            batch_ys = torch.from_numpy(Img_output)
            batch_ys = batch_ys.type(torch.FloatTensor)
            batch_ys = batch_ys.to(device)

            [y_pred, y_layers_sym, phi] = model(batch_ys)
            end = time()

            Prediction_value = y_pred.cpu().data.numpy()
            phi_value = phi.cpu().data.numpy()

            loss_sym = torch.mean(torch.pow(y_layers_sym[0], 2))
            for k in range(layer_num - 1):
                loss_sym += torch.mean(torch.pow(y_layers_sym[k + 1], 2))

            loss_sym = loss_sym.cpu().data.numpy()

            X_rec = np.clip(col2im_CS_py(Prediction_value.transpose(), row, col, row_new, col_new), 0, 1)

            rec_PSNR = psnr(X_rec*255, Iorg.astype(np.float64))
            rec_SSIM = ssim(X_rec*255, Iorg.astype(np.float64), data_range=255)

            sio.savemat('OPINE_Net_phi_%d.mat' % CS_ratio, {'Phi': phi_value})
            print("Run time for image %2d is %.4fs, PSNR is %.2f" % (img_no, (end - start), rec_PSNR))

            Img_rec_yuv[:,:,0] = X_rec*255

            im_rec_rgb = cv2.cvtColor(Img_rec_yuv, cv2.COLOR_YCrCb2BGR)
            im_rec_rgb = np.clip(im_rec_rgb, 0, 255).astype(np.uint8)
            # cv2.imwrite("%s_OPINE_Net_ratio_%d_epoch_%d_PSNR_%.2f.png" % (imgName, CS_ratio, epoch_num, rec_PSNR), im_rec_rgb)
            cv2.imwrite("%s_OPINENet_Orth_Single_ratio_%d_epoch_%d_PSNR_%.2f_SSIM_%.4f.png" % (imgName, CS_ratio, epoch_num, rec_PSNR, rec_SSIM), im_rec_rgb)
            del y_pred

            PSNR_All[0, img_no] = rec_PSNR
            SSIM_All[0, img_no] = rec_SSIM
            # print(img_no)

    print(test_dir)
    print('\n')
    output_data = "CS ratio is %d, Avg PSNR/SSIM is %.2f/%.4f, cpkt NO. is %d \n" % (CS_ratio, np.mean(PSNR_All), np.mean(SSIM_All), epoch_num)
    print(output_data)

    output_file_name = "PSNR_Results_%s_RGB.txt" % (model_dir)
    output_file = open(output_file_name, 'a')
    output_file.write(output_data)
    output_file.close()

    print("Reconstruction READY")

if __name__ == '__main__':
    test()