
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

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

epoch_num = int(sys.argv[1])
layer_num = int(sys.argv[2])
group_num = int(sys.argv[3])
phi_weight = float(sys.argv[4])
cs_ratio = int(sys.argv[5])
test_dir = sys.argv[6]
learning_rate = float(sys.argv[7])


CS_ratio = cs_ratio
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if CS_ratio == 4:
    n_input = 43
elif CS_ratio == 1:
    n_input = 10
elif CS_ratio == 10:
    n_input = 109
elif CS_ratio == 25:
    n_input = 272
elif CS_ratio == 30:
    n_input = 327
elif CS_ratio == 40:
    n_input = 436
elif CS_ratio == 50:
    n_input = 545


n_output = 1089

class MySign(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        output = input.new(input.size())
        output[input >= 0] = 1
        output[input < 0] = -1
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        grad_input = grad_output.clone()
        return grad_input


MyBinarize = MySign.apply


class BasicBlock(torch.nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear module
        """
        super(BasicBlock, self).__init__()
        # self.linear = torch.nn.Linear(1, 1)  # One in and one out
        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))

        self.conv1_w = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 1, 3, 3)))
        self.conv2_w = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv3_w = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv4_w = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv5_w = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv55_w = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv555_w = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv6_w = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 32, 3, 3)))

    def forward(self, x, PhiTPhi, PhiTb):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """

        x = x - self.lambda_step * torch.mm(x, PhiTPhi)
        x = x + self.lambda_step * PhiTb
        xx = x.view(-1, 1, 33, 33)

        # x = self.conv1(xx)
        x3 = F.conv2d(xx, self.conv1_w, padding=1)

        x = F.conv2d(x3, self.conv2_w, padding=1)
        x = F.relu(x)
        # x = self.conv3(x)
        x4 = F.conv2d(x, self.conv3_w, padding=1)
        # x = F.softshrink(x, self.soft_thr)
        # x = F.relu(x)
        x = torch.mul(torch.sign(x4), F.relu(torch.abs(x4) - self.soft_thr))
        # x = torch.sign(x) * (F.relu(torch.abs(x) - self.soft_thr))
        # tf.multiply(tf.sign(x4_ista), tf.nn.relu(tf.abs(x4_ista) - soft_thr))
        # x = F.relu(self.conv4(x))
        x = F.relu(F.conv2d(x, self.conv4_w, padding=1))

        x = F.conv2d(x, self.conv5_w, padding=1)
        x = F.relu(x)
        x = F.conv2d(x, self.conv55_w, padding=1)
        x = F.relu(x)
        x = F.conv2d(x, self.conv555_w, padding=1)
        x = F.relu(x)
        x = F.conv2d(x, self.conv6_w, padding=1)
        x = x + xx

        x5 = F.relu(F.conv2d(x4, self.conv4_w, padding=1))
        x5 = F.conv2d(x5, self.conv5_w, padding=1)
        y_pred = x.view(-1, 1089)
        x6 = x5 - x3
        return [y_pred, x6]


class ISTANet(torch.nn.Module):
    def __init__(self, block, LayerNo):
        """
        In the constructor we instantiate two nn.Linear module
        """
        super(ISTANet, self).__init__()
        # self.block = block
        self.Phi = nn.Parameter(init.xavier_normal_(torch.Tensor(n_input, 1089)))
        self.Phi_scale = nn.Parameter(torch.Tensor([0.01]))
        onelayer = []
        self.LayerNo = LayerNo

        fc = BasicBlock()

        for i in range(LayerNo):
            onelayer.append(fc)

        self.fcs = nn.ModuleList(onelayer)

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        Phi_ = MyBinarize(self.Phi)
        Phi = self.Phi_scale * Phi_
        # Phi = torch.sign(self.Phi)
        PhiTPhi = torch.mm(torch.transpose(Phi, 0, 1), Phi)
        PhiTb = torch.mm(x, PhiTPhi)
        x = PhiTb
        layers_sym = []
        for i in range(self.LayerNo):
            # layers.append(self.block(layers[-1], PhiTPhi, PhiTb))
            # x = self.onelayer[i](x)
            [x, x_sym] = self.fcs[i](x, PhiTPhi, PhiTb)
            layers_sym.append(x_sym)
        y_pred = x
        return [y_pred, layers_sym, Phi]



# our model
model = ISTANet(BasicBlock(), layer_num)
model = nn.DataParallel(model)
model = model.to(device)


nrtrain = 88912
batch_size = 64

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
# criterion = torch.nn.MSELoss(size_average=True)
# criterion.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop


model_dir = "OPINE_Net_%d_group_%d_%d_Binary_Norm_%.2f_Single_%.4f" % (layer_num, group_num, CS_ratio, phi_weight, learning_rate)
output_file_name = "Log_output_%s.txt" % (model_dir)


model.load_state_dict(torch.load('./%s/net_params_%d.pkl' % (model_dir, epoch_num)))
# model.load_state_dict(torch.load('net_params_%d.pkl' % epoch_num, map_location=lambda storage, loc: storage))


def rgb2ycbcr(rgb):
    m = np.array([[ 65.481, 128.553, 24.966],
                  [-37.797, -74.203, 112],
                  [ 112, -93.786, -18.214]])
    shape = rgb.shape
    if len(shape) == 3:
        rgb = rgb.reshape((shape[0] * shape[1], 3))
    ycbcr = np.dot(rgb, m.transpose() / 255.)
    ycbcr[:,0] += 16.
    ycbcr[:,1:] += 128.
    return ycbcr.reshape(shape)

# ITU-R BT.601
# https://en.wikipedia.org/wiki/YCbCr
# YUV -> RGB
def ycbcr2rgb(ycbcr):
    m = np.array([[ 65.481, 128.553, 24.966],
                  [-37.797, -74.203, 112],
                  [ 112, -93.786, -18.214]])
    shape = ycbcr.shape
    if len(shape) == 3:
        ycbcr = ycbcr.reshape((shape[0] * shape[1], 3))
    rgb = copy.deepcopy(ycbcr)
    rgb[:,0] -= 16.
    rgb[:,1:] -= 128.
    rgb = np.dot(rgb, np.linalg.inv(m.transpose()) * 255.)
    return rgb.clip(0, 255).reshape(shape)

def imread_CS_py(Iorg):
    block_size = 33
    [row, col] = Iorg.shape
    row_pad = block_size-np.mod(row,block_size)
    col_pad = block_size-np.mod(col,block_size)
    Ipad = np.concatenate((Iorg, np.zeros([row, col_pad])), axis=1)
    Ipad = np.concatenate((Ipad, np.zeros([row_pad, col+col_pad])), axis=0)
    [row_new, col_new] = Ipad.shape

    return [Iorg, row, col, Ipad, row_new, col_new]


def img2col_py(Ipad, block_size):
    [row, col] = Ipad.shape
    row_block = row/block_size
    col_block = col/block_size
    block_num = int(row_block*col_block)
    img_col = np.zeros([block_size**2, block_num])
    count = 0
    for x in range(0, row-block_size+1, block_size):
        for y in range(0, col-block_size+1, block_size):
            img_col[:, count] = Ipad[x:x+block_size, y:y+block_size].reshape([-1])
            # img_col[:, count] = Ipad[x:x+block_size, y:y+block_size].transpose().reshape([-1])
            count = count + 1
    return img_col


def col2im_CS_py(X_col, row, col, row_new, col_new):
    block_size = 33
    X0_rec = np.zeros([row_new, col_new])
    count = 0
    for x in range(0, row_new-block_size+1, block_size):
        for y in range(0, col_new-block_size+1, block_size):
            X0_rec[x:x+block_size, y:y+block_size] = X_col[:, count].reshape([block_size, block_size])
            # X0_rec[x:x+block_size, y:y+block_size] = X_col[:, count].reshape([block_size, block_size]).transpose()
            count = count + 1
    X_rec = X0_rec[:row, :col]
    return X_rec


def psnr(img1, img2):
    img1.astype(np.float32)
    img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


# Test_Img = './Test_Image512'
Test_Img = './' + test_dir

filepaths = glob.glob(Test_Img + '/*.tif')

ImgNum = len(filepaths)
PSNR_All = np.zeros([1, ImgNum], dtype=np.float32)
SSIM_All = np.zeros([1, ImgNum], dtype=np.float32)


for img_no in range(ImgNum):

    imgName = filepaths[img_no]


    # Iorg_rgb = np.array(Image.open(imgName), dtype='float32')
    #
    # Iorg_yuv = rgb2ycbcr(Iorg_rgb)

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
    # print("Run time for this image is %.4f, PSNR is %.2f, loss sym is %.4f" % ((end - start), rec_PSNR, loss_sym))

    # img_rec_name = "%s_rec_%s_%d_PSNR_%.2f.tif" % (imgName, model_dir, epoch_num, rec_PSNR)

    Img_rec_yuv[:,:,0] = X_rec*255

    im_rec_rgb = cv2.cvtColor(Img_rec_yuv, cv2.COLOR_YCrCb2BGR)
    im_rec_rgb = np.clip(im_rec_rgb, 0, 255).astype(np.uint8)
    # cv2.imwrite("%s_OPINE_Net_ratio_%d_epoch_%d_PSNR_%.2f.png" % (imgName, CS_ratio, epoch_num, rec_PSNR), im_rec_rgb)
    cv2.imwrite("%s_OPINENet_Orth_Single_ratio_%d_epoch_%d_PSNR_%.2f_SSIM_%.4f.png" % (imgName, CS_ratio, epoch_num, rec_PSNR, rec_SSIM), im_rec_rgb)
    del y_pred

    PSNR_All[0, img_no] = rec_PSNR
    SSIM_All[0, img_no] = rec_SSIM
    print(img_no)

print(test_dir)
print('\n')
output_data = "CS ratio is %d, Avg PSNR/SSIM is %.2f/%.4f, cpkt NO. is %d \n" % (CS_ratio, np.mean(PSNR_All), np.mean(SSIM_All), epoch_num)
print(output_data)

output_file_name = "PSNR_Results_%s_RGB.txt" % (model_dir)
output_file = open(output_file_name, 'a')
output_file.write(output_data)
output_file.close()

print("Reconstruction READY")