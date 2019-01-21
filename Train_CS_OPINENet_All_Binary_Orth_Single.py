
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torch.optim as optim
import scipy.io as sio
import numpy as np
import os
import sys
from torch.utils.data import Dataset, DataLoader


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

start_epoch = int(sys.argv[1])
end_epoch = int(sys.argv[2])
learning_rate = float(sys.argv[3])
layer_num = int(sys.argv[4])
group_num = int(sys.argv[5])
phi_weight = float(sys.argv[6])
cs_ratio = int(sys.argv[7])
gpu_list = sys.argv[8]


os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list


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


print_flag = 1

if print_flag:
    num_count = 0
    for para in model.parameters():
        num_count += 1
        print('Layer %d' % num_count)
        print(para.size())

Training_data = sio.loadmat('Training_Data_0_20_Img91.mat')
Training_labels = Training_data['labels']

nrtrain = 88912
batch_size = 128


class RandomDataset(Dataset):
    def __init__(self, data, length):
        self.data = data
        self.len = length

    def __getitem__(self, index):
        return torch.Tensor(self.data[index, :]).float()

    def __len__(self):
        return self.len


rand_loader = DataLoader(dataset=RandomDataset(Training_labels, nrtrain), batch_size=batch_size, num_workers=4, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model_dir = "OPINE_Net_%d_group_%d_%d_Binary_Norm_%.2f_Single_%.4f" % (layer_num, group_num, CS_ratio, phi_weight, learning_rate)

output_file_name = "Log_output_%s.txt" % (model_dir)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)


if start_epoch > 0:
    pre_model_dir = model_dir
    model.load_state_dict(torch.load('./%s/net_params_%d.pkl' % (pre_model_dir, start_epoch)))


Eye_I = torch.eye(n_input).to(device)

# Training loop
for epoch_i in range(start_epoch+1, end_epoch+1):
    idx_all = np.random.permutation(nrtrain)

    for data in rand_loader:
        # randidx = np.random.randint(nrtrain, size=batch_size, dtype=np.int32)
        batch_ys = data


        batch_ys = batch_ys.to(device)

        [y_pred, y_layers_sym, Phi] = model(batch_ys)

        # Compute and print loss
        loss = torch.mean(torch.pow(y_pred-batch_ys, 2))

        loss_sym = torch.mean(torch.pow(y_layers_sym[0], 2))
        for k in range(layer_num-1):
            loss_sym += torch.mean(torch.pow(y_layers_sym[k+1], 2))

        loss_phi = torch.mean(torch.pow(torch.mm(Phi, torch.transpose(Phi, 0, 1))-Eye_I, 2))

        # PhiTPhix = torch.mm(torch.mm(batch_ys, torch.transpose(Phi, 0, 1)), Phi)
        # loss_I = torch.mean(torch.pow(PhiTPhix - batch_ys, 2))


        loss_w1 = torch.Tensor([0.01]).to(device)
        loss_w2 = torch.Tensor([phi_weight]).to(device)
        # loss_w3 = torch.Tensor([0.01]).to(device)

        loss_all = loss + torch.mul(loss_w1, loss_sym) + torch.mul(loss_w2, loss_phi)
        # loss_all = loss + torch.mul(loss_w1, loss_sym) + torch.mul(loss_w2, loss_phi) + torch.mul(loss_w3, loss_I)

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()

        output_data = "[%02d/%02d] Loss: %.4f, Loss_sym: %.4f, Loss_phi: %.8f\n" % (epoch_i, end_epoch, loss.item(), loss_sym.item(), loss_phi.item())
        # output_data = "[%02d/%02d] Loss: %.4f, Loss_sym: %.4f, Loss_phi: %.8f, Loss_I: %.4f\n" % (epoch_i, end_epoch, loss.item(), loss_sym.item(), loss_phi.item(), loss_I.item())
        print(output_data)

    output_file = open(output_file_name, 'a')
    output_file.write(output_data)
    output_file.close()

    torch.save(model.state_dict(), "./%s/net_params_%d.pkl" % (model_dir, epoch_i))  # save only the parameters
