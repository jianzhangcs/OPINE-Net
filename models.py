import torch
import argparse
import torch.nn as nn
from torch.nn import init
import common



class BasicBlock(torch.nn.Module):
    def __init__(self):
        super(BasicBlock, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.soft_thr = common.Soft_Thr(initial_soft_thr=0.01)

        conv = common.default_conv
        bb = common.BasicBlock

        self.bb2 = common.BasicBlock(conv,32,32,3)
        self.conv3 = conv(32, 32, 3, bias=False)

        self.conv4 = conv(32, 32, 3, bias=False)
        self.conv5 = conv(32, 32, 3, bias=False)

        m_head = [conv(1, 32, 3, bias=False)] # conv1
        m_trunk = [common.BasicBlock(conv,32,32,3) for _ in range(3)] # conv55 555
        m_trunk.append(conv(32, 1, 3, bias=False)) # conv 6

        self.head = nn.Sequential(*m_head)
        self.trunk = nn.Sequential(*m_trunk)
        self.branch = nn.Sequential(self.conv4, nn.ReLU(), self.conv5)

    def forward(self, x, PhiTPhi, PhiTb):
        x = x - self.lambda_step * torch.mm(x, PhiTPhi)
        x = x + self.lambda_step * PhiTb
        #xx = x.view(-1, 1, 33, 33)
        x = x.view(-1, 1, 33, 33)

        x3 = self.head(x)
        res = self.bb2(x3)
        x4 = self.conv3(res)
        res = self.soft_thr(x4)
        res = self.conv4(res)
        res = self.conv5(res)
        res = self.trunk(res)
        x = res + x

        x5 = self.branch(x4)
        y_pred = x.view(-1, 1089)
        x6 = x5 - x3
        return [y_pred, x6]

class Mask_Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        output = input.new(input.size())
        output[input >= 0] = 1
        output[input < 0] = -1
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


MyBinarize = Mask_Func.apply


class ISTANet(torch.nn.Module):
    def __init__(self, block, LayerNo, n_input):
        super(ISTANet, self).__init__()
        # self.block = block
        self.Phi = nn.Parameter(init.xavier_normal_(torch.Tensor(n_input, 1089)))
        self.Phi_scale = nn.Parameter(torch.Tensor([0.01]))
        self.LayerNo = LayerNo
        fc = BasicBlock()
        m_layers = [fc for _ in range(LayerNo)]
        self.fcs = nn.ModuleList(m_layers)

    def forward(self, x):
        Phi_ = MyBinarize(self.Phi)
        Phi = self.Phi_scale * Phi_
        # Phi = torch.sign(self.Phi)
        PhiTPhi = torch.mm(torch.transpose(Phi, 0, 1), Phi)
        PhiTb = torch.mm(x, PhiTPhi)
        x = PhiTb
        layers_sym = []
        for fc in self.fcs:
            [x, x_sym] = fc(x, PhiTPhi, PhiTb)
            layers_sym.append(x_sym)
        y_pred = x
        return [y_pred, layers_sym, Phi]


def get_ISTANet(layer_num,n_input):
    return ISTANet(BasicBlock(), layer_num, n_input)