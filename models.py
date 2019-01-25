import torch
import argparse
import torch.nn as nn
from torch.nn import init
import common
import torch.nn.functional as F



class BasicBlock(torch.nn.Module):
    def __init__(self):
        super(BasicBlock, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.soft_thr = common.Soft_Thr(initial_soft_thr=0.01)

        conv = common.default_conv

        feat_num = 32

        m_head = [conv(1, feat_num, 3, bias=False)] # conv1

        m_forward = [conv(feat_num, feat_num, 3, bias=False), nn.ReLU(), conv(feat_num, feat_num, 3, bias=False)]
        m_backward = [conv(feat_num, feat_num, 3, bias=False), nn.ReLU(), conv(feat_num, feat_num, 3, bias=False)]

        m_trunk = [common.BasicBlock(conv, feat_num, feat_num, 3) for _ in range(2)]
        m_trunk.append(conv(feat_num, 1, 3, bias=False))

        self.head = nn.Sequential(*m_head)
        self.forward_transform = nn.Sequential(*m_forward)
        self.backward_transform = nn.Sequential(*m_backward)
        self.trunk = nn.Sequential(*m_trunk)


    def forward(self, x, PhiTPhi, PhiTb):
        x = x - self.lambda_step * torch.mm(x, PhiTPhi)
        x = x + self.lambda_step * PhiTb
        x = x.view(-1, 1, 33, 33)

        x_head = self.head(x)
        x_forward = self.forward_transform(x_head)
        xx = self.soft_thr(x_forward)
        xx = self.backward_transform(xx)
        xx = F.relu(xx)
        res = self.trunk(xx)
        x = res + x

        x_pred = x.view(-1, 1089)

        x_forward_backward = self.backward_transform(x_forward)
        x_diff = x_head - x_forward_backward

        return [x_pred, x_diff]


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
    def __init__(self, block, LayerNo, n_input, share_flag):
        super(ISTANet, self).__init__()
        # self.block = block
        self.Phi = nn.Parameter(init.xavier_normal_(torch.Tensor(n_input, 1089)))
        # Phi2 = nn.Parameter(init.xavier_normal_(torch.Tensor(n_input, 1089)))
        # self.Phis = nn.ModuleList([Phi1,Phi2])
        self.Phi_scale = nn.Parameter(torch.Tensor([0.01]))
        self.LayerNo = LayerNo
        basicblock = BasicBlock()

        if share_flag == 1:
            m_layers = [basicblock for _ in range(LayerNo)]
        else:
            m_layers = [BasicBlock() for _ in range(LayerNo)]

        self.basicblocks = nn.ModuleList(m_layers)

    def forward(self, x):
        Phi_ = MyBinarize(self.Phi)
        Phi = self.Phi_scale * Phi_
        # Phi = torch.sign(self.Phi)
        PhiTPhi = torch.mm(torch.transpose(Phi, 0, 1), Phi)
        PhiTb = torch.mm(x, PhiTPhi)
        x = PhiTb
        layers_sym = []
        for basicblock in self.basicblocks:
            [x, x_sym] = basicblock(x, PhiTPhi, PhiTb)
            layers_sym.append(x_sym)
        y_pred = x
        return [y_pred, layers_sym, Phi]


def get_ISTANet(layer_num,n_input, share_flag):
    return ISTANet(BasicBlock(), layer_num, n_input, share_flag)



class ISTANet_Multiple(torch.nn.Module):
    def __init__(self, block, LayerNo, n_input, share_flag):
        super(ISTANet_Multiple, self).__init__()
        # self.block = block

        # Phi1 = nn.Parameter(init.xavier_normal_(torch.Tensor(n_input, 1089)))
        # Phi2 = nn.Parameter(init.xavier_normal_(torch.Tensor(n_input, 1089)))
        self.Phis = nn.ParameterList([nn.Parameter(init.xavier_normal_(torch.Tensor(k, 1089))) for k in n_input])

        # Phi_scale1 = nn.Parameter(torch.Tensor([0.01]))
        # Phi_scale2 = nn.Parameter(torch.Tensor([0.01]))
        self.Phi_scales = nn.ParameterList([nn.Parameter(torch.Tensor([0.01])) for _ in n_input])

        self.LayerNo = LayerNo

        basicblock = BasicBlock()

        if share_flag == 1:
            m_layers = [basicblock for _ in range(LayerNo)]
        else:
            m_layers = [BasicBlock() for _ in range(LayerNo)]

        # m_layers = [basicblock for _ in range(LayerNo)]
        self.basicblocks = nn.ModuleList(m_layers)

    def forward(self, x, phi_index):
        Phi_ = MyBinarize(self.Phis[phi_index])
        Phi = self.Phi_scales[phi_index] * Phi_

        # Phi = torch.sign(self.Phi)
        PhiTPhi = torch.mm(torch.transpose(Phi, 0, 1), Phi)
        PhiTb = torch.mm(x, PhiTPhi)
        x = PhiTb
        layers_sym = []
        for basicblock in self.basicblocks:
            [x, x_sym] = basicblock(x, PhiTPhi, PhiTb)
            layers_sym.append(x_sym)
        y_pred = x
        return [y_pred, layers_sym, Phi]


def get_ISTANet_Multiple(layer_num,n_input, share_flag):
    return ISTANet_Multiple(BasicBlock(), layer_num, n_input, share_flag)