'''
Gelu activation
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class gelu(nn.Module):
    def __init__(self):
        super(gelu, self).__init__()
        self.pi = torch.tensor(np.pi)
        #self.pi=torch.tensor(np.pi).cuda()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(2 / self.pi) * (x + 0.044715 * torch.pow(x, 3))))

class ResidualDenseBlock_8C(nn.Module):
    '''
    Residual Dense Block
    style: 8 convs
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    '''

    def __init__(self, nc):
        super(ResidualDenseBlock_8C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nc, nc, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(nc * 2, nc, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(nc * 3, nc, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4 = nn.Conv2d(nc * 4, nc, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv5 = nn.Conv2d(nc * 5, nc, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv6 = nn.Conv2d(nc * 6, nc, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv7 = nn.Conv2d(nc * 7, nc, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv8 = nn.Conv2d(nc * 8, nc, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv9 = nn.Conv2d(nc * 9, nc, kernel_size=1, stride=1, padding=0, bias=True)
        self.gelu = gelu()

    def forward(self, x):
        x1 = self.gelu(self.conv1(x))
        x2 = self.gelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.gelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.gelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.gelu(self.conv5(torch.cat((x, x1, x2, x3, x4), 1)))
        x6 = self.gelu(self.conv6(torch.cat((x, x1, x2, x3, x4, x5), 1)))
        x7 = self.gelu(self.conv7(torch.cat((x, x1, x2, x3, x4, x5, x6), 1)))
        x8 = self.gelu(self.conv8(torch.cat((x, x1, x2, x3, x4, x5, x6, x7), 1)))
        x9 = self.conv9(torch.cat((x, x1, x2, x3, x4, x5, x6, x7, x8), 1))
        return x9.mul(0.2) + x


class Feedbackblock(nn.Module):
    def __init__(self, num_features):
        super(Feedbackblock, self).__init__()
        self.feature_extract = ResidualDenseBlock_8C(num_features)
        self.should_reset = True
        self.last_hidden = None

    def reset_state(self):
        self.should_reset = True

    def forward(self, x):
        if self.should_reset:
            self.last_hidden = torch.zeros(x.size()).cuda()
            self.last_hidden.copy_(x)
            self.should_reset = False
        x = (x + self.last_hidden) / 2
        x = self.feature_extract(x)
        self.last_hidden = x
        return x


class DFN(nn.Module):
    def __init__(self, iterations=7,blocks=2):
        super(DFN, self).__init__()
        self.iterations = iterations
        self.block_nums=blocks
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
        )

        self.output_layer = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
        )
        self.gelu = gelu()
        self.FB=nn.Sequential(*[Feedbackblock(32) for _ in range(self.block_nums)])
    def all_reset_state(self):
        for block in self.FB:
            block.reset_state()
    def forward(self, input):
        x_list = []
        self.all_reset_state()
        for idx in range(self.iterations):
            x = self.gelu(self.input_layer(input))
            x = self.FB(x)
            x = self.output_layer(x)
            x = x + input
            x_list.append(x)
        return x, x_list