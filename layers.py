import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math

def squash(inputs, axis=-1):
    norm = torch.norm(inputs, p=2, dim=axis, keepdim=True)
    scale = norm**2 / (1 + norm**2) / (norm + 1e-8)
    return scale * inputs

class DenseCapsule(nn.Module):
    def __init__(self, in_num_caps, in_dim_caps, out_num_caps, out_dim_caps, args):
        super(DenseCapsule, self).__init__()
        self.in_num_caps = in_num_caps
        self.in_dim_caps = in_dim_caps
        self.out_num_caps = out_num_caps
        self.out_dim_caps = out_dim_caps
        self.args = args
        self.weight = nn.Parameter(0.1 * torch.randn(out_num_caps, in_num_caps, out_dim_caps, in_dim_caps))

    def forward(self, x):
        x_hat = torch.squeeze(torch.matmul(self.weight, x[:, None, :, :, None]), dim=-1)
        x_hat_detached = x_hat.detach()

        b = Variable(torch.zeros(x.size(0), self.out_num_caps, self.in_num_caps)).cuda()

        keep_features = list()
        k = [self.args.args_k_0, self.args.args_k_1, self.args.args_k_2]

        for i in range(3):
            c = F.softmax(b, dim=1)
            c = c[:, :, :, None]

            if i == 2:
                outputs = squash(torch.sum(c * x_hat, dim=-2, keepdim=True))
                keep_features.append(outputs * k[i])
            else:
                outputs = squash(torch.sum(c * x_hat_detached, dim=-2, keepdim=True))
                b = b + torch.sum(outputs * x_hat_detached, dim=-1)
                keep_features.append(outputs * k[i])

        outputs = torch.sum(torch.squeeze(torch.stack(keep_features), dim=-2), dim=0)
        outputs = outputs.view(outputs.size(0), outputs.size(1), 4, 4)

        return outputs

class PrimaryCapsule(nn.Module):
    def __init__(self, in_channels, out_channels, dim_caps, stride=1):
        super(PrimaryCapsule, self).__init__()
        self.dim_caps = dim_caps
        self.conv2d = conv_basic_dy(in_channels, out_channels, stride)

    def forward(self, x):
        outputs = self.conv2d(x)
        outputs = outputs.view(x.size(0), -1, self.dim_caps)
        return squash(outputs)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 3.

class SEModule_small(nn.Module):
    def __init__(self, channel):
        super(SEModule_small, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel, bias=False),
            Hsigmoid()
        )

    def forward(self, x):
        y = self.fc(x)
        return x * y

class conv_basic_dy(nn.Module):
    def __init__(self, inplanes, planes, stride):
        super(conv_basic_dy, self).__init__()
        
        self.conv = conv3x3(inplanes, planes, stride)
        self.dim = int(math.sqrt(inplanes*4))
        squeeze = max(inplanes*4, self.dim ** 2) // 16
        if squeeze < 4:
            squeeze = 4
        
        self.q = nn.Conv2d(inplanes, self.dim, 1, stride, 0, bias=False)

        self.p = nn.Conv2d(self.dim, planes, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(self.dim)
        self.bn2 = nn.BatchNorm1d(self.dim)

        self.avg_pool = nn.AdaptiveAvgPool2d(2)  

        self.fc = nn.Sequential(
            nn.Linear(inplanes*4, squeeze, bias=False),
            SEModule_small(squeeze),
        ) 
        self.fc_phi = nn.Linear(squeeze, self.dim**2, bias=False)
        self.fc_scale = nn.Linear(squeeze, planes, bias=False)
        self.hs = Hsigmoid()     
        
    def forward(self, x):
        r = self.conv(x)
        b, c, _, _= x.size()                               
        y = self.fc(self.avg_pool(x).view(b, c*4))                                    
        phi = self.fc_phi(y).view(b, self.dim, self.dim)  
        scale = self.hs(self.fc_scale(y)).view(b,-1,1,1)  
                                                           
        r = scale.expand_as(r)*r                           
                                                        
        out = self.bn1(self.q(x))                        
        _, _, h, w = out.size()                             

        out = out.view(b,self.dim,-1)                  
        out = self.bn2(torch.matmul(phi, out)) + out     
        out = out.view(b,-1,h,w)                        
        out = self.p(out) + r
         
        return out                                   


class NormLayer(nn.Module):
    def __init__(self, dim=None):
        super(NormLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        x = torch.norm(x, dim=self.dim)
        return x


class ConcatLayer(nn.Module):
    def forward(self, x1, x2, x3):

        x = torch.cat([x1, x2, x3], dim=1)
        x = x.permute(0, 2, 1)
        x = x[:, :, :, None]
        return x
