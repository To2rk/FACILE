from torch import nn
import torch
from layers import *
import torch.nn.functional as F

class FACILE(nn.Module):
    def __init__(self, input_size, classes, args):
        super(FACILE, self).__init__()
        self.input_size = input_size
        self.classes = classes
        self.args = args

        self.layer1 = nn.Sequential(
            conv_basic_dy(input_size[0], 8, 1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            PrimaryCapsule(8, 8, 8, stride=5),
            nn.Dropout(args.rate)
        )

        self.layer2 = nn.Sequential(
            conv_basic_dy(input_size[0], 16, 2),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            PrimaryCapsule(16, 16, 8, stride=7),
            nn.Dropout(args.rate)
        )

        self.layer3 = nn.Sequential(
            conv_basic_dy(input_size[0], 32, 3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            PrimaryCapsule(32, 32, 8, stride=9),
            nn.Dropout(args.rate)
        )
        self.con_Layer = ConcatLayer()

        self.dropout = nn.Dropout(args.rate)
        self.digitcaps = DenseCapsule(in_num_caps=83, in_dim_caps=8,
                                      out_num_caps=classes, out_dim_caps=16, args=self.args)
        self.out_feature = NormLayer(dim=-1)

    def forward(self, x):

        x1 = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x)

        x = self.con_Layer(x1, x2, x3)
        x = x.squeeze(-1)
        x = x.permute(0, 2, 1)
        x = self.dropout(x)
        x = self.digitcaps(x)
        x = x.view(x.size(0), x.size(1), 16)

        logits = self.out_feature(x)

        return logits


def mish(input):
    return input * torch.tanh(F.softplus(input))