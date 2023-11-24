import sys

import sys
import os
import torch
import random
import numpy as np

from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from timm.models.layers import trunc_normal_

from vision_transformer import *

from torch.nn import Linear



def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


class SupConVitGAClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super(SupConVitGAClassifier, self).__init__()
        self._fc1 = nn.Sequential(nn.Linear(2048, 768), nn.ReLU())
        # self._fc1 = nn.Sequential(nn.Linear(1280, 768), nn.ReLU())
        self.embed_dim = 768
        self.feat_dim = 128
        self.num_classes = num_classes
        self.encoder_q = VisionTransformer_(embed_dim=self.embed_dim, depth=3, num_heads=12)

        self.head_q = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim), nn.GELU(), nn.Linear(self.embed_dim, self.feat_dim)
        )

    def forward(self, x):
        if self.training:
            x = self._fc1(x)  # [B, n, 768]
            q = self.encoder_q(x) # [B, 768]
            qq = nn.functional.normalize(q, dim=1)
            qq = self.head_q(qq)
            qq = nn.functional.normalize(qq, dim=1)
            return qq, q[0].clone().detach()
        else:
            x = self._fc1(x)  # [B, n, 768]
            q = self.encoder_q(x)

            # qq = nn.functional.normalize(q, dim=1)
            # qq = self.head_q(qq)
            # qq = nn.functional.normalize(qq, dim=1)

            return q
