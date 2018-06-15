import torch.nn as nn
import torch.nn.functional as F

u"""
  Deep Residual Network Implementation
"""

class PlainBlock(nn.Module):
  u"""
    Plain Architecture for ResNet

    this makes
      conv2d in_d, out_d / 4, with kernel = (3, 3)
      conv2d out_d, oud_d / 4, with kernel = (3, 3)

    ex) normal plain block
      conv2d 64 -> 64, with kernel = (3, 3)
      conv2d 64 -> 64, with kernel = (3, 3)

    Args:
      in_d: input dimention. ex) number of input color channels
      out_d: output dimention. ex) number of output color channels
      stride
  """

  def __init__(self, in_d, out_d, stride=1):
    super(PlainBlock, self).__init__()

    self.conv1 = nn.Conv2d(in_d, out_d, 3, stride=stride, padding=1, bias=False)
    self.conv2 = nn.Conv2d(out_d, out_d, 3, stride=1, padding=1, bias=False)

    if in_d is out_d:
      self.shortcut = nn.Sequential()
    else:
      self.shortcut = nn.Sequential(
        nn.Conv2d(in_d, out_d, 1, stride, bias=False)
      )

  def forward(self, x):
    out = self.conv1(x)
    out = F.relu(out)
    out = self.conv2(out)
    out += self.shortcut(x)
    out = F.relu(out)
    return out

class BottleneckBlock(nn.Module):
  u"""
    Bottoleneck Architecture for ResNet

                    INPUT
      Conv
      ReLU
      Conv                      Shortcut
      ReLU
      Conv
                    ReLU
                   OUTPUT


    this makes
      conv2d in_d, oud_d / 4
      conv2d out_d / 4, out_d / 4
      conv2d out_d / 4, out_d

    ex) normal bottleneck block
      conv2d 256 -> 64, with kernel = (1, 1)
      conv2d 64 -> 64, with kernel = (3, 3)
      conv2d 64 -> 256, with kernel = (1, 1)

    Args:
      in_d: input dimention. ex) number of input color channels
      out_d: output dimention. ex) number of output color channels
      stride
  """

  def __init__(self, in_d, out_d, stride=1):
    super(BottleneckBlock, self).__init__()
    mid_d = int(out_d / 4)
    if mid_d < 1:
      mid_d = 1

    self.conv1 = nn.Conv2d(in_d, mid_d, 1)
    self.conv2 = nn.Conv2d(mid_d, mid_d, 3, stride, padding=1)
    self.conv3 = nn.Conv2d(mid_d, out_d, 1)

    if in_d == out_d:
      self.shortcut = nn.Sequential()
    else:
      self.shortcut = nn.Sequential(
        nn.Conv2d(in_d, out_d, 1, stride, bias=False)
      )

  def forward(self, x):
    out = self.conv1(x)
    out = F.relu(out)
    out = self.conv2(out)
    out = F.relu(out)
    out = self.conv3(out)
    out += self.shortcut(x)
    out = F.relu(out)
    return out

class BottleneckV2Block(nn.Module):
  u"""
    Bottleneck architecture version 2 - improved

                    INPUT
      BatchNorm
      ReLU
      Conv
      BatchNorm
      ReLU
      Conv                  Shortcut
      BatchNorm
      ReLU
      Dropout
      Conv
                    Output

    this makes
      conv2d in_d, oud_d / 4
      conv2d out_d / 4, out_d / 4
      conv2d out_d / 4, out_d

    ex) normal bottleneck block
      conv2d 256 -> 64, with kernel = (1, 1)
      conv2d 64 -> 64, with kernel = (3, 3)
      conv2d 64 -> 256, with kernel = (1, 1)

    Args:
      in_d: input dimention. ex) number of input color channels
      out_d: output dimention. ex) number of output color channels
      stride

  """
  def __init__(self, in_d, out_d, stride=1):
    super(BottleneckBlock, self).__init__()
    mid_d = int(out_d / 4)
    if mid_d < 1:
      mid_d = 1

    self.bn1 = nn.BatchNorm2d(in_d)
    self.conv1 = nn.Conv2d(in_d, mid_d, 1, bias=False)
    self.bn2 = nn.BatchNorm2d(mid_d)
    self.conv2 = nn.Conv2d(mid_d, mid_d, 3, stride, padding=1, bias=False)
    self.bn3 = nn.BatchNorm2d(mid_d)
    self.do = nn.Dropout2d()
    self.conv3 = nn.Conv2d(mid_d, out_d, 1, bias=False)

    if in_d is out_d:
      self.shortcut = nn.Sequential()
    else:
      self.shortcut = nn.Sequential(
        nn.Conv2d(in_d, out_d, 1, stride, bias=False)
      )

  def forward(self, x):
    out = self.bn1(x)
    out = F.relu(out)
    out = self.conv1(out)
    out = self.bn2(out)
    out = F.relu(out)
    out = self.conv2(out)
    out = self.bn3(out)
    out = F.relu(out)
    out = self.do(out)
    out = self.conv3(out)
    out += self.shortcut(x)
    return out

class ResNet(nn.Module):
  u"""
    Residual Network

    Args:
      Block: BottleneckBlock | PlainBlock
      blocks: resnet blocks. ex) blocks = [32, 64, 64, 32]
  """

  def __init__(self, Block, blocks):
    super(ResNet, self).__init__()

    _layers = []
    for index in range(1, len(blocks)):
      _layers.append(Block(blocks[index - 1], blocks[index]))
    self.layer = nn.Sequential(*_layers)

  def forward(self, x):
    return self.layer(x)
