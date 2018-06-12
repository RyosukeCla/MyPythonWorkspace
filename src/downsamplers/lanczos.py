import torch.nn as nn
import numpy as np
import torch

u"""
  Lanczos downsampler module
"""

class Lanczos(nn.Module):
  u"""
   Lanczos downsampler

   Args:
    in_d: input dimention. ex) number of input color channels
    factor: downsample factor. ex) factor = 2 means that this produces a 0.5 X image
    kernel_width
    support: Lanczos N. ex) support = 2 means Lanczos 2
  """
  def __init__(self, in_d, factor, kernel_width, support=1):
    super(Lanczos, self).__init__()

    self.kernel = self.get_kernel(factor, kernel_width, support)

    self.sampler = nn.Conv2d(in_d, in_d, kernel_size=self.kernel.shape, stride=factor, padding=0)
    self.sampler.weight.data[:] = 0
    self.sampler.bias.data[:] = 0

    kernel_torch = torch.from_numpy(self.kernel)
    for i in range(in_d):
      self.sampler.weight.data[i, i] = kernel_torch

    if  self.kernel.shape[0] % 2 == 1:
      pad = int((self.kernel.shape[0] - 1) / 2.)
    else:
      pad = int((self.kernel.shape[0] - factor) / 2.)

    self.padding = nn.ReplicationPad2d(pad)

  def get_kernel(self, factor, kernel_width, support=1):
    kernel = np.zeros([kernel_width, kernel_width])

    center = (kernel_width) / 2.0

    pi2 = np.pi * np.pi

    for i in range(kernel.shape[0]):
      for j in range(kernel.shape[1]):
        di = abs(i - center) / factor
        dj = abs(j - center) / factor

        val = 1
        if di != 0:
          val *= support * np.sin(np.pi * di) * np.sin(np.pi * di / support) / (pi2 * di * di)
        if dj != 0:
          val *= support * np.sin(np.pi * dj) * np.sin(np.pi * dj / support) / (pi2 * dj * dj)

        kernel[i][j] = val

    kernel /= kernel.sum()

    return kernel

  def forward(self, x):
    out = self.padding(x)
    out = self.sampler(out)
    return out
