import math

import torch
from torch.utils.cpp_extension import load

dilation = load(name='dilation', sources=[
    '../colorhandpose3d/extensions/dilation2d_cuda.cpp',
    '../colorhandpose3d/extensions/dilation.cu'
])
dilation2d = load(name='dilation2d', sources=[
    '../colorhandpose3d/extensions/dilation2d.cpp'
])
torch.set_printoptions(threshold=5000)

x = torch.zeros(3, 32, 32).cuda()
padding = [4, 4]
stride = [1, 1]
rates = [1, 1]
kernel = torch.ones(8, 8) / float(8 * 8)
kernel = kernel.cuda()
x[0, 15, 15] = 1.
x[1, 0, 0] = 1.
x[2, 31, 31] = 1.

# Calculate output height and width
output_height = math.floor((x.shape[1] + 2 * padding[0] - kernel.shape[0]) / stride[0]) + 1
output_width = math.floor((x.shape[2] + 2 * padding[1] - kernel.shape[1]) / stride[1]) + 1

block_size = (output_height * output_width + 1024 - 1) / 1024

print('block is {}, {}'.format(block_size, x.shape[0]))

# C++ implementation
output = dilation.dilation(x, kernel, stride[0], stride[1], rates[0],
                               rates[1], padding[0], padding[1],
                               output_height, output_width)

print(output[1])

output2 = dilation2d.dilation2d(x.unsqueeze(1).cpu(), kernel.unsqueeze(0).cpu(), stride[0], stride[1], rates[0],
                               rates[1], padding[0], padding[1],
                               output_height, output_width)

print(output2[1])
print(output.shape, output2.shape)
print(torch.eq(output.cpu(), output2).all())
