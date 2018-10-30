import math
import timeit

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

batch_size = 8
x = torch.zeros(batch_size, 256, 256).cuda()
padding = [10, 10]
stride = [1, 1]
rates = [1, 1]
kernel = torch.ones(21, 21) / float(21 * 21)
kernel = kernel.cuda()
x[0, 0, 0] = 1.

# Calculate output height and width
output_height = math.floor((x.shape[1] + 2 * padding[0] - kernel.shape[0]) / stride[0]) + 1
output_width = math.floor((x.shape[2] + 2 * padding[1] - kernel.shape[1]) / stride[1]) + 1

print(output_height, output_width)

# C++ implementation
output = dilation.dilation(x, kernel, stride[0], stride[1], rates[0],
                               rates[1], padding[0], padding[1],
                               output_height, output_width)

print(output[0])

start = timeit.default_timer()
output2 = dilation2d.dilation2d(x.unsqueeze(1).cpu(), kernel.unsqueeze(0).cpu(), stride[0], stride[1], rates[0],
                               rates[1], padding[0], padding[1],
                               output_height, output_width)
print('CPU time: {}'.format(timeit.default_timer() - start))

print(output2[0])
print(output.shape, output2.shape)
for i in range(batch_size):
    print('batch_idx: {} equal {}'.format(i, torch.eq(output[i].cpu(), output2[i]).all()))
