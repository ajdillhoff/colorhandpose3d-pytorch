from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(name='dilation2d',
      ext_modules=[CUDAExtension('dilation', [
          'dilation2d_cuda.cpp',
          'dilation.cu'
      ])],
      cmdclass={'build_ext': BuildExtension})
