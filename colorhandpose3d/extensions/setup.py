from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(name='dilation2d',
      ext_modules=[CppExtension('dilation2d', ['dilation2d.cpp'])],
      cmdclass={'build_ext': BuildExtension})
