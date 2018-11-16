# ColorHandPose3D-PyTorch
This project implements ColorHandPose3D using the PyTorch framework. The original project is available at [https://github.com/lmb-freiburg/hand3d].

# Installation
The dilation2d extension is available for both CPU and CUDA. To install, navigate to `extensions/dilation2d/` and run `python setup.py install`. This will compile and install the extension. It was originally compiled under CUDA 9.2 and PyTorch 0.4.1.

# Dependencies
- CUDA 9.2
- PyTorch 0.4.1
- RoIAlign.pytorch [https://github.com/longcw/RoIAlign.pytorch]

# TO-DO
- RHD test set evaluations
- RHD training routine

# References
Zimmermann, C., & Brox, T. (2017). Learning to Estimate 3D Hand Pose from Single RGB Images. Retrieved from [http://arxiv.org/abs/1705.01389]
