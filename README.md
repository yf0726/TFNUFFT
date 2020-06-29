# Overview

This repotory is a TensorFlow 2.0 adaptation for [PyNufft](https://github.com/jyhmiinlin/pynufft/), which is an implementation of Fessler's min-max NUFFT.

# Background
Fast Fourier transform is a widely used in signal processing. However, it fails when the sampled data is non-uniform, eg. collect MR data using spiral trajectory.
Before MR image reconstruction, we need to resample the 'off-grid' (non-uniform) data back to Cartesian grid. This step is often called as gridding. 
Fessler proposed NUFFT using on min-max criterion, which solves this problem.

# Usage
Given an image in the shape of (batch, Nx, Ny, Nz) (notice that the first dimension should be for batch number), and a trajectory normalized between [-pi, pi], we can apply tfNufft as:
```
from tfNufft import tfNUFFT
tfNufftObj = tfNUFFT() 
tfNufftObj.plan(trajectory, Nd, Kd, Jd, batch_image.shape[0])
#Nd: image size (Nx,Ny,Nz), Kd k-space size, Jd interpolation size
tfNufftObj.preapre_for_tf() #convert array to tensor
Nufft_k = tfNufftObj.forward(batch_image) #forward and get non-uniform spectrum data
Nufft_adj = tfNufftObj.adjoint(Nufft_k) #apply adjoint operation
```

# Folders
- data: data folder, containing image and trajectory (both 2D and 3D).
- pyNufft.py: The original pyNufft written by [jyhmiinlin](https://github.com/jyhmiinlin/pynufft/).
- tfNufft.py: The TF2.0 adapation.
- helper_numpy.py, solve_cpu.py: helper functions.
- test.ipynb: The notebook validates the Nufft code (showing that the result of forward tfNufft is correct; auto-grad and batch work).
- test.py: A simple example for image reconstruction using tfNufft.

# Reference

1. Fessler J A, Sutton B P. Nonuniform fast Fourier transforms using min-max interpolation[J]. IEEE transactions on signal processing, 2003, 51(2): 560-574.
2. Lin, Jyh-Miin. “Python Non-Uniform Fast Fourier Transform (PyNUFFT): An Accelerated Non-Cartesian MRI Package on a Heterogeneous Platform (CPU/GPU).” Journal of Imaging 4.3 (2018): 51.
3. J.-M. Lin and H.-W. Chung, Pynufft: python non-uniform fast Fourier transform for MRI Building Bridges in Medical Sciences 2017, St John’s College, CB2 1TP Cambridge, UK
