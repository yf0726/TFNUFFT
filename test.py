import helper
import scipy.io as scio
import os
from nufft_cpu import NUFFT_cpu
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
gray = cm.gray

Nd = (64, 64, 64)  # time grid, tuple
Kd = (128,128,128)  # frequency grid, tuple
Jd = (3, 3, 3)  # interpolator

DATA_PATH = '/Users/yan/Documents/document/EPFL/thesis/codes/3dgridding/pynufft-master/src/data/'

image = np.load(DATA_PATH + 'phantom_3D_128_128_128.npz')['arr_0'][0::2, 0::2, 0::2]

folder_path = r'/Users/yan/Documents/document/EPFL/thesis/codes/3dgridding/testing/'
# trajectory = scio.loadmat(os.path.join(folder_path, '3D_traj.mat'))['traj']
trajectory = scio.loadmat(os.path.join(folder_path, 'test_full_spiral.mat'))['test']

om = trajectory['k'][0][0]
om = om / om.max() * 3  # normalized between (-pi,pi)

NufftObj = NUFFT_cpu()
NufftObj.plan(om, Nd, Kd, Jd)
kspace =NufftObj.forward(image)
# print(1)


restore_image1 = NufftObj.solve(kspace, 'cg', maxiter=200)
# restore_image2 = NufftObj.solve(kspace, 'L1TVOLS', maxiter=200, rho=0.1)
# adj_img = NufftObj.adjoint(kspace)
adj_img = NufftObj.selfadjoint(image)
Nmid = int(Nd[0] *0.1)
plt.figure(figsize=(10,10))
plt.subplot(2, 2, 1)
plt.imshow(abs(image[:, :, Nmid]), label='original', cmap=gray)
plt.title('original at slice %s'%Nmid)
plt.subplot(2, 2, 2)
plt.imshow(abs(restore_image1[:, :, Nmid]), label='CG', cmap=gray)
plt.title('conjugate gradient at slice %s'%Nmid)
plt.subplot(2, 2, 3)
plt.imshow(abs(adj_img[:, :, Nmid]), label='adjoint', cmap=gray)
plt.title('adjoint at slice %s'%Nmid)
plt.show()