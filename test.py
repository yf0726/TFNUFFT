import os
from nufft_cpu import NUFFT_cpu
import numpy as np
import matplotlib.pyplot as plt
import cv2

from matplotlib import cm
gray = cm.gray

DATA_PATH = './data/'

image = np.load(DATA_PATH + 'phantom_3D_128_128_128.npz')['arr_0']

image = cv2.resize(image,(48,48))
image = image[:,:,::4]
image = image[:,:,:-2]

om = np.load(os.path.join(DATA_PATH, 'trajectory.npy'))

om = om / om.max() * 3  # normalized between (-pi,pi)

Nd = image.shape  # time grid, tuple
Kd = Nd  # frequency grid, tuple
Jd = (3, 3, 3)  # interpolator


NufftObj = NUFFT_cpu()
NufftObj.plan(om, Nd, Kd, Jd)
kspace =NufftObj.forward(image)

restore_image1 = NufftObj.solve(kspace, 'cg', maxiter=200)
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