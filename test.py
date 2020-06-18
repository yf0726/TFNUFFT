import os
from tfNufft import tfNUFFT
from pyNufft import pyNUFFT
import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib import cm
import scipy.io as scio
gray = cm.gray

DATA_PATH = './data/'

image = np.load(DATA_PATH + 'phantom_3D_128_128_128.npz')['arr_0']

image = cv2.resize(image,(48,48))
image = image[:,:,::4]
image = image[:,:,:-2]
image = image[:-1,:-1,:-1]



traj = np.load(os.path.join(DATA_PATH, 'sparse_trajectory.npy'))
new_traj = np.zeros_like(traj)
new_traj[:,0] = traj[:,1]
new_traj[:,1] = traj[:,0]

om = new_traj
om = om / om.max() * 3  # normalized between (-pi,pi)

Nd = image.shape  # time grid, tuple
Kd = Nd  # frequency grid, tuple
Jd = (3, 3, 3)  # interpolator

# Nufft implementaion using numpy
# pyNufftObj = pyNUFFT()
# pyNufftObj.plan(om, Nd, Kd, Jd)
# kspace = pyNufftObj.forward(image)
# adj_img2 = pyNufftObj.adjoint(kspace)
# restore_image2 = pyNufftObj.solve(kspace, 'cg', maxiter=200)

tfNufftObj = tfNUFFT()
tfNufftObj.plan(om, Nd, Kd, Jd)
tfNufftObj.preapre_for_tf()
kspace = tfNufftObj.forward(image)
adj_img1 = tfNufftObj.adjoint(kspace)
# restore_image1 = tfNufftObj.solve(kspace, 'cg', maxiter=200)

# To check validity of Nufft approximation
if Nd[0]//2==0:
    E_k = scio.loadmat('./data/E_k_even.mat')['k']
else:
    E_k = scio.loadmat('./data/E_k_odd.mat')['k']

adj_img2 = tfNufftObj.adjoint(E_k)

Nmid = int(Nd[0] * 0.5)
plt.figure(figsize=(10,10))
plt.subplot(2, 2, 1)
plt.imshow(abs(image[:, :, Nmid]), label='original', cmap=gray)
plt.title('original at slice %s'%Nmid)
plt.subplot(2, 2, 2)
plt.imshow(abs(adj_img1[:, :, Nmid]), label='adj_ksp_nufft', cmap=gray)
plt.title('adjoint using ksp_nufft at slice %s'%Nmid)
plt.subplot(2, 2, 3)
plt.imshow(abs(adj_img2[:, :, Nmid]), label='adj_E_k', cmap=gray)
plt.title('adjoint using E_k at slice %s'%Nmid)
plt.show()