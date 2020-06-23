import os
from tfNufft import tfNUFFT
from pyNufft import pyNUFFT
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
# image = image[:-1,:-1,:-1]


traj = np.load(os.path.join(DATA_PATH, '3dspiral.npy'))
new_traj = traj.copy()
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

batch_image = np.expand_dims(image, axis=0)
batch_image = np.tile(batch_image, [3,1,1,1])

tfNufftObj = tfNUFFT()
tfNufftObj.plan(om, Nd, Kd, Jd,batch_image.shape[0])
tfNufftObj.preapre_for_tf()
kspace = tfNufftObj.forward(batch_image)
adj_img1 = tfNufftObj.adjoint(kspace)

Nmid = int(Nd[2] * 0.5)
plt.figure(figsize=(10,10))
plt.subplot(1, 2, 1)
plt.imshow(abs(batch_image[0, :, :, Nmid]), label='original', cmap=gray)
plt.title('original at slice %s'%Nmid)
plt.subplot(1, 2, 2)
plt.imshow(abs(adj_img1[0, :, :, Nmid]), label='adj_ksp_nufft', cmap=gray)
plt.title('adjoint using ksp_nufft at slice %s'%Nmid)
plt.show()