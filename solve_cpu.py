"""
CPU solvers
======================================
"""

import scipy
# import numpy
# import helper
from scipy.sparse import linalg
import tensorflow as tf


# def cDiff(x, d_indx):
#     """
#     Compute image gradient, which needs the results of indxmap_diff(Nd)
#     :param x: The image array
#     :param d_indx: The index of the shifted image
#     :type x: numpy.float array, matrix size = Nd
#     :type d_indx: int32
#     :returns: x_diff: Image gradient determined by d_indx
#     :rtype: x_diff: numpy.complex64
#     """
#     x_diff = numpy.asarray(x.copy(), order='C')
#     x_diff.flat = x_diff.ravel()[d_indx] - x.ravel()
#     return x_diff


# def _create_kspace_sampling_density(nufft):
#     """
#     Compute k-space sampling density
#     """
#     y = numpy.ones(nufft.multi_M, dtype=numpy.complex64)
#     if nufft.parallel_flag is 1:
#         w = numpy.abs(nufft.xx2k(nufft.adjoint(y)))[..., 0]  # **2) ))
#     else:
#         w = numpy.abs(nufft.xx2k(nufft.adjoint(y)))  # **2) ))
#     nufft.st['w'] = w  # self.nufftobj.vec2k(w)
#     RTR = nufft.st['w']  # see __init__() in class "nufft"
#     return RTR


# def create_laplacian_kernel(nufft):
#     """
#     Create the multi-dimensional laplacian kernel in k-space
#
#     :param nufft: the NUFFT object
#     :return: uker: the multi-dimensional laplacian kernel in k-space (no fft shift used)
#     :rtype: numpy ndarray
#     """
# #===============================================================================
# # #        # Laplacian oeprator, convolution kernel in spatial domain
# #         # related to constraint
# #===============================================================================
#     uker = numpy.zeros(nufft.st['Kd'][:],dtype=numpy.complex64,order='C')
#     n_dims= numpy.size(nufft.st['Nd'])
#
#     ################################
#     #    Multi-dimensional laplacian kernel (generalize the above 1D - 3D to multi-dimensional arrays)
#     ################################
#
#     indx = [slice(0, 1) for ss in range(0, n_dims)] # create the n_dims dimensional slice which are all zeros
#     uker[tuple(indx)] = - 2.0*n_dims # Equivalent to  uker[0,0,0] = -6.0
#     for pp in range(0,n_dims):
# #         indx1 = indx.copy() # indexing the 1 Only for Python3
#         indx1 = list(indx)# indexing; adding list() for Python2/3 compatibility
#         indx1[pp] = 1
#         uker[tuple(indx1)] = 1
# #         indx1 = indx.copy() # indexing the -1  Only for Python3
#         indx1 = list(indx)# indexing the 1 Python2/3 compatible
#         indx1[pp] = -1
#         uker[tuple(indx1)] = 1
#     ################################
#     #    FFT of the multi-dimensional laplacian kernel
#     ################################
#     uker =numpy.fft.fftn(uker) #, self.nufftobj.st['Kd'], range(0,numpy.ndim(uker)))
#     return uker


# def _pipe_density(nufft, maxiter):
#     '''
#     Create the density function by iterative solution
#     Generate pHp matrix
#     '''
#     #         W = pipe_density(self.st['p'])
#     # sampling density function
#
#     W = numpy.ones(nufft.multi_M, dtype=nufft.dtype)
#     #         V1= self.st['p'].getH()
#     #     VVH = V.dot(V.getH())
#
#     for pp in range(0, maxiter):
#         #             E = self.st['p'].dot(V1.dot(W))
#
#         E = nufft.forward(nufft.adjoint(W))
#         W = (W / E)
#
#     return W


def solve(nufft, y, solver=None, *args, **kwargs):
    """
    Solve NUFFT.
    The current version supports solvers = 'cg'.

    :param nufft: NUFFT_cpu object
    :param y: (M,) array, non-uniform data, sampled ksp
    :return: x: image

    """

    if None == solver:
        solver = 'cg'
        print(" use default solver: conjugate gradient")

        """
        Hermitian matrix A
        cg: conjugate gradient
        bicgstab: biconjugate gradient stablizing
        bicg: biconjugate gradient
        gmres: 
        lgmres:
        """

    def spHsp(x):
        k = x.reshape(nufft.multi_Kd, order='C')
        k = nufft.k2y2k(k)
        return tf.reshape(k,(-1,1))

    A = scipy.sparse.linalg.LinearOperator((nufft.Kdprod * nufft.batch, nufft.Kdprod * nufft.batch), matvec=spHsp,
                                           rmatvec=spHsp, )

    methods = {'cg': scipy.sparse.linalg.cg,
               'bicgstab': scipy.sparse.linalg.bicgstab,
               'bicg': scipy.sparse.linalg.bicg,
               'gmres': scipy.sparse.linalg.gmres,
               'lgmres': scipy.sparse.linalg.lgmres,
               }
    k = nufft.y2k(y)
    k2 = methods[solver](A, k.numpy().ravel(), *args, **kwargs)  # ,show=True)

    xx = nufft.k2xx(k2[0].reshape(nufft.multi_Kd))
    x = xx / nufft.sn
    return x
