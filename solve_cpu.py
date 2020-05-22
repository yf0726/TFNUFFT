"""
CPU solvers
======================================
"""

import scipy
from scipy.sparse import linalg
import tensorflow as tf


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
