"""
tfNUFFT class
This code is an Tensorflow 2.0 adaptation of pyNUFFT: https://github.com/jyhmiinlin/pynufft
An implementation of the min-max NUFFT of Fessler and Sutton
=======================================
"""

from __future__ import absolute_import
import tensorflow as tf
import numpy as np
import helper_numpy as helper


class tfNUFFT:
    """
    Class NUFFT implementing using tensorflow 2.0
   """

    def __init__(self):
        """
        Constructor.

        :param None:
        :type None: Python NoneType
        :return: NUFFT: the NUFFT instance
        :rtype: NUFFT: the NUFFT class

        :Example:
        # >>> from tfnufft import NUFFT
        # >>> NufftObj = NUFFT()
        """
        self.dtype = tf.complex64  # : initial value: tf.complex64
        self.debug = 0  #: initial value: 0
        self.Nd = ()  # : initial value: ()
        self.Kd = ()  # : initial value: ()
        self.Jd = ()  #: initial value: ()
        self.ndims = 0  # : initial value: 0
        self.ft_axes = ()  # : initial value: ()
        self.batch = None  # : initial value: None
        pass

    def plan(self, om, Nd, Kd, Jd, ft_axes=None, batch=None):
        """
        Plan the NUFFT object with the geometry provided.

        :param om: The M off-grid locates in the frequency domain,
                    which is normalized between [-pi, pi]
        :param Nd: The matrix size of the equispaced image.
                   Example: Nd=(256,256) for a 2D image;
                             Nd = (128,128,128) for a 3D image
        :param Kd: The matrix size of the oversampled frequency grid.
                   Example: Kd=(512,512) for 2D image;
                            Kd = (256,256,256) for a 3D image
        :param Jd: The interpolator size.
                   Example: Jd=(6,6) for 2D image;
                            Jd = (6,6,6) for a 3D image
        :param ft_axes: (Optional) The axes for Fourier transform.
                        The default is all axes if 'None' is given.
        :param batch: (Optional) Batch mode.
                     If the batch is provided, the last appended axis is the number
                     of identical NUFFT to be transformed.
                     The default is 'None'.
        :type om: numpy.float array, matrix size = M * ndims
        :type Nd: tuple, ndims integer elements.
        :type Kd: tuple, ndims integer elements.
        :type Jd: tuple, ndims integer elements.
        :type ft_axes: None, or tuple with optional integer elements.
        :type batch: None, or integer
        :returns: 0
        :rtype: int

        :ivar Nd: initial value: Nd
        :ivar Kd: initial value: Kd
        :ivar Jd: initial value: Jd
        :ivar ft_axes: initial value: None
        :ivar batch: initial value: None

        :Example:

        # >>> from tfNufft import tfNUFFT
        # >>> NufftObj = NUFFT()
        # >>> NufftObj.plan(om, Nd, Kd, Jd)

        """

        self.ndims = len(Nd)  # : initial value: len(Nd)
        if ft_axes is None:
            ft_axes = range(0, self.ndims)
        self.ft_axes = ft_axes  # default: all axes (range(0, self.ndims)

        self.st = helper.plan(om, Nd, Kd, Jd, ft_axes=ft_axes)

        self.Nd = self.st['Nd']
        self.Kd = self.st['Kd']
        self.sn = self.st['sn']
        # self.sn = tf.ones(shape=self.st['sn'].shape)

        if batch is None:  # single-coil
            self.parallel_flag = 0
            self.batch = 1
        else:  # multi-coil
            self.parallel_flag = 1 #for current code multi-coil is not yet implemented
            self.batch = batch

        # self.multi_Nd = self.Nd
        # self.multi_Kd = self.Kd
        # self.multi_M = (self.st['M'],)
        self.M = (self.st['M'],)
        # self.multi_prodKd = (tf.math.reduce_prod(self.Kd),)

        self.sp = self.st['p'].copy().tocsr() #interpolator
        self.spH = (self.st['p'].getH().copy()).tocsr() #return the Hermitian transpose.

        self.Kdprod = np.int32(np.prod(self.st['Kd']))
        self.Jdprod = np.int32(np.prod(self.st['Jd']))
        del self.st

        self.Ndorder, self.Kdorder, self.nelem = helper.preindex_copy(
            self.Nd,
            self.Kd)
        return 0

    def preapre_for_tf(self):
        """
        Convert numpy arrays needed in forward and adjoint operations into tensor.
        """
        self.sn = tf.cast(self.sn, dtype=self.dtype)
        self.sp = self.convert_sparse_matrix_to_sparse_tensor(self.sp)
        self.spH = self.convert_sparse_matrix_to_sparse_tensor(self.spH)

    def convert_sparse_matrix_to_sparse_tensor(self, X):
        """
        Convert scipy.csr matrix to tensorflow sparse matrix
        :param X: The scipy.sparseTensor, shape(self.M, self.nelem)
        :return: tensorflow sparse matrix
        """
        coo = X.tocoo()
        indices = np.mat([coo.row, coo.col]).transpose()
        sparse = tf.sparse.SparseTensor(indices, coo.data, coo.shape)
        return tf.cast(sparse, dtype=self.dtype)

    def solve(self, y, solver=None, *args, **kwargs):
        """
        Solve NUFFT.
        :param y: data, tensor.complex64. The shape = (M,) or (M, batch)
        :param solver: currently only 'cg' conjugate gradient is provided
        :param maxiter: the number of iterations
        :type y: Tensor, dtype = tensor.complex64
        :type solver: string
        :type maxiter: int
        :return: numpy array with size of Nd or  Nd + (batch,)
        :rtype: Tensor.
        """
        from solve_cpu import solve
        x2 = solve(self, y, solver, *args, **kwargs)
        return x2

    def forward(self, x):
        """
        Forward NUFFT
        :param x: The input tensor, with the size of Nd or Nd + (batch,).
        :type: Tensor with the dtype of self.dtype, tf.complex64 default
        :return: y: The output tensor, with the size of (M,) or (M, batch). sampled k space
        :rtype: Tensor with the dtype of self.dtype, tf.complex64 default
        """
        x = tf.cast(x, dtype=self.dtype)
        k = self._x2xx(x) #Scaling with self.sn, scaling factor
        k = self._xx2k(k) #fft
        y = self._k2y(k) #A
        return y

    def adjoint(self, y):
        """
        Adjoint NUFFT

        :param y: The non-uniform kspace data, with the size of (M,) or (M, batch)
        :type: tensor with the dtype of tf.complex64
        :return: x: The output tensor, with the size of Nd or Nd + (batch, )
        :rtype: tensor with the dtype of tf.complex64
        """
        k = self._y2k(y) #Non-uniform K space to gridded K space
        k = self._k2xx(k) #ifft and cropping
        x = self._xx2x(k) #scaling
        return x

    def selfadjoint(self, x):
        """
        selfadjoint NUFFT (Toeplitz)

        :param x: The input tensor, with size = Nd
        :type: Tensor with dtype = tf.complex64
        :return: x: The output tensor, with size = Nd
        :rtype: Tensor with dtype = tf.complex64
        """
        x2 = self._xx2x(self._k2xx(self.k2y2k(self._xx2k(self._x2xx(x)))))

        return x2

    def _x2xx(self, x):
        """
        Inplace multiplication of self.x_Nd by the scaling factor self.sn.
        :param x: The input equispaced image, with the size of self.Nd
        :type: Tensor with dtype = tf.complex64
        :return: xx: The scaled image, with the size of self.Nd
        :rtype: Tensor with dtype = tf.complex64
        """
        xx = x * self.sn
        return xx

    def _xx2k(self, xx):
        """
        In this step we perform oversampled FFT on padded frequency domain, shape Kd.
        Notice that the forward FFT does not perform normalization and the inverse transform are scaled by 1/n.
        https://docs.scipy.org/doc/numpy-1.15.0/reference/routines.fft.html#normalization
        :param xx: The scaled image, with the size of self.Nd
        :type: Tensor with dtype = self.dtype, tf.complex64 default
        :return: The oversampled FFT data, with the size of self.Kd
        :rtype: Tensor with dtype = self.dtype, tf.complex64 default
        """
        self.padding = []
        for (Nd, Kd) in zip(self.Nd, self.Kd):
            self.padding.append([0, (Kd-Nd)])
            #first dimension means zeros added before, second dimension means zeros added after

        self.padding = tf.constant(self.padding)
        output_x = tf.pad(xx, self.padding, 'CONSTANT')

        if len(self.Kd) == 3:
            k = tf.signal.fft3d(output_x)
        elif len(self.Kd) == 2:
            k = tf.signal.fft2d(output_x)
        k = tf.cast(k, dtype=self.dtype)
        return k

    def _k2vec(self, k):
        """
        :param k: The oversampled FFT data of input image, with size of self.Kd
        :type: Tensor with dtype = self.dtype, tf.complex64 default
        :return: The vectorized FFT data, with size of (self.Kdprod, 1)
        :rtype: Tensor with dtype = self.dtype, tf.complex64 default
        """
        k_vec = tf.reshape(k, (-1,1))
        # k_vec = tf.cast(k_vec, self.dtype)
        return k_vec

    def _vec2y(self, k_vec):
        """
        :param k_vec: The vectorized FFT data.
        :type: Tensor with dtype = self.dtype, tf.complex64 default
        :return: Non-uniform K space data with size of self.M
        :rtype: Tensor with dtype = self.dtype, tf.complex64 default
        """
        y = tf.sparse.sparse_dense_matmul(self.sp, k_vec)
        return y

    def _k2y(self, k):
        """
        Generate k space data from oversampled FFT data.
        :param k: The oversampled FFT data, with size of self.Kd
        :type: Tensor with dtype = self.dtype, tf.complex64 default
        :return: Non-uniform K space data, with size of self.M
        :rtype: Tensor with dtype = self.dtype, tf.complex64 default
        """
        k = self._k2vec(k) #vectorized the oversampled FFT data
        y = self._vec2y(k) #generate non-uniform K space data by the Sparse Matrix-Vector Multiplication
        return y

    def _y2vec(self, y):
        '''
        Re-gridding from non-uniform k space data.
        :param y: The k space data.
        :type: Tensor with dtype = self.dtype, tf.complex64 default
        :return: Vectorized gridded FT data.
        :rtype: Tensor with dtype = self.dtype, tf.complex64 default
        '''
        # k_vec = self.spH.dot(y)
        k_vec = tf.sparse.sparse_dense_matmul(self.spH, y)
        return k_vec

    def vec2k(self, k_vec):
        '''
        Sorting the vector to k-spectrum Kd array
        :param k_vec: Vectorized gridded FT data.
        :type: Tensor with dtype = self.dtype, tf.complex64 default
        :return: Gridded FT data with shape self.Kd
        :rtype: Tensor with dtype = self.dtype, tf.complex64 default
        '''
        k = tf.reshape(k_vec, self.Kd)
        return k

    def _y2k(self, y):
        """
        Private: gridding by the Sparse Matrix-Vector Multiplication
        :param y: The non-uniform kspace data, with the size of (M,) or (M, batch)
        :type: Tensor with dtype = self.dtype, tf.complex64 default
        :return: Gridded FT data with shape self.Kd
        :rtype: Tensor with dtype = self.dtype, tf.complex64 default
        """
        k_vec = self._y2vec(y)
        k = self.vec2k(k_vec) #reshape vector to Kd
        # k = tf.cast(k, dtype=self.dtype)
        return k

    def _k2xx(self, k):
        """
        Private: the inverse FFT and image cropping (which is the reverse of
                 _xx2k() method)
        :param k: Gridded FT data with size self.Kd
        :type: Tensor with dtype = self.dtype, tf.complex64 default
        :return: Cropped FT data with size self.Nd
        :rtype: Tensor with dtype = self.dtype, tf.complex64 default
        """
        if len(self.Kd) == 3:
            k = tf.signal.ifft3d(k)
            # k = tf.signal.fftshift(tf.signal.ifft3d(tf.signal.ifftshift(k)))
        elif len(self.Kd) == 2:
            k = tf.signal.ifft2d(k)
            # k = tf.signal.fftshift(tf.signal.ifft2d(tf.signal.ifftshift(k)))

        if self.padding[0,1]>0: #if we padded the frequency domain
            xx = k[:-1*self.padding[0,1],
                   :-1*self.padding[1,1],
                   :-1*self.padding[2,1]]
        else:
            xx = k
        xx = tf.cast(xx, dtype=self.dtype)
        return xx

    def _xx2x(self, xx):
        """
        Private: rescaling, which is identical to the  _x2xx() method
        :param xx: The input image, with the size of self.Nd
        :type: Tensor with dtype = self.dtype, tf.complex64 default
        :return: The scaled image, with the size of self.Nd
        :rtype: Tensor with dtype = self.dtype, tf.complex64 default
        """
        x = self._x2xx(xx)
        return x

    def k2y2k(self, k):
        """
        The integrated interpolation-gridding by the Sparse Matrix-Vector Multiplication
        :param k: The oversampled FFT data of input image, with size of self.Kd
        :type: Numpy array from solve_cpu.spHsp
        :return: Gridded FT data with shape self.Kd
        :rtype: Tensor with dtype = self.dtype, tf.complex64 default
        """
        k = tf.cast(k, dtype=self.dtype)
        Xk = self._k2vec(k)
        y = self._vec2y(Xk)
        y = self._y2vec(y)
        k = self.vec2k(y)
        return k
