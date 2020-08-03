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
        self.ft_axes = None  # : initial value: ()
        self.batch = None  # : initial value: None
        pass

    def plan(self, om, Nd, Kd, Jd):
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

        self.batch = Nd[0]
        self.Nd = Nd[1:]
        self.ndims = len(self.Nd)  # : initial value: len(Nd)
        self.Kd = Kd
        self.Jd = Jd


        self.st = helper.plan(om, self.Nd, self.Kd, self.Jd)

        self.sn = self.st['sn']
        self.M = (self.st['M'],)

        self.sp = self.st['p'].copy().tocsr()  # interpolator
        self.spH = (self.st['p'].getH().copy()).tocsr()  # return the Hermitian transpose.

        self.spdense = self.st['p'].copy().todense()

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
        Convert scipy.csr matrix to tensorflow sparse tensor
        :param X: The scipy.sparseTensor, shape(self.M, self.nelem)
        :return: tensorflow sparse tensor
        """
        coo = X.tocoo()
        indices = np.mat([coo.row, coo.col]).transpose()
        sparse = tf.sparse.SparseTensor(indices, coo.data, coo.shape)
        return tf.cast(sparse, dtype=self.dtype)

    def solve(self, y, solver=None, *args, **kwargs):
        """
        Solve NUFFT.
        :param y: data, tensor.complex64. The shape = (M,) or (M, self.batch)
        :param solver: currently only 'cg' conjugate gradient is provided
        :param maxiter: the number of iterations
        :type y: Tensor, dtype = tensor.complex64
        :type solver: string
        :type maxiter: int
        :return: numpy array with size of (self.batch,) + self.Nd.
        :rtype: Tensor.
        """
        from solve_cpu import solve
        x2 = solve(self, y, solver, *args, **kwargs)
        return x2

    def forward(self, x):
        """
        Forward NUFFT
        :param x: The input tensor, with the size of (self.batch,) + self.Nd.
        :type: Tensor with the dtype of self.dtype, tf.complex64 default
        :return: y: The output tensor, with the size of (M,) or (M, self.batch). sampled k space
        :rtype: Tensor with the dtype of self.dtype, tf.complex64 default
        """
        x = tf.cast(x, dtype=self.dtype)
        k = self.x2xx(x) #Scaling with self.sn, scaling factor
        k = self.xx2k(k) #fft
        y = self.k2y(k) #A
        y = tf.transpose(y)
        return y

    def adjoint(self, y):
        """
        Adjoint NUFFT

        :param y: The non-uniform kspace data, with the size of (M,) or (M, self.batch)
        :type: tensor with the dtype of tf.complex64
        :return: x: The output tensor, with the size of (self.batch,) + self.Nd.
        :rtype: tensor with the dtype of tf.complex64
        """
        k = self.y2k(tf.transpose(y)) #Non-uniform K space to gridded K space
        k = self.k2xx(k) #ifft and cropping
        x = self.xx2x(k) #scaling
        return x

    def selfadjoint(self, x):
        """
        selfadjoint NUFFT (Toeplitz)

        :param x: The input tensor, with size = (self.batch,) + self.Nd
        :type: Tensor with dtype = tf.complex64
        :return: x: The output tensor, with size = (self.batch,) + self.Nd
        :rtype: Tensor with dtype = tf.complex64
        """
        x2 = self.xx2x(self.k2xx(self.k2y2k(self.xx2k(self.x2xx(x)))))

        return x2

    def x2xx(self, x):
        """
        Inplace multiplication of self.x_Nd by the scaling factor self.sn.
        :param x: The input equispaced image, with the size of (self.batch,) + self.Nd.
        :type: Tensor with dtype = tf.complex64
        :return: xx: The scaled image, with the size of (self.batch,) + Nd.
        :rtype: Tensor with dtype = tf.complex64
        """
        xx = x * self.sn
        return xx

    def xx2k(self, xx):
        """
        In this step we perform oversampled FFT on padded frequency domain, shape self.Kd.
        Notice that the forward FFT does not perform normalization and the inverse transform are scaled by 1/n.
        https://docs.scipy.org/doc/numpy-1.15.0/reference/routines.fft.html#normalization
        :param xx: The scaled image, with the size of (self.batch,) + self.Nd.
        :type: Tensor with dtype = self.dtype, tf.complex64 default
        :return: The oversampled FFT data, with the size of (self.batch,) + self.Kd
        :rtype: Tensor with dtype = self.dtype, tf.complex64 default
        """
        if self.Nd[0] >= self.Kd[0]:
            self.padding = tf.constant((0,) + self.Nd) - tf.constant((0,) + self.Kd)
        else:
            self.padding = tf.constant((0,) + self.Kd) - tf.constant((0,) + self.Nd)
        self.padding = tf.reshape(self.padding, (-1,1))
        self.padding = tf.concat([tf.zeros((len(self.padding), 1), dtype=tf.int32), self.padding], 1)
        output_x = tf.pad(xx, self.padding, 'CONSTANT')# batch dimension does not need padding

        if len(self.Kd) == 3:
            k = tf.signal.fft3d(output_x)
        elif len(self.Kd) == 2:
            k = tf.signal.fft2d(output_x)
        k = tf.cast(k, dtype=self.dtype)
        return k

    def k2vec(self, k):
        """
        :param k: The oversampled FFT data of input image, with size of (batch,) + self.Kd.
        :type: Tensor with dtype = self.dtype, tf.complex64 default
        :return: The vectorized FFT data, with size of (self.Kdprod, self.batch)
        :rtype: Tensor with dtype = self.dtype, tf.complex64 default
        """
        k_vec = tf.reshape(k, (self.batch, -1))
        k_vec = tf.transpose(k_vec)
        return k_vec

    def vec2y(self, k_vec):
        """
        :param k_vec: The vectorized FFT data.
        :type: Tensor with dtype = self.dtype, tf.complex64 default
        :return: Non-uniform K space data with size of self.M + (self.batch,)
        :rtype: Tensor with dtype = self.dtype, tf.complex64 default
        """
        y = tf.sparse.sparse_dense_matmul(self.sp, k_vec)
        return y

    def k2y(self, k):
        """
        Generate k space data from oversampled FFT data.
        :param k: The oversampled FFT data, with size of (batch,) + self.Kd.
        :type: Tensor with dtype = self.dtype, tf.complex64 default
        :return: Non-uniform K space data, with size of self.M + (self.batch,)
        :rtype: Tensor with dtype = self.dtype, tf.complex64 default
        """
        k = self.k2vec(k) #vectorized the oversampled FFT data
        y = self.vec2y(k) #generate non-uniform K space data by the Sparse Matrix-Vector Multiplication
        return y

    def y2vec(self, y):
        '''
        Re-gridding from non-uniform k space data.
        :param y: The k space data.
        :type: Tensor with dtype = self.dtype, tf.complex64 default
        :return: Vectorized gridded FT data.
        :rtype: Tensor with dtype = self.dtype, tf.complex64 default
        '''
        k_vec = tf.sparse.sparse_dense_matmul(self.spH, y)

        return k_vec

    def vec2k(self, k_vec):
        '''
        Sorting the vector to k-spectrum Kd array
        :param k_vec: Vectorized gridded FT data.
        :type: Tensor with dtype = self.dtype, tf.complex64 default
        :return: Gridded FT data with shape (self.batch,) + self.Kd
        :rtype: Tensor with dtype = self.dtype, tf.complex64 default
        '''
        k_vec = tf.transpose(k_vec)
        k = tf.reshape(k_vec, (self.batch,)+self.Kd)
        return k

    def y2k(self, y):
        """
        Private: gridding by the Sparse Matrix-Vector Multiplication
        :param y: The non-uniform kspace data, with the size of self.M + (self.batch,)
        :type: Tensor with dtype = self.dtype, tf.complex64 default
        :return: Gridded FT data with shape (self.batch,) + self.Kd
        :rtype: Tensor with dtype = self.dtype, tf.complex64 default
        """
        k_vec = self.y2vec(y)
        k = self.vec2k(k_vec) #reshape vector to Kd
        # k = tf.cast(k, dtype=self.dtype)
        return k

    def k2xx(self, k):
        """
        Private: the inverse FFT and image cropping (which is the reverse of
                 xx2k() method)
        :param k: Gridded FT data with size (self.batch,) + self.Kd
        :type: Tensor with dtype = self.dtype, tf.complex64 default
        :return: Cropped FT data with size (self.batch,) + self.Nd
        :rtype: Tensor with dtype = self.dtype, tf.complex64 default
        """
        if len(self.Kd) == 3:
            k = tf.signal.ifft3d(k)
        elif len(self.Kd) == 2:
            k = tf.signal.ifft2d(k)

        if self.padding[1][1] > 0: #if we padded the frequency domain
            if len(self.Kd) == 3:
                xx = k[:,
                     :-1*self.padding[1][1],
                     :-1*self.padding[3][1]]
            elif len(self.Kd) == 2:
                xx = k[:,
                     :-1 * self.padding[1][1],
                     :-1 * self.padding[2][1]]
        else:
            xx = k
        xx = tf.cast(xx, dtype=self.dtype)
        return xx

    def xx2x(self, xx):
        """
        Private: rescaling, which is identical to the  x2xx() method
        :param xx: The input image, with the size of (self.batch,) + self.Nd
        :type: Tensor with dtype = self.dtype, tf.complex64 default
        :return: The scaled image, with the size of (self.batch,) + self.Nd
        :rtype: Tensor with dtype = self.dtype, tf.complex64 default
        """
        x = self.x2xx(xx)
        return x

    def k2y2k(self, k):
        """
        The integrated interpolation-gridding by the Sparse Matrix-Vector Multiplication
        :param k: The oversampled FFT data of input image, with size of (self.batch,) + self.Kd
        :type: Numpy array from solve_cpu.spHsp
        :return: Gridded FT data
        :rtype: Tensor with dtype = self.dtype, tf.complex64 default
        """
        k = tf.cast(k, dtype=self.dtype)
        Xk = self.k2vec(k)
        y = self.vec2y(Xk)
        y = self.y2vec(y)
        k = self.vec2k(y)
        return k

    def test_derivative(self, x):
        """
        Not using SparseTensor but numpy matrix to test derivative.
        """
        # sp = tf.sparse.to_dense(self.sp) # error:Could not find valid device for node. Node:{{node SparseToDense}}
        sp = tf.cast(self.spdense, dtype=self.dtype)
        x = tf.cast(x, dtype=self.dtype)
        k = self.x2xx(x) #Scaling with self.sn, scaling factor
        k = self.xx2k(k) #fft
        k_vec = self.k2vec(k) #vectorized the oversampled FFT data
        y = tf.linalg.matmul(sp, k_vec)
        y = tf.transpose(y)
        return y


