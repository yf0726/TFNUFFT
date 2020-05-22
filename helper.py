import tensorflow as tf
import math
dtype = tf.complex64

import numpy as np
import scipy

def plan(om, Nd, Kd, Jd, ft_axes = None):
    """
    Plan for the NUFFT object.

    :param om: Coordinate
    :param Nd: Image shape
    :param Kd: Oversampled grid shape
    :param Jd: Interpolator size
    :param ft_axes: Axes where FFT takes place
    :param format: Output format of the interpolator.
                    'CSR': the precomputed Compressed Sparse Row (CSR) matrix.
                    'pELL': partial ELLPACK which precomputes the concatenated 1D interpolators.
    :type om: numpy.float
    :type Nd: tuple of int
    :type Kd: tuple of int
    :type Jd: tuple of int
    :type ft_axes: tuple of int
    :type format: string, 'CSR' or 'pELL'
    :return st: dictionary for NUFFT

    """

    if type(Nd) != tuple:
        raise TypeError('Nd must be tuple, e.g. (256, 256)')

    if type(Kd) != tuple:
        raise TypeError('Kd must be tuple, e.g. (512, 512)')

    if type(Jd) != tuple:
        raise TypeError('Jd must be tuple, e.g. (6, 6)')

    if (len(Nd) != len(Kd)) | (len(Nd) != len(Jd))  | len(Kd) != len(Jd):
        raise KeyError('Nd, Kd, Jd must be in the same length, e.g. Nd=(256,256),Kd=(512,512),Jd=(6,6)')

    # dd = np.size(Nd)
    dd = tf.shape(Nd).numpy()[0]

    if ft_axes is None:
        ft_axes = tuple(xx for xx in range(0, dd)) # ft_axes = (0,1,2) when dd = 3

    # print('ft_axes = ', ft_axes)
    ft_flag = () # tensor

    for pp in range(0, dd):
        if pp in ft_axes:
            ft_flag += (True, )
        else:
            ft_flag += (False, )
    # print('ft_flag = ', ft_flag)
###############################################################
# check input errors
###############################################################
    st = {}


###############################################################
# First, get alpha and beta: the weighting and freq
# of formula (28) in Fessler's paper
# in order to create slow-varying image space scaling
###############################################################

    # st['tol'] = 0
    st['Jd'] = Jd
    st['Nd'] = Nd
    st['Kd'] = Kd
    M = om.shape[0]
    # st['M'] = np.int32(M)
    st['M'] = tf.cast(M, tf.int32)
    st['om'] = om

###############################################################
# create scaling factors st['sn'] given alpha/beta
# higher dimension implementation
###############################################################

    """
    Now compute the 1D scaling factors
    snd: list
    """

    for dimid in range(0, dd):

        (tmp_alpha, tmp_beta) = nufft_alpha_kb_fit(
            Nd[dimid], Jd[dimid], Kd[dimid])
        st.setdefault('alpha', []).append(tmp_alpha)
        st.setdefault('beta', []).append(tmp_beta)

    snd = [] ## formula 28
    for dimid in range(0, dd):
        snd += [nufft_scale(
            Nd[dimid],
            Kd[dimid],
            st['alpha'][dimid],
            st['beta'][dimid]), ]
    """
     higher-order Kronecker product of all dimensions
    """

    # [J? M] interpolation coefficient vectors.
    # Iterate over all dimensions and
    # multiply the coefficients of all dimensions

    ud = []
    for dimid in range(0, dd):  # iterate through all dimensions
        N = Nd[dimid]
        J = Jd[dimid]
        K = Kd[dimid]
        alpha = st['alpha'][dimid]
        beta = st['beta'][dimid]
       ###############################################################
        # formula 29 , 26 of Fessler's paper
        ###############################################################

        # pseudo-inverse of CSSC using large N approx [J? J?]
        if ft_flag[dimid] is True:

            ud += [min_max(N, J, K, alpha, beta, om[:, dimid], ft_flag[dimid]),] ## min_max interpolator's coefficient

        else:
            ud += [tf.ones((1, M), dtype = dtype).T,]

    """
    Now compute the column indices for 1D interpolators
    Each length-Jd interpolator includes Jd points, which are linked to Jd k-space locations
    kd is a tuple storing the 1D interpolators.
    A following Kronecker product will be needed.
    """
    kd = []
    for dimid in range(0, dd):  # iterate over all dimensions
        tmp = OMEGA_k(Jd[dimid],Kd[dimid], om[:,dimid], Kd, dimid, dd, ft_flag[dimid])
        kd += [tf.transpose(tmp), ]

    CSR = full_kron(ud, kd, Jd, Kd, M)
    st['p'] = CSR
    st['sn'] = tf.math.real(kronecker_scale(snd)) # only real scaling is

    return st #new


def nufft_alpha_kb_fit(N, J, K):
    """
    Find parameters alpha and beta for scaling factor st['sn']
    The alpha is hardwired as [1,0,0...] when J = 1 (uniform scaling factor)

    :param N: size of image
    :param J: size of interpolator
    :param K: size of oversampled k-space
    :type N: int
    :type J: int
    :type K: int
    :returns: alphas:
    :returns: beta:
    :rtype: alphas: list of float
    :rtype: beta:
    """

    beta = 1
    Nmid = (N - 1.0) / 2.0
    if N > 40:
        L = 13
    else:
        # L = np.ceil(N / 3).astype(np.int16)
        L = tf.cast(tf.math.ceil(N/3), tf.int16)

    # nlist = np.arange(0, N) * 1.0 - Nmid
    nlist = tf.range(0, N, dtype=tf.float32) - Nmid
    if J > 1:
        (kb_a, kb_m) = kaiser_bessel('string', J, 'best', 0, K / N)
        sn_kaiser = 1 / kaiser_bessel_ft(nlist / K, J, kb_a, kb_m, 1.0)
    elif J == 1:  # The case when samples are on regular grids
        # sn_kaiser = numpy.ones((1, N), dtype=dtype)
        sn_kaiser = tf.ones((1, N), dtype=dtype)

    gam = 2 * math.pi / K
    X_ant = beta * gam * tf.reshape(nlist, (-1, 1))
    X_post = tf.reshape(tf.range(0, L + 1,dtype=tf.float32),(1, -1))
    X = tf.multiply(X_post, X_ant)
    X = tf.math.cos(X)
    X = tf.cast(X, dtype=dtype)
    sn_kaiser = tf.transpose(tf.reshape(sn_kaiser,(1,-1)))
    sn_kaiser = tf.cast(tf.math.conj(sn_kaiser),dtype=dtype)
    # alphas = tf.linalg.lstsq(X, sn_kaiser) ## Seems like there are some data type issue with tf.linalg.lstsq
    alphas = np.linalg.lstsq(np.nan_to_num(X.numpy()), np.nan_to_num(sn_kaiser.numpy()), rcond=-1)[0]
    alphas = np.real(alphas)

    if J > 1:
        alphas[0] = alphas[0]
        alphas[1:] = alphas[1:] / 2.0
    elif J == 1:  # cases on grids
        alphas[0] = 1.0
        alphas[1:] = 0.0

    return (alphas, beta)

def kaiser_bessel(x, J, alpha, kb_m, K_N):
    if K_N != 2:
        kb_m = 0
        alpha = 2.34 * J
    else:
        kb_m = 0

        # Parameters in Fessler's code
        # because it was experimentally determined to be the best!
        # input: number of interpolation points
        # output: Kaiser_bessel parameter

        jlist_bestzn = {2: 2.5,
                        3: 2.27,
                        4: 2.31,
                        5: 2.34,
                        6: 2.32,
                        7: 2.32,
                        8: 2.35,
                        9: 2.34,
                        10: 2.34,
                        11: 2.35,
                        12: 2.34,
                        13: 2.35,
                        14: 2.35,
                        15: 2.35,
                        16: 2.33}

        if J in jlist_bestzn:
            alpha = J * jlist_bestzn[J]
        else:
            tmp_key = (jlist_bestzn.keys())
            min_ind = tf.math.argmin(abs(tmp_key - J * tf.ones(len(tmp_key))))
            p_J = tmp_key[min_ind]
            alpha = J * jlist_bestzn[p_J]
    kb_a = alpha
    return (kb_a, kb_m)

def kaiser_bessel_ft(u, J, alpha, kb_m, d):
    '''
    Interpolation weight for given J/alpha/kb-m
    '''

    u = u.numpy() * (1.0 + 0.0j)
    import scipy.special
    z = tf.math.sqrt((2 * math.pi * (J / 2) * u) ** 2.0 - alpha ** 2.0)
    nu = d / 2 + kb_m
    y = ((2 * math.pi) ** (d / 2)) * ((J / 2) ** d) * (alpha ** kb_m) / \
        scipy.special.iv(kb_m, alpha) * scipy.special.jv(nu, z) / (z ** nu)
    y = tf.math.real(y)
    return y

def nufft_scale(Nd, Kd, alpha, beta):
    Nmid = (Nd - 1) / 2.0
    sn = nufft_scale1(Nd, Kd, alpha, beta, Nmid)
    # dd = numpy.size(Nd)
    # if dd == 1:
    #     sn = nufft_scale1(Nd, Kd, alpha, beta, Nmid)
    # else:
    #     sn = 1
    #     for dimid in numpy.arange(0, dd):
    #         tmp = nufft_scale1(Nd[dimid], Kd[dimid], alpha[dimid],
    #                            beta[dimid], Nmid[dimid])
    #         sn = numpy.dot(list(sn), tmp.H)
    return sn

def nufft_scale1(N, K, alpha, beta, Nmid): ## formula 28
    '''
    Calculate image space scaling factor
    '''
#     import types
#     if alpha is types.ComplexType:
    alpha = tf.math.real(alpha)
    alpha = tf.cast(alpha,dtype=dtype)
#         print('complex alpha may not work, but I just let it as')

    L = len(alpha) - 1
    if L > 0:
        sn = tf.zeros((N, 1), dtype=dtype)
        n = tf.reshape(tf.range(0, N, dtype=type(Nmid)),(-1, 1))
        i_gam_n_n0 = (2 * math.pi / K) * (n - Nmid) * beta
        # i_gam_n_n0 = 1j * tf.cast(i_gam_n_n0, dtype=dtype)
        for l1 in range(-L, L + 1):
            alf = alpha[abs(l1)]
            if l1 < 0:
                alf = tf.math.conj(alf)
            sn = sn + alf * tf.math.exp(1j*tf.cast(i_gam_n_n0 * l1,dtype=dtype))
    else: ## seems L always larger than 0
        sn = tf.matmul(alpha, tf.ones((N, 1)))
    return sn

def min_max(N, J, K, alpha, beta, om, ft_flag): ## equation 25
    T = nufft_T( N,  J,  K,  alpha,  beta) # T =inv(CSSC)
    ###############################################################
    # formula 30  of Fessler's paper
    ###############################################################
    (r, arg) = nufft_r(om, N, J, K, alpha, beta)  # large N approx [J? M]
    ###############################################################
    # Min-max interpolator
    ###############################################################
    c = T.dot(r)
    u2 = OMEGA_u(c, N, K, om, arg, ft_flag)
    u2 = tf.math.conj(tf.transpose(u2))
    return u2

def nufft_T(N, J, K, alpha, beta):
    '''
     Equation (29) and (26) in Fessler and Sutton 2003.
     Create the overlapping matrix CSSC (diagonal dominant matrix)
     of J points, then find the pseudo-inverse of CSSC '''

#     import scipy.linalg
    L = alpha.shape[0] - 1
    cssc = tf.zeros((J, J))
    [j2, j1] = tf.meshgrid(range(1,J+1),range(1,J+1))
    overlapping_mat = j2 - j1
    for l1 in range(-L, L + 1):
        for l2 in range(-L, L + 1):
            alf1 = alpha[abs(l1)]
#             if l1 < 0: alf1 = numpy.conj(alf1)
            alf2 = alpha[abs(l2)]
#             if l2 < 0: alf2 = numpy.conj(alf2)
            tmp = overlapping_mat + beta * (l1 - l2)
            tmp = dirichlet(1.0 * tmp.numpy() / (1.0 * K / N))
            cssc = cssc + tf.cast(alf1 * alf2 * tmp, dtype=cssc.dtype)
    return mat_inv(cssc)

def nufft_r(om, N, J, K, alpha, beta): ## 30
    '''
    Equation (30) of Fessler & Sutton's paper
    '''
    def iterate_sum(rr, alf, r1):
        rr = rr + alf * r1
        return rr
    def iterate_l1(L, alpha, arg, beta, K, N, rr):
        oversample_ratio = (1.0 * K / N)
        for l1 in range(-L, L + 1):
            alf = alpha[abs(l1)] * 1.0
    #         if l1 < 0:
    #             alf = numpy.conj(alf)
        #             r1 = numpy.sinc(1.0*(arg+1.0*l1*beta)/(1.0*K/N))
            input_array = (arg + 1.0 * l1 * beta) / oversample_ratio
            r1 = dirichlet(input_array)
            rr = iterate_sum(rr, alf, r1)
        return rr
    M = tf.shape(om).numpy()[0]  # 1D size
    gam = 2.0 * math.pi / (K * 1.0)
    nufft_offset0 = nufft_offset(om, J, K)  # om/gam -  nufft_offset , [M,1]
    dk = 1.0 * om / gam - nufft_offset0  # om/gam -  nufft_offset , [M,1]
    arg = outer_sum(tf.range(-1.0, -(J + 1), -1), dk)
    L = tf.shape(alpha).numpy()[0] - 1
    rr = tf.zeros((J, M), dtype=tf.float32)
    rr = iterate_l1(L, alpha, arg, beta, K, N, rr)
    return (rr, arg)

def OMEGA_u(c, N, K, omd, arg, ft_flag):
    gam = 2.0 * math.pi / (K * 1.0)
    phase_scale = 1.0j * gam * (N*1.0 - 1.0) / 2.0
    phase = tf.math.exp(phase_scale * arg.numpy())  # [J? M] linear phase
    if ft_flag is True:
        u = phase * c
        phase0 = tf.math.exp( - 1.0j*omd*N/2.0)
        u = phase0 * u
    else:
        u = c
    return u


def outer_sum(xx, yy):
    xx = tf.cast(xx, dtype=yy.dtype)
    return tf.transpose([xx])+yy

def nufft_offset(om, J, K): ## equation 7
    '''
    For every om point (outside regular grids), find the nearest
    central grid (from Kd dimension)
    '''
    gam = 2.0 * math.pi / (K * 1.0)
    k0 = 1.0 * om / gam - 1.0 * J / 2.0  # new way
    k0 = tf.math.floor(k0)
    return k0


def dirichlet(x):
    # x = tf.where(tf.abs(x) < 1e-20, 1e-20 * tf.ones_like(x), x)
    # x = tf.sin(x) / x
    # x = tf.cast(x, dtype=tf.float32)
    x = np.sinc(x)
    return x

def mat_inv(A):
#     I = numpy.eye(A.shape[0], A.shape[1])
    B = scipy.linalg.pinv2(A)
    return B

def OMEGA_k(J,K, omd, Kd, dimid, dd, ft_flag):
    """
    Compute the index of k-space k_indx
    """
        # indices into oversampled FFT components
    # FORMULA 7
    M = omd.shape[0]
    koff = nufft_offset(omd, J, K)
    # FORMULA 9, find the indexes on Kd grids, of each M point
    if ft_flag is True: # tensor
        k_indx = outer_sum(tf.range(1.0, J + 1), koff)
        k_indx = tf.math.mod(k_indx, K)
    else:
        k_indx = tf.reshape(omd, (1, M))
        k_indx = tf.cast(k_indx, dtype=tf.int16)

    """
        JML: For GPU computing, indexing must be C-order (row-major)
        Multi-dimensional cuda or opencl arrays are row-major (order="C"), which  starts from the higher dimension.
        Note: This is different from the MATLAB indexing(for fortran order, colum major, low-dimension first
    """

    if dimid < dd - 1:  # trick: pre-convert these indices into offsets!
        #            ('trick: pre-convert these indices into offsets!')
        reduce_prod = tf.cast(tf.math.reduce_prod(Kd[dimid+1:dd]), dtype=k_indx.dtype)
        k_indx = k_indx * reduce_prod - 1
    """
    Note: F-order matrices must be reshaped into an 1D array before sparse matrix-vector multiplication.
    The original F-order (in Fessler and Sutton 2003) is not suitable for GPU array (C-order).
    Currently, in-place reshaping in F-order only works in numpy.
    """
    #             if dimid > 0:  # trick: pre-convert these indices into offsets!
    #                 #            ('trick: pre-convert these indices into offsets!')
    #                 kd[dimid] = kd[dimid] * numpy.prod(Kd[0:dimid]) - 1
    return k_indx

def full_kron(ud, kd, Jd, Kd, M):
    ud2, kd2, Jd2 = rdx_N(ud, kd, Jd)
    CSR  = create_csr(ud2[0], kd2[0], Kd, Jd, M) # must have
    return CSR

def rdx_N(ud, kd, Jd):
    ud2 = (khatri_rao_u(ud), )
    kd2 = (khatri_rao_k(kd), )
    Jd2 = (tf.reduce_prod(Jd), )
    return ud2, kd2, Jd2

def khatri_rao_k(kd):
    dd = len(kd)
    kk = kd[0]  # [M, J1] # pointers to indices
    M = kd[0].shape[0]
    Jprod = kd[0].shape[1]
    kk = kk.numpy()
    for dimid in range(1, dd):
        Jprod *= kd[dimid].shape[1] #numpy.prod(Jd[:dimid + 1])
        kk = block_outer_sum(kk, kd[dimid].numpy()) + 1  # outer sum of indices
        kk = kk.reshape((M, Jprod), order='C')
    return kk

def khatri_rao_u(ud):
    dd = len(ud)
    M = ud[0].shape[0]
    uu = ud[0]  # [M, J1]
    Jprod = ud[0].shape[1]
    for dimid in range(1, dd):
        Jprod *= ud[dimid].shape[1]#numpy.prod(Jd[:dimid + 1])
        uu = np.einsum('mi,mj->mij', uu, ud[dimid]) ## not understanding what is this line doing with 'mi,mj->mij'
        uu = uu.reshape((M, Jprod), order='C')
    return uu

def block_outer_sum(x1, x2):
    '''
    Update the new index after adding a new axis
    '''
    (M, J1) = x1.shape
    (M, J2) = x2.shape
    xx1 = x1.reshape((M, J1, 1), order='C')  # [J1 1 M] from [J1 M]
    xx2 = x2.reshape((M, 1, J2), order='C')  # [1 J2 M] from [J2 M]
    y = xx1 + xx2
    return y  # [J1 J2 M]

def create_csr(uu, kk, Kd, Jd, M):
    csrdata =uu.ravel(order='C')#numpy.reshape(uu.T, (Jprod * M, ), order='C')
    Jdprod = tf.math.reduce_prod(Jd)
    rowptr = tf.range(0, (M+1)*Jdprod, Jdprod)
    # colume indices, from 1 to prod(Kd), convert array to list
    colindx =kk.ravel(order='C')

    # The shape of sparse matrix
    csrshape = (M, tf.math.reduce_prod(Kd))

    # Build sparse matrix (interpolator)
#     csr = scipy.sparse.csr_matrix((csrdata, (rowindx, colindx)),
#                                        shape=csrshape)
    csr = scipy.sparse.csr_matrix((csrdata, colindx, rowptr),
                                  shape=csrshape)
    # csr = tf.sparse.SparseTensor(indices=tf.concat([colindx, rowptr], 1),
    #                        values=csrdata,
    #                        dense_shape=csrshape)
    return csr

def kronecker_scale(snd):
    """
    Compute the Kronecker product of the scaling factor.

    :param snd: Tuple of 1D scaling factors
    :param dd: Number of dimensions
    :type snd: tuple of 1D numpy.array
    :return: sn: The multi-dimensional Kronecker of the scaling factors
    :rtype: Nd array
    """
    dd = len(snd)
    shape_broadcasting = ()
    for dimid in range(0, dd):
        shape_broadcasting += (1, )
#     sn = numpy.array(1.0 + 0.0j)
    sn = tf.reshape(1.0, shape_broadcasting)
    sn = tf.cast(sn, dtype=snd[0].dtype)
    for dimid in range(0, dd):
        sn_shape = list(shape_broadcasting)
        sn_shape[dimid] = snd[dimid].shape[0]
        tmp = tf.reshape(snd[dimid], tuple(sn_shape))
#         print('tmp.shape = ', tmp.shape)
        ###############################################################
        # higher dimension implementation: multiply over all dimension
        ###############################################################
        sn = sn * tmp # multiply using broadcasting
    return sn


def preindex_copy(Nd, Kd):
    """
    Building the array index for copying two arrays of sizes Nd and Kd.
    Only the front parts of the input/output arrays are copied.
    The oversize  parts of the input array are truncated (if Nd > Kd),
    and the smaller size are zero-padded (if Nd < Kd)

    :param Nd: tuple, the dimensions of array1
    :param Kd: tuple, the dimensions of array2
    :type Nd: tuple with integer elements
    :type Kd: tuple with integer elements
    :return: inlist: the index of the input array
    :return: outlist: the index of the output array
    :return: nelem: the length of the inlist and outlist (equal length)
    :rtype: inlist: list with integer elements
    :rtype: outlist: list with integer elements
    :rtype: nelem: int
    """
    ndim = len(Nd)
    kdim = len(Kd)
    if ndim != kdim:
        print("mismatched dimensions!")
        print("Nd and Kd must have the same dimensions")
        raise
    else:
        nelem = 1
        min_dim = ()
        for pp in range(ndim - 1, -1,-1):
            YY = tf.math.minimum(Nd[pp], Kd[pp])
            nelem *= YY
            min_dim = (YY,) + min_dim
        mylist = tf.range(0, nelem)
        BB=()
        for pp in range(ndim - 1, 0, -1):
             a = tf.math.floor(mylist/min_dim[pp])
             b = mylist%min_dim[pp]
             mylist = tf.cast(a, tf.int32)
             BB=(b,) + BB

        if ndim == 1:
            mylist = tf.cast(tf.range(0, nelem), tf.int32)
        else:
            mylist = tf.math.floor(tf.cast(tf.range(0, nelem), tf.int32)/tf.reduce_prod(min_dim[1:]))
            mylist = tf.cast(mylist,tf.int32)

        inlist = mylist
        outlist = mylist
        for pp in range(0, ndim-1):
            inlist = inlist*Nd[pp+1] + BB[pp]
            outlist = outlist*Kd[pp+1] + BB[pp]

    return tf.cast(inlist, tf.int32), tf.cast(outlist, tf.int32), tf.cast(nelem, tf.int32)
