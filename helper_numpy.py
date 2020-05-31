"""
Helper functions
Mainly for implementing the equations in Fessler's paper
=======================================
"""

import numpy
dtype = numpy.complex64
import scipy


def OMEGA_u(c, N, K, omd, arg, ft_flag):
    gam = 2.0 * numpy.pi / (K * 1.0)

    phase_scale = 1.0j * gam * (N * 1.0 - 1.0) / 2.0
    phase = numpy.exp(phase_scale * arg)  # [J? M] linear phase

    if ft_flag is True:
        u = phase * c
        phase0 = numpy.exp(- 1.0j * omd * N / 2.0)
        u = phase0 * u

    else:
        u = c
    return u


def OMEGA_k(J, K, omd, Kd, dimid, dd, ft_flag):
    """
    Compute the index of k-space k_indx
    """
    # indices into oversampled FFT components
    # FORMULA 7
    M = omd.shape[0]
    koff = nufft_offset(omd, J, K)
    # FORMULA 9, find the indexes on Kd grids, of each M point
    if ft_flag is True:  # tensor
        k_indx = numpy.mod(
            outer_sum(
                numpy.arange(
                    1,
                    J + 1) * 1.0,
                koff),
            K)
    else:
        k_indx = numpy.reshape(omd, (1, M)).astype(numpy.int)

    """
        JML: For GPU computing, indexing must be C-order (row-major)
        Multi-dimensional cuda or opencl arrays are row-major (order="C"), which  starts from the higher dimension.
        Note: This is different from the MATLAB indexing(for fortran order, colum major, low-dimension first
    """

    if dimid < dd - 1:  # trick: pre-convert these indices into offsets!
        #            ('trick: pre-convert these indices into offsets!')
        k_indx = k_indx * numpy.prod(Kd[dimid + 1:dd]) - 1
    """
    Note: F-order matrices must be reshaped into an 1D array before sparse matrix-vector multiplication.
    The original F-order (in Fessler and Sutton 2003) is not suitable for GPU array (C-order).
    Currently, in-place reshaping in F-order only works in numpy.
    """
    return k_indx


def create_csr(uu, kk, Kd, Jd, M):
    csrdata = uu.ravel(order='C')  # numpy.reshape(uu.T, (Jprod * M, ), order='C')

    Jdprod = numpy.prod(Jd)
    rowptr = numpy.arange(0, (M + 1) * Jdprod, Jdprod)
    # colume indices, from 1 to prod(Kd), convert array to list
    colindx = kk.ravel(order='C')  # numpy.reshape(kk.T, (Jprod * M, ), order='C')

    # The shape of sparse matrix
    csrshape = (M, numpy.prod(Kd))

    # Build sparse matrix (interpolator)
    csr = scipy.sparse.csr_matrix((csrdata, colindx, rowptr),
                                  shape=csrshape)
    return csr


def rdx_N(ud, kd, Jd):
    ud2 = (khatri_rao_u(ud),)
    kd2 = (khatri_rao_k(kd),)
    Jd2 = (numpy.prod(Jd),)

    return ud2, kd2, Jd2


def full_kron(ud, kd, Jd, Kd, M):
    ud2, kd2, Jd2 = rdx_N(ud, kd, Jd)
    CSR = create_csr(ud2[0], kd2[0], Kd, Jd, M)  # must have
    return CSR


def khatri_rao_k(kd):
    dd = len(kd)

    kk = kd[0]  # [M, J1] # pointers to indices
    M = kd[0].shape[0]
    #     uu = ud[0]  # [M, J1]
    Jprod = kd[0].shape[1]
    for dimid in range(1, dd):
        Jprod *= kd[dimid].shape[1]  # numpy.prod(Jd[:dimid + 1])

        kk = block_outer_sum(kk, kd[dimid]) + 1  # outer sum of indices
        kk = kk.reshape((M, Jprod), order='C')

    return kk


def khatri_rao_u(ud):
    dd = len(ud)
    M = ud[0].shape[0]
    uu = ud[0]  # [M, J1]
    Jprod = ud[0].shape[1]
    for dimid in range(1, dd):
        Jprod *= ud[dimid].shape[1]  # numpy.prod(Jd[:dimid + 1])
        uu = numpy.einsum('mi,mj->mij', uu, ud[dimid])
        uu = uu.reshape((M, Jprod), order='C')

    return uu


def rdx_kron(ud, kd, Jd, radix=None):
    """
    Radix-n Kronecker product of multi-dimensional array
    :param ud: 1D interpolators
    :type ud: tuple of (M, Jd[d]) numpy.complex64 arrays
    :param kd: 1D indices to interpolators
    :type kd: tuple of (M, Jd[d]) numpy.uint arrays
    :param Jd: 1D interpolator sizes
    :type Jd: tuple of int
    :param radix: radix of Kronecker product
    :type radix: int
    :returns: uu: 1D interpolators
    :type uu: tuple of (M, Jd[d]) numpy.complex64 arrays
    :param kk: 1D indices to interpolators
    :type kk: tuple of (M, Jd[d]) numpy.uint arrays
    :param JJ: 1D interpolator sizes
    :type JJ: tuple of int
    """
    M = ud[0].shape[0]
    dd = len(Jd)
    if radix is None:
        radix = dd
    if radix > dd:
        radix = dd

    ud2 = ()
    kd2 = ()
    Jd2 = ()
    for count in range(0, int(numpy.ceil(dd / radix)), ):
        d_start = count * radix
        d_end = (count + 1) * radix
        if d_end > dd:
            d_end = dd
        ud3, kd3, Jd3 = rdx_N(ud[d_start:d_end], kd[d_start:d_end], Jd[d_start:d_end])
        ud2 += ud3
        kd2 += kd3
        Jd2 += Jd3
    return ud2, kd2, Jd2  # (uu, ), (kk, ), (Jprod, )#, Jprod


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
        shape_broadcasting += (1,)
    #     sn = numpy.array(1.0 + 0.0j)
    sn = numpy.reshape(1.0, shape_broadcasting)
    for dimid in range(0, dd):
        sn_shape = list(shape_broadcasting)
        sn_shape[dimid] = snd[dimid].shape[0]
        tmp = numpy.reshape(snd[dimid], tuple(sn_shape))
        #         print('tmp.shape = ', tmp.shape)
        ###############################################################
        # higher dimension implementation: multiply over all dimension
        ###############################################################
        sn = sn * tmp  # multiply using broadcasting
    return sn


def min_max(N, J, K, alpha, beta, om, ft_flag):
    T = nufft_T(N, J, K, alpha, beta)
    ###############################################################
    # formula 30  of Fessler's paper
    ###############################################################
    (r, arg) = nufft_r(om, N, J,
                       K, alpha, beta)  # large N approx [J? M]
    ###############################################################
    # Min-max interpolator
    ###############################################################
    c = T.dot(r)
    u2 = OMEGA_u(c, N, K, om, arg, ft_flag).T.conj()
    return u2


def plan(om, Nd, Kd, Jd, ft_axes=None, format='CSR', radix=None):
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

    #         self.debug = 0  # debug

    if type(Nd) != tuple:
        raise TypeError('Nd must be tuple, e.g. (256, 256)')

    if type(Kd) != tuple:
        raise TypeError('Kd must be tuple, e.g. (512, 512)')

    if type(Jd) != tuple:
        raise TypeError('Jd must be tuple, e.g. (6, 6)')

    if (len(Nd) != len(Kd)) | (len(Nd) != len(Jd)) | len(Kd) != len(Jd):
        raise KeyError('Nd, Kd, Jd must be in the same length, e.g. Nd=(256,256),Kd=(512,512),Jd=(6,6)')

    dd = numpy.size(Nd)

    if ft_axes is None:
        ft_axes = tuple(xx for xx in range(0, dd))

    #     print('ft_axes = ', ft_axes)
    ft_flag = ()  # tensor

    for pp in range(0, dd):
        if pp in ft_axes:
            ft_flag += (True,)
        else:
            ft_flag += (False,)
    #     print('ft_flag = ', ft_flag)
    ###############################################################
    # check input errors
    ###############################################################
    st = {}

    ###############################################################
    # First, get alpha and beta: the weighting and freq
    # of formula (28) in Fessler's paper
    # in order to create slow-varying image space scaling
    ###############################################################
    #     for dimid in range(0, dd):
    #         (tmp_alpha, tmp_beta) = nufft_alpha_kb_fit(
    #             Nd[dimid], Jd[dimid], Kd[dimid])
    #         st.setdefault('alpha', []).append(tmp_alpha)
    #         st.setdefault('beta', []).append(tmp_beta)
    st['tol'] = 0
    st['Jd'] = Jd
    st['Nd'] = Nd
    st['Kd'] = Kd
    M = om.shape[0]
    st['M'] = numpy.int32(M)
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

    snd = []
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

            #         ###############################################################
            #         # formula 30  of Fessler's paper
            #         ###############################################################

            #         ###############################################################
            #         # fast approximation to min-max interpolator
            #         ###############################################################

            #             c, arg = min_max(N, J, K, alpha, beta, om[:, dimid])
            #         ###############################################################
            #        # QR: a more accurate solution but slower than above fast approximation
            #        ###############################################################

            #             c, arg = QR_process(om[:,dimid], N, J, K, snd[dimid])

            #### phase shift
            #             ud += [QR2(om[:,dimid], N, J, K, snd[dimid], ft_flag[dimid]),]
            ud += [min_max(N, J, K, alpha, beta, om[:, dimid], ft_flag[dimid]), ]

        else:
            ud += [numpy.ones((1, M), dtype=dtype).T, ]

    """
    Now compute the column indices for 1D interpolators
    Each length-Jd interpolator includes Jd points, which are linked to Jd k-space locations
    kd is a tuple storing the 1D interpolators.
    A following Kronecker product will be needed.
    """
    kd = []
    for dimid in range(0, dd):  # iterate over all dimensions

        kd += [OMEGA_k(Jd[dimid], Kd[dimid], om[:, dimid], Kd, dimid, dd, ft_flag[dimid]).T, ]

    CSR = full_kron(ud, kd, Jd, Kd, M)
    st['p'] = CSR
    st['sn'] = kronecker_scale(snd).real  # only real scaling is relevant

    return st  # new


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
        for pp in range(ndim - 1, -1, -1):
            YY = numpy.minimum(Nd[pp], Kd[pp])
            nelem *= YY
            min_dim = (YY,) + min_dim
        mylist = numpy.arange(0, nelem).astype(numpy.int32)
        #             a=mylist
        BB = ()
        for pp in range(ndim - 1, 0, -1):
            a = numpy.floor(mylist / min_dim[pp])
            b = mylist % min_dim[pp]
            mylist = a
            BB = (b,) + BB

        if ndim == 1:
            mylist = numpy.arange(0, nelem).astype(numpy.int32)
        else:
            mylist = numpy.floor(numpy.arange(0, nelem).astype(numpy.int32) / numpy.prod(min_dim[1:]))

        inlist = mylist
        outlist = mylist
        for pp in range(0, ndim - 1):
            inlist = inlist * Nd[pp + 1] + BB[pp]
            outlist = outlist * Kd[pp + 1] + BB[pp]

    return inlist.astype(numpy.int32), outlist.astype(numpy.int32), nelem.astype(numpy.int32)


def dirichlet(x):
    return numpy.sinc(x)


def outer_sum(xx, yy):
    """
    Superseded by numpy.add.outer() function
    """

    return numpy.add.outer(xx, yy)


def nufft_offset(om, J, K):
    '''
    For every om point (outside regular grids), find the nearest
    central grid (from Kd dimension)
    '''
    gam = 2.0 * numpy.pi / (K * 1.0)
    k0 = numpy.floor(1.0 * om / gam - 1.0 * J / 2.0)  # new way
    return k0


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
        L = numpy.ceil(N / 3).astype(numpy.int16)

    nlist = numpy.arange(0, N) * 1.0 - Nmid

    if J > 1:
        (kb_a, kb_m) = kaiser_bessel('string', J, 'best', 0, K / N)
        sn_kaiser = 1 / kaiser_bessel_ft(nlist / K, J, kb_a, kb_m, 1.0)
    elif J == 1:  # The case when samples are on regular grids
        sn_kaiser = numpy.ones((1, N), dtype=dtype)
    gam = 2 * numpy.pi / K
    X_ant = beta * gam * nlist.reshape((N, 1), order='F')
    X_post = numpy.arange(0, L + 1)
    X_post = X_post.reshape((1, L + 1), order='F')
    X = numpy.dot(X_ant, X_post)  # [N,L]
    X = numpy.cos(X)
    sn_kaiser = sn_kaiser.reshape((N, 1), order='F').conj()
    X = numpy.array(X, dtype=dtype)
    sn_kaiser = numpy.array(sn_kaiser, dtype=dtype)
    coef = numpy.linalg.lstsq(numpy.nan_to_num(X), numpy.nan_to_num(sn_kaiser), rcond=-1)[0]
    alphas = coef
    if J > 1:
        alphas[0] = alphas[0]
        alphas[1:] = alphas[1:] / 2.0
    elif J == 1:  # cases on grids
        alphas[0] = 1.0
        alphas[1:] = 0.0
    alphas = numpy.real(alphas)
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
            min_ind = numpy.argmin(abs(tmp_key - J * numpy.ones(len(tmp_key))))
            p_J = tmp_key[min_ind]
            alpha = J * jlist_bestzn[p_J]
    kb_a = alpha
    return (kb_a, kb_m)


def kaiser_bessel_ft(u, J, alpha, kb_m, d):
    '''
    Interpolation weight for given J/alpha/kb-m
    '''

    u = u * (1.0 + 0.0j)
    import scipy.special
    z = numpy.sqrt((2 * numpy.pi * (J / 2) * u) ** 2.0 - alpha ** 2.0)
    nu = d / 2 + kb_m
    y = ((2 * numpy.pi) ** (d / 2)) * ((J / 2) ** d) * (alpha ** kb_m) / \
        scipy.special.iv(kb_m, alpha) * scipy.special.jv(nu, z) / (z ** nu)
    y = numpy.real(y)
    return y


def nufft_scale1(N, K, alpha, beta, Nmid):
    '''
    Calculate image space scaling factor
    '''
    alpha = numpy.real(alpha)

    L = len(alpha) - 1
    if L > 0:
        sn = numpy.zeros((N, 1))
        n = numpy.arange(0, N).reshape((N, 1), order='F')
        i_gam_n_n0 = 1j * (2 * numpy.pi / K) * (n - Nmid) * beta
        for l1 in range(-L, L + 1):
            alf = alpha[abs(l1)]
            if l1 < 0:
                alf = numpy.conj(alf)
            sn = sn + alf * numpy.exp(i_gam_n_n0 * l1)
    else:
        sn = numpy.dot(alpha, numpy.ones((N, 1)))
    return sn


def nufft_scale(Nd, Kd, alpha, beta):
    dd = numpy.size(Nd)
    Nmid = (Nd - 1) / 2.0
    if dd == 1:
        sn = nufft_scale1(Nd, Kd, alpha, beta, Nmid)
    else:
        sn = 1
        for dimid in numpy.arange(0, dd):
            tmp = nufft_scale1(Nd[dimid], Kd[dimid], alpha[dimid],
                               beta[dimid], Nmid[dimid])
            sn = numpy.dot(list(sn), tmp.H)
    return sn


def mat_inv(A):
    B = scipy.linalg.pinv2(A)
    return B


def nufft_T(N, J, K, alpha, beta):
    '''
     Equation (29) and (26) in Fessler and Sutton 2003.
     Create the overlapping matrix CSSC (diagonal dominant matrix)
     of J points, then find the pseudo-inverse of CSSC '''

    #     import scipy.linalg
    L = numpy.size(alpha) - 1
    cssc = numpy.zeros((J, J))
    [j1, j2] = numpy.mgrid[1:J + 1, 1:J + 1]
    overlapping_mat = j2 - j1
    for l1 in range(-L, L + 1):
        for l2 in range(-L, L + 1):
            alf1 = alpha[abs(l1)]
            alf2 = alpha[abs(l2)]
            tmp = overlapping_mat + beta * (l1 - l2)

            tmp = dirichlet(1.0 * tmp / (1.0 * K / N))
            cssc = cssc + alf1 * alf2 * tmp

    return mat_inv(cssc)


def nufft_r(om, N, J, K, alpha, beta):
    '''
    Equation (30) of Fessler & Sutton's paper
    '''

    def iterate_sum(rr, alf, r1):
        rr = rr + alf * r1
        return rr

    def iterate_l1(L, alpha, arg, beta, K, N, rr):
        oversample_ratio = (1.0 * K / N)
        import time
        t0 = time.time()
        for l1 in range(-L, L + 1):
            alf = alpha[abs(l1)] * 1.0
            input_array = (arg + 1.0 * l1 * beta) / oversample_ratio
            r1 = dirichlet(input_array)
            rr = iterate_sum(rr, alf, r1)
        return rr

    M = numpy.size(om)  # 1D size
    gam = 2.0 * numpy.pi / (K * 1.0)
    nufft_offset0 = nufft_offset(om, J, K)  # om/gam -  nufft_offset , [M,1]
    dk = 1.0 * om / gam - nufft_offset0  # om/gam -  nufft_offset , [M,1]
    arg = outer_sum(-numpy.arange(1, J + 1) * 1.0, dk)
    L = numpy.size(alpha) - 1
    rr = numpy.zeros((J, M), dtype=numpy.float32)
    rr = iterate_l1(L, alpha, arg, beta, K, N, rr)
    return (rr, arg)


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
