# Authors: Christian Thurau
# License: BSD 3 Clause
"""
PyMF base class and PyMF Semi Non-negative Matrix Factorization.

    SNMF(NMF) : Class for semi non-negative matrix factorization

[1] Ding, C., Li, T. and Jordan, M.. Convex and Semi-Nonnegative Matrix Factorizations.
IEEE Trans. on Pattern Analysis and Machine Intelligence 32(1), 45-55.

"""
import numpy as np
import logging
import logging.config
import scipy.sparse
from numpy.linalg import eigh
from scipy.special import factorial

_EPS = np.finfo(float).eps


def eighk(M, k=0):
    """ Returns ordered eigenvectors of a squared matrix. Too low eigenvectors
    are ignored. Optionally only the first k vectors/values are returned.

    Arguments
    ---------
    M - squared matrix
    k - (default 0): number of eigenvectors/values to return

    Returns
    -------
    w : [:k] eigenvalues
    v : [:k] eigenvectors

    """
    values, vectors = eigh(M)

    # get rid of too low eigenvalues
    s = np.where(values > _EPS)[0]
    vectors = vectors[:, s]
    values = values[s]

    # sort eigenvectors according to largest value
    idx = np.argsort(values)[::-1]
    values = values[idx]
    vectors = vectors[:, idx]

    # select only the top k eigenvectors
    if k > 0:
        values = values[:k]
        vectors = vectors[:, :k]

    return values, vectors


def cmdet(d):
    """ Returns the Volume of a simplex computed via the Cayley-Menger
    determinant.

    Arguments
    ---------
    d - euclidean distance matrix (shouldn't be squared)

    Returns
    -------
    V - volume of the simplex given by d
    """
    D = np.ones((d.shape[0] + 1, d.shape[0] + 1))
    D[0, 0] = 0.0
    D[1:, 1:] = d ** 2
    j = np.float32(D.shape[0] - 2)
    f1 = (-1.0) ** (j + 1) / ((2 ** j) * ((factorial(j)) ** 2))
    cmd = f1 * np.linalg.det(D)

    # sometimes, for very small values, "cmd" might be negative, thus we take
    # the absolute value
    return np.sqrt(np.abs(cmd))


def simplex(d):
    """ Computed the volume of a simplex S given by a coordinate matrix D.

    Arguments
    ---------
    d - coordinate matrix (k x n, n samples in k dimensions)

    Returns
    -------
    V - volume of the Simplex spanned by d
    """
    # compute the simplex volume using coordinates
    D = np.ones((d.shape[0] + 1, d.shape[1]))
    D[1:, :] = d
    V = np.abs(np.linalg.det(D)) / factorial(d.shape[1] - 1)
    return V


class PyMFBase:
    """
    PyMF Base Class. Does nothing useful apart from providing some basic methods.
    """
    # some small value

    _EPS = _EPS

    def __init__(self, data, num_bases=4, **kwargs):
        """
        """

        # create logger
        self._logger = logging.getLogger("pymf")

        # add ch to logger
        if len(self._logger.handlers) < 1:
            # create console handler and set level to debug
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            # create formatter
            formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

            # add formatter to ch
            ch.setFormatter(formatter)

            self._logger.addHandler(ch)

        # set variables
        self.data = data
        self._num_bases = num_bases

        # initialize H and W to random values
        self._data_dimension, self._num_samples = self.data.shape

    def residual(self):
        """ Returns the residual in % of the total amount of data

        Returns
        -------
        residual : float
        """
        res = np.sum(np.abs(self.data - np.dot(self.W, self.H)))
        total = 100.0 * res / np.sum(np.abs(self.data))
        return total

    def frobenius_norm(self):
        """ Frobenius norm (||data - WH||) of a data matrix and a low rank
        approximation given by WH. Minimizing the Fnorm ist the most common
        optimization criterion for matrix factorization methods.

        Returns:
        -------
        frobenius norm: F = ||data - WH||

        """
        # check if W and H exist
        if hasattr(self, 'H') and hasattr(self, 'W'):
            if scipy.sparse.issparse(self.data):
                tmp = self.data[:, :] - (self.W * self.H)
                tmp = tmp.multiply(tmp).sum()
                err = np.sqrt(tmp)
            else:
                err = np.sqrt(np.sum((self.data[:, :] - np.dot(self.W, self.H)) ** 2))
        else:
            err = None

        return err

    def _init_w(self):
        """ Initialize W to random values [0,1].
        """
        # add a small value, otherwise nmf and related methods get into trouble as
        # they have difficulties recovering from zero.
        self.W = np.random.random((self._data_dimension, self._num_bases)) + 10 ** -4

    def _init_h(self):
        """ Initialize H to random values [0,1].
        """
        self.H = np.random.random((self._num_bases, self._num_samples)) + 10 ** -4

    def _update_h(self):
        """ Overwrite for updating H.
        """
        pass

    def _update_w(self):
        """ Overwrite for updating W.
        """
        pass

    def _converged(self, i):
        """
        If the optimization of the approximation is below the machine precision,
        return True.

        Parameters
        ----------
            i   : index of the update step

        Returns
        -------
            converged : boolean
        """
        derr = np.abs(self.ferr[i] - self.ferr[i - 1]) / self._num_samples
        if derr < self._EPS:
            return True
        else:
            return False

    def factorize(self, niter=100, show_progress=False,
                  compute_w=True, compute_h=True, compute_err=True):
        """ Factorize s.t. WH = data

        Parameters
        ----------
        niter : int
                number of iterations.
        show_progress : bool
                print some extra information to stdout.
        compute_h : bool
                iteratively update values for H.
        compute_w : bool
                iteratively update values for W.
        compute_err : bool
                compute Frobenius norm |data-WH| after each update and store
                it to .ferr[k].

        Updated Values
        --------------
        .W : updated values for W.
        .H : updated values for H.
        .ferr : Frobenius norm |data-WH| for each iteration.
        """

        if show_progress:
            self._logger.setLevel(logging.INFO)
        else:
            self._logger.setLevel(logging.ERROR)

            # create W and H if they don't already exist
        # -> any custom initialization to W,H should be done before
        if not hasattr(self, 'W') and compute_w:
            self._init_w()

        if not hasattr(self, 'H') and compute_h:
            self._init_h()

            # Computation of the error can take quite long for large matrices,
        # thus we make it optional.
        if compute_err:
            self.ferr = np.zeros(niter)

        for i in range(niter):
            if compute_w:
                self._update_w()

            if compute_h:
                self._update_h()

            if compute_err:
                self.ferr[i] = self.frobenius_norm()
                self._logger.info('FN: %s (%s/%s)' % (self.ferr[i], i + 1, niter))
            else:
                self._logger.info('Iteration: (%s/%s)' % (i + 1, niter))

            # check if the err is not changing anymore
            if i > 1 and compute_err:
                if self._converged(i):
                    # adjust the error measure
                    self.ferr = self.ferr[:i]
                    break


class SNMF(PyMFBase):
    """
    SNMF(data, num_bases=4)

    Semi Non-negative Matrix Factorization. Factorize a data matrix into two
    matrices s.t. F = | data - W*H | is minimal. For Semi-NMF only H is
    constrained to non-negativity.

    Parameters
    ----------
    data : array_like, shape (_data_dimension, _num_samples)
        the input data
    num_bases: int, optional
        Number of bases to compute (column rank of W and row rank of H).
        4 (default)

    Attributes
    ----------
    W : "data_dimension x num_bases" matrix of basis vectors
    H : "num bases x num_samples" matrix of coefficients
    ferr : frobenius norm (after calling .factorize())

    Example
    -------
    Applying Semi-NMF to some rather stupid data set:

    >>> import numpy as np
    >>> data = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
    >>> snmf_mdl = SNMF(data, num_bases=2)
    >>> snmf_mdl.factorize(niter=10)

    The basis vectors are now stored in snmf_mdl.W, the coefficients in snmf_mdl.H.
    To compute coefficients for an existing set of basis vectors simply    copy W
    to snmf_mdl.W, and set compute_w to False:

    >>> data = np.array([[1.5], [1.2]])
    >>> W = np.array([[1.0, 0.0], [0.0, 1.0]])
    >>> snmf_mdl = SNMF(data, num_bases=2)
    >>> snmf_mdl.W = W
    >>> snmf_mdl.factorize(niter=1, compute_w=False)

    The result is a set of coefficients snmf_mdl.H, s.t. data = W * snmf_mdl.H.
    """

    def _update_w(self):
        # In Ding et al. (2008) this is F = XG(G^TG)^-1
        W1 = np.dot(self.data[:, :], self.H.T)
        W2 = np.dot(self.H, self.H.T)
        self.W = np.dot(W1, np.linalg.inv(W2))

    def _update_h(self):
        # Corresponds to the update of G while fixing F in Ding et al. (2008)
        def separate_positive(m):
            return (np.abs(m) + m) / 2.0

        def separate_negative(m):
            return (np.abs(m) - m) / 2.0

        XW = np.dot(self.data[:, :].T, self.W)

        WW = np.dot(self.W.T, self.W)
        WW_pos = separate_positive(WW)
        WW_neg = separate_negative(WW)

        XW_pos = separate_positive(XW)
        H1 = (XW_pos + np.dot(self.H.T, WW_neg)).T

        XW_neg = separate_negative(XW)
        H2 = (XW_neg + np.dot(self.H.T, WW_pos)).T + 10 ** -9

        self.H *= np.sqrt(H1 / H2)


def _test():
    import doctest
    doctest.testmod()


if __name__ == "__main__":
    _test()
