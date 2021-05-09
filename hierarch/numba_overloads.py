import numpy as np
from numba import types
from numba.extending import overload, register_jitable
from numba.core.errors import TypingError


@overload(np.all)
def np_all(x, axis=None):

    # ndarray.all with axis arguments for 2D arrays.

    @register_jitable
    def _np_all_axis0(arr):
        out = np.logical_and(arr[0], arr[1])
        for v in iter(arr[2:]):
            for idx, v_2 in enumerate(v):
                out[idx] = np.logical_and(v_2, out[idx])
        return out

    @register_jitable
    def _np_all_flat(x):
        out = x.all()
        return out

    @register_jitable
    def _np_all_axis1(arr):
        out = np.logical_and(arr[:, 0], arr[:, 1])
        for idx, v in enumerate(arr[:, 2:]):
            for v_2 in iter(v):
                out[idx] = np.logical_and(v_2, out[idx])
        return out

    if isinstance(axis, types.Optional):
        axis = axis.type

    if not isinstance(axis, (types.Integer, types.NoneType)):
        raise TypingError("'axis' must be 0, 1, or None")

    if not isinstance(x, types.Array):
        raise TypingError('Only accepts NumPy ndarray')

    if not (1 <= x.ndim <= 2):
        raise TypingError('Only supports 1D or 2D NumPy ndarrays')

    if isinstance(axis, types.NoneType):
        def _np_all_impl(x, axis=None):
            return _np_all_flat(x)
        return _np_all_impl

    elif x.ndim == 1:
        def _np_all_impl(x, axis=None):
            return _np_all_flat(x)
        return _np_all_impl

    elif x.ndim == 2:
        def _np_all_impl(x, axis=None):
            if axis == 0:
                return _np_all_axis0(x)
            else:
                return _np_all_axis1(x)
        return _np_all_impl

    else:
        def _np_all_impl(x, axis=None):
            return _np_all_flat(x)
        return _np_all_impl


@overload(np.random.dirichlet)
def dirichlet(alpha, size=None):

    dtype = np.float64

    @register_jitable
    def dirichlet_arr(alpha, out):

        # Gamma distribution method to generate a Dirichlet distribution

        a_len = len(alpha)
        size = out.size
        flat = out.flat
        for i in range(0, size, a_len):
            exp = np.zeros(len(alpha), dtype=np.float64)
            # calculate gamma random numbers per alpha specifications
            for k, w in enumerate(alpha):
                exp[k] = np.random.gamma(w, 1)
            # normalize sum to 1
            exp /= exp.sum()
            for j, v in enumerate(exp):
                flat[i+j] = v.item()

    if not isinstance(alpha, (types.Sequence, types.Array)):
        raise TypeError("np.random.dirichlet(): alpha should be an "
                        "array or sequence, got %s" % (alpha,))

    if not isinstance(alpha, types.Array):
        alpha = np.array(alpha)

    if size in (None, types.none):
        def dirichlet_impl(alpha, size=None):
            out = np.zeros(len(alpha), dtype)
            dirichlet_arr(alpha, out)
            return out

    elif isinstance(size, types.Integer):
        def dirichlet_impl(alpha, size=None):
            """
            dirichlet(..., size=int)
            """
            out = np.zeros((size, len(alpha)), dtype)
            dirichlet_arr(alpha, out)
            return out

    elif isinstance(size, types.BaseTuple):
        def dirichlet_impl(alpha, size=None):
            """
            dirichlet(..., size=tuple)
            """
            out = np.zeros(size + (len(alpha),), dtype)
            dirichlet_arr(alpha, out)
            return out

    else:
        raise TypeError("np.random.dirichlet(): size should be int or "
                        "tuple or None, got %s" % (size,))

    return dirichlet_impl
