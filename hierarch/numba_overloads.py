import numpy as np
from numba import types
from numba.extending import overload, register_jitable


@overload(np.random.dirichlet)
def dirichlet(alpha, size=None):
    @register_jitable
    def dirichlet_arr(alpha, out):

        # Gamma distribution method to generate a Dirichlet distribution

        for a_val in iter(alpha):
            if a_val <= 0:
                raise ValueError("dirichlet: alpha must be > 0.0")

        a_len = len(alpha)
        size = out.size
        flat = out.flat
        for i in range(0, size, a_len):
            # calculate gamma random numbers per alpha specifications
            norm = 0  # use this to normalize every the group total to 1
            for k, w in enumerate(alpha):
                flat[i + k] = np.random.gamma(w, 1)
                norm += flat[i + k].item()
            for k, w in enumerate(alpha):
                flat[i + k] /= norm

    if not isinstance(alpha, (types.Sequence, types.Array)):
        raise TypeError(
            "np.random.dirichlet(): alpha should be an "
            "array or sequence, got %s" % (alpha,)
        )

    if size in (None, types.none):

        def dirichlet_impl(alpha, size=None):
            out = np.empty(len(alpha))
            dirichlet_arr(alpha, out)
            return out

    elif isinstance(size, types.Integer):

        def dirichlet_impl(alpha, size=None):
            """
            dirichlet(..., size=int)
            """
            out = np.empty((size, len(alpha)))
            dirichlet_arr(alpha, out)
            return out

    elif isinstance(size, (types.UniTuple)) and isinstance(size.dtype, types.Integer):

        def dirichlet_impl(alpha, size=None):
            """
            dirichlet(..., size=tuple)
            """
            out = np.empty(size + (len(alpha),))
            dirichlet_arr(alpha, out)
            return out

    else:
        raise TypeError(
            "np.random.dirichlet(): size should be int or "
            "tuple of ints or None, got %s" % size
        )

    return dirichlet_impl
