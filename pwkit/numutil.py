# -*- mode: python; coding: utf-8 -*-
# Copyright 2014 Peter Williams <peter@newton.cx> and collaborators.
# Licensed under the MIT License.

"""numutil - NumPy and generic numerical utilities.

Decorators:

broadcastize - Make a Python function automatically broadcast arguments.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = (b'broadcastize').split ()

import functools
import numpy as np


class _Broadcaster (object):
    def __init__ (self, n_arr, subfunc):
        self._subfunc = subfunc
        self._n_arr = int (n_arr)

        if self._n_arr < 1:
            raise ValueError ('broadcastiz\'ed function must take at least 1 '
                              'array argument')

        functools.update_wrapper (self, subfunc)


    def __call__ (self, *args, **kwargs):
        if len (args) < self._n_arr:
            raise TypeError ('expected at least %d arguments, got %d'
                             % (self._n_arr, len (args)))

        bc_raw = np.broadcast_arrays (*args[:self._n_arr])
        bc_1d = tuple (np.atleast_1d (a) for a in bc_raw)
        rest = args[self._n_arr:]
        result = self._subfunc (*(bc_1d + rest), **kwargs)

        if not len (bc_raw[0].shape):
            return np.asscalar (result)
        return result


class _BroadcasterDecorator (object):
    """Decorator to make functions automatically work on vectorized arguments.

    @broadcastize (3) # myfunc's first 3 arguments should be arrays.
    def myfunc (arr1, arr2, arr3, non_vec_arg1, non_vec_arg2):
        ...

    This decorator makes it so that the child function's arguments are assured
    to be Numpy arrays of at least 1 dimension, and all having the same shape.
    The child can then perform vectorized computations without having to
    special-case scalar-vs.-vector possibilities or worry about manual
    broadcasting. If the inputs to the function are indeed all scalars, the output
    is converted back to scalar upon return.

    Therefore, for the caller, the function appears to have magic broadcasting
    rules equivalent to a Numpy ufunc. Meanwhile the implementer can get
    broadcasting behavior without having to special case the actual inputs.

    """
    def __init__ (self, n_arr):
        """Decorator for making a auto-broadcasting function. Arguments:

        n_arr - The number of array arguments accepted by the decorated function.
                These arguments come at the beginning of the argument list.

        """
        self._n_arr = int (n_arr)
        if self._n_arr < 1:
            raise ValueError ('broadcastiz\'ed function must take at least 1 '
                              'array argument')

    def __call__ (self, subfunc):
        return _Broadcaster (self._n_arr, subfunc)


broadcastize = _BroadcasterDecorator
