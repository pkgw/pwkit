# -*- mode: python; coding: utf-8 -*-
# Copyright 2018 Peter Williams <peter@newton.cx> and collaborators.
# Licensed under the MIT License.

"""This module helps you run the synchrotron code described in Fleischman &
Kuznetsov (2010) [`DOI:10.1088/0004-637X/721/2/1127
<https://doi.org/10.1088/0004-637X/721/2/1127>`_]. The code is provided as a
precompiled binary module. It’s meant to be called from IDL, but we won’t let
that stop us!

Installing the code
-------------------

To do anything useful with this module, you must first obtain the precompiled
module. This isn't the sort of module you’d want to install into your system
shared libraries, so for applications you’ll probably be storing it in some
random directory. Therefore, all actions in this module start by specifying
the path to the library.

The module can be downloaded from as part of a Supplementary Data archive
attached to the journal paper. At the moment, the direct link is `here
<http://iopscience.iop.org/0004-637X/721/2/1127/suppdata/apj351391_sourcecode.tar.gz>`_,
but that might change over time. The `journal’s website for the paper
<https://doi.org/10.1088/0004-637X/721/2/1127>`_ should always have a link.

The archive contains compiled versions of the code for Windows, 32-bit Linux,
and 64-bit Linux. It is quite worrisome that maybe one day these files will
stop working, but that’s what we’ve got right now.

Anyway, you should download and unpack the archive and copy the desired file
to wherever makes the most sense for your software environment and
application. On 64-bit Linux, the file name is ``libGS_Std_HomSrc_CEH.so.64``.
Any variable named *shlib_path* that comes up in the API should be a path to
this file. Note that relative paths should include a directory component (e.g.
``./libGS_Std_HomSrc_CEH.so.64``); the :mod:`ctypes` module treats bare
filenames specially.

"""
from __future__ import absolute_import, division, print_function

__all__ = '''
'''

import ctypes
import numpy as np


# Here we have the very low-level interface to the compiled code. These aren't
# documented in the main documentation since, hopefully, regular users will
# never need to use it.

pointer_pair = ctypes.c_void_p * 2

IN_VAL_AREA = 0
IN_VAL_DEPTH = 1
IN_VAL_T0 = 2
IN_VAL_EPSILON = 3
IN_VAL_KAPPA = 4
IN_VAL_INTEG_METH = 5
IN_VAL_EMIN = 6
IN_VAL_EMAX = 7
IN_VAL_EBREAK = 8
IN_VAL_DELTA1 = 9
IN_VAL_DELTA2 = 10
IN_VAL_N0 = 11
IN_VAL_NB = 12
IN_VAL_B = 13
IN_VAL_THETA = 14
IN_VAL_FREQ0 = 15
IN_VAL_LOGDFREQ = 16
IN_VAL_EDIST = 17
IN_VAL_NFREQ = 18
IN_VAL_PADIST = 19
IN_VAL_LCBDY = 20
IN_VAL_BEAMDIR = 21
IN_VAL_DELTAMU = 22
IN_VAL_A4 = 23
# 24 is unused
# "CEH" library only:
IN_VAL_FCCR = 25
IN_VAL_FWHCR = 26
IN_VAL_RENORMFLAG = 27
# "C" library only:
IN_VAL_QFLAG = 28

EDIST_THM = 2
EDIST_PLW = 3
EDIST_DPL = 4
EDIST_TNT = 5
EDIST_KAP = 6
EDIST_PLP = 7
EDIST_PLG = 8
EDIST_TNP = 9
EDIST_TNG = 10

PADIST_ISO = 1
PADIST_ELC = 2
PADIST_GLC = 3
PADIST_GAU = 4
PADIST_SGA = 5

OUT_VAL_FREQ = 0
OUT_VAL_OINT = 1
OUT_VAL_ODAMP = 2
OUT_VAL_XINT = 3
OUT_VAL_XDAMP = 4


class FK10Invoker(object):
    "The lowest-level interface to the FK10 code."

    def __init__(self, shlib_path):
        self.shlib_path = shlib_path
        self.shlib = ctypes.CDLL(self.shlib_path)

        self.get_mw = self.shlib.GET_MW
        self.get_mw.restype = ctypes.c_int
        self.get_mw.argtypes = (ctypes.c_int, ctypes.POINTER(pointer_pair))


    def __call__(self, in_values, out_values=None):
        if not isinstance(in_values, np.ndarray):
            raise ValueError('in_values must be an ndarray')
        if not in_values.flags.c_contiguous:
            raise ValueError('in_values must be C-contiguous')
        if in_values.dtype != np.float32:
            raise ValueError('in_values must have the C-float dtype')
        if in_values.shape != (29,):
            raise ValueError('in_values must have a shape of (29,)')

        n_freqs = int(in_values[IN_VAL_NFREQ])

        if out_values is None:
            out_values = np.empty((n_freqs, 5), dtype=np.float32)
            out_values.fill(np.nan)
        else:
            if not isinstance(out_values, np.ndarray):
                raise ValueError('out_values must be an ndarray')
            if not out_values.flags.c_contiguous:
                raise ValueError('out_values must be C-contiguous')
            if not out_values.flags.writeable:
                raise ValueError('out_values must be writeable')
            if out_values.dtype != np.float32:
                raise ValueError('out_values must have the C-float dtype')
            if out_values.shape != (n_freqs, 5):
                raise ValueError('out_values must have a shape of ({},5), where the first '
                                 'dimension comes from in_values'.format(n_freqs))

        in_ptr = in_values.ctypes.data_as(ctypes.c_void_p)
        out_ptr = out_values.ctypes.data_as(ctypes.c_void_p)
        res = self.get_mw(2, pointer_pair(in_ptr, out_ptr))

        if res != 0:
            raise RuntimeError('bad inputs to GET_MW function; return code was {}'.format(res))

        return out_values


def make_in_vals_array():
    return np.zeros(29, dtype=np.float32)
