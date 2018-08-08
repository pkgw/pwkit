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


# Some diagnostics of the low-level code.

def make_figure9_plot(fk10func, set_unused=True):
    """Reproduce Figure 9 of the Fleischman & Kuznetsov (2010) paper, using our
    low-level interfaces. Uses OmegaPlot, of course.

    Input parameters, etc., come from the file ``Flare071231a.pro`` that is
    distributed with the paper’s Supplementary Data archive.

    Invoke with something like::

      from pwkit import fk10
      func = fk10.FK10Invoker('path/to/libGS_Std_HomSrc_CEH.so.64')
      fk10.make_figure9_plot(func).show()

    """
    import omega as om

    in_vals = make_in_vals_array()
    in_vals[IN_VAL_AREA] = 1.33e18
    in_vals[IN_VAL_DEPTH] = 6e8
    in_vals[IN_VAL_T0] = 2.1e7
    in_vals[IN_VAL_INTEG_METH] = 16
    in_vals[IN_VAL_EMIN] = 0.016
    in_vals[IN_VAL_EMAX] = 4.0
    in_vals[IN_VAL_DELTA1] = 3.7
    in_vals[IN_VAL_N0] = 3e9
    in_vals[IN_VAL_NB] = 5e9 * (1e9 / in_vals[IN_VAL_N0])
    in_vals[IN_VAL_B] = 48
    in_vals[IN_VAL_THETA] = 50
    in_vals[IN_VAL_FREQ0] = 5e8
    in_vals[IN_VAL_LOGDFREQ] = 0.02
    in_vals[IN_VAL_EDIST] = EDIST_PLW
    in_vals[IN_VAL_NFREQ] = 100
    in_vals[IN_VAL_PADIST] = PADIST_GLC
    in_vals[IN_VAL_LCBDY] = 90
    in_vals[IN_VAL_DELTAMU] = 0.4
    in_vals[IN_VAL_FCCR] = 12
    in_vals[IN_VAL_FWHCR] = in_vals[IN_VAL_FCCR]
    in_vals[IN_VAL_RENORMFLAG] = 1
    in_vals[IN_VAL_QFLAG] = 2

    if set_unused:
        # Sanity-checking: these parameters shouldn't affect the calculated
        # result.
        in_vals[IN_VAL_EPSILON] = 0.05
        in_vals[IN_VAL_KAPPA] = 4.0
        in_vals[IN_VAL_EBREAK] = 1.0
        in_vals[IN_VAL_DELTA2] = 6.0
        in_vals[IN_VAL_BEAMDIR] = 90
        in_vals[IN_VAL_A4] = 1

    out_vals = fk10func(in_vals)
    freqs = out_vals[:,OUT_VAL_FREQ]
    tot_ints = out_vals[:,OUT_VAL_OINT] + out_vals[:,OUT_VAL_XINT]
    pos = (tot_ints > 0)

    p = om.quickXY(freqs[pos], tot_ints[pos], 'Calculation', xlog=1, ylog=1)

    nu_obs = np.array([1.0, 2.0, 3.75, 9.4, 17.0, 34.0])
    int_obs = np.array([12.0, 43.0, 29.0, 6.3, 1.7, 0.5])
    p.addXY(nu_obs, int_obs, 'Observations', lines=False)

    p.defaultKeyOverlay.hAlign = 0.93
    p.setBounds(0.5, 47, 0.1, 60)
    p.setLabels('Emission frequency, GHz', 'Total intensity, sfu')
    return p
