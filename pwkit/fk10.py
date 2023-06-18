# -*- mode: python; coding: utf-8 -*-
# Copyright 2018 Peter Williams <peter@newton.cx> and collaborators.
# Licensed under the MIT License.

"""This module helps you run the synchrotron code described in Fleischman &
Kuznetsov (2010; hereafter FK10) [`DOI:10.1088/0004-637X/721/2/1127
<https://doi.org/10.1088/0004-637X/721/2/1127>`_]. The code is provided as a
precompiled binary module. It’s meant to be called from IDL, but we won’t let
that stop us!

The main interface to the code is the :class:`Calculator` class. But before
you can use it, you must install the code, as described below.


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

__all__ = """
Calculator
""".split()

import ctypes
import numpy as np

from . import cgs


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
# all libraries again:
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

E0_MEV = (
    cgs.me * cgs.c**2 * cgs.evpererg * 1e-6
)  # helpful: electron rest-mass-energy in MeV (~= 0.511)


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
            raise ValueError("in_values must be an ndarray")
        if not in_values.flags.c_contiguous:
            raise ValueError("in_values must be C-contiguous")
        if in_values.dtype != np.float32:
            raise ValueError("in_values must have the C-float dtype")
        if in_values.shape != (29,):
            raise ValueError("in_values must have a shape of (29,)")

        n_freqs = int(in_values[IN_VAL_NFREQ])

        if out_values is None:
            out_values = np.empty((n_freqs, 5), dtype=np.float32)
            out_values.fill(np.nan)
        else:
            if not isinstance(out_values, np.ndarray):
                raise ValueError("out_values must be an ndarray")
            if not out_values.flags.c_contiguous:
                raise ValueError("out_values must be C-contiguous")
            if not out_values.flags.writeable:
                raise ValueError("out_values must be writeable")
            if out_values.dtype != np.float32:
                raise ValueError("out_values must have the C-float dtype")
            if out_values.shape != (n_freqs, 5):
                raise ValueError(
                    "out_values must have a shape of ({},5), where the first "
                    "dimension comes from in_values".format(n_freqs)
                )

        in_ptr = in_values.ctypes.data_as(ctypes.c_void_p)
        out_ptr = out_values.ctypes.data_as(ctypes.c_void_p)
        res = self.get_mw(2, pointer_pair(in_ptr, out_ptr))

        if res != 0:
            raise Exception(
                "bad inputs to GET_MW function; return code was {}".format(res)
            )

        return out_values


def make_in_vals_array():
    return np.zeros(29, dtype=np.float32)


# Some diagnostics of the low-level code.


def do_figure9_calc_lowlevel(shlib_path, set_unused=True):
    """Reproduce the calculation used to produce Figure 9 of the Fleischman &
    Kuznetsov (2010) paper, using our low-level interfaces.

    Input parameters, etc., come from the file ``Flare071231a.pro`` that is
    distributed with the paper’s Supplementary Data archive.

    Invoke with something like::

      from pwkit import fk10
      arr = fk10.do_figure9_calc('path/to/libGS_Std_HomSrc_CEH.so.64')

    """
    fk10func = FK10Invoker(shlib_path)

    in_vals = make_in_vals_array()
    in_vals[IN_VAL_AREA] = 1.33e18
    in_vals[IN_VAL_DEPTH] = 6e8
    in_vals[IN_VAL_T0] = 2.1e7
    # EPSILON (these markers are to aid counting indices)
    # KAPPA
    in_vals[IN_VAL_INTEG_METH] = 16
    in_vals[IN_VAL_EMIN] = 0.016
    in_vals[IN_VAL_EMAX] = 4.0
    # EBREAK
    in_vals[IN_VAL_DELTA1] = 3.7
    # DELTA2
    in_vals[IN_VAL_N0] = 3e9
    in_vals[IN_VAL_NB] = 5e9 / 3
    in_vals[IN_VAL_B] = 48
    in_vals[IN_VAL_THETA] = 50
    in_vals[IN_VAL_FREQ0] = 5e8
    in_vals[IN_VAL_LOGDFREQ] = 0.02
    in_vals[IN_VAL_EDIST] = EDIST_PLW
    in_vals[IN_VAL_NFREQ] = 100
    in_vals[IN_VAL_PADIST] = PADIST_GLC
    in_vals[IN_VAL_LCBDY] = 90
    # BEAMDIR
    in_vals[IN_VAL_DELTAMU] = 0.4
    # A4
    # (slot 24 unused)
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

    return fk10func(in_vals)


def do_figure9_calc_highlevel(shlib_path):
    """Reproduce the calculation used to produce Figure 9 of the Fleischman &
    Kuznetsov (2010) paper, using our high-level interfaces.

    The meat is moved to a class method on Calculator since it's handy to have
    a quick way to create an instance that has reasonable settings for all of
    its parameters.

    """
    return Calculator.new_for_fk10_fig9(shlib_path).compute_lowlevel()


def make_figure9_plot(shlib_path, use_lowlevel=True, **kwargs):
    """Reproduce Figure 9 of the Fleischman & Kuznetsov (2010) paper, using our
    low-level interfaces. Uses OmegaPlot, of course.

    Input parameters, etc., come from the file ``Flare071231a.pro`` that is
    distributed with the paper’s Supplementary Data archive.

    Invoke with something like::

      from pwkit import fk10
      fk10.make_figure9_plot('path/to/libGS_Std_HomSrc_CEH.so.64').show()

    """
    import omega as om

    if use_lowlevel:
        out_vals = do_figure9_calc_lowlevel(shlib_path, **kwargs)
    else:
        out_vals = do_figure9_calc_highlevel(shlib_path, **kwargs)

    freqs = out_vals[:, OUT_VAL_FREQ]
    tot_ints = out_vals[:, OUT_VAL_OINT] + out_vals[:, OUT_VAL_XINT]
    pos = tot_ints > 0

    p = om.quickXY(freqs[pos], tot_ints[pos], "Calculation", xlog=1, ylog=1)

    nu_obs = np.array([1.0, 2.0, 3.75, 9.4, 17.0, 34.0])
    int_obs = np.array([12.0, 43.0, 29.0, 6.3, 1.7, 0.5])
    p.addXY(nu_obs, int_obs, "Observations", lines=False)

    p.defaultKeyOverlay.hAlign = 0.93
    p.setBounds(0.5, 47, 0.1, 60)
    p.setLabels("Emission frequency, GHz", "Total intensity, sfu")
    return p


# The high-level interface that someone might actually want to use.


class Calculator(object):
    """An interface to the FK10 synchrotron routines.

    This class maintains state about the input parameters that can be passed
    to the routines, and can invoke them for you.

    **Constructor arguments**

    *shlib_path*
      The path to the compiled FK10 code, as described in the module-level
      documentation.

    Newly-constructed objects are initialized with::

      self.set_hybrid_parameters(12, 12)
      self.set_ignore_q_terms(False)
      self.set_trapezoidal_integration(15)

    """

    def __init__(self, shlib_path):
        self.func = FK10Invoker(shlib_path)
        self.in_vals = make_in_vals_array()

        # Some desirable defaults:
        self.set_hybrid_parameters(12, 12)
        self.set_ignore_q_terms(False)
        self.set_trapezoidal_integration(15)

        # The "intensity" results out of the FK10 code just scale linearly
        # with this parameter, so we can basically set it arbitrarily.
        self.in_vals[IN_VAL_AREA] = 1.0

        # The choice of this parameter actually affects the outputs from the
        # code, because it annoyingly tries to do the RT integral for us.
        # Fortunately find_rt_coefficients() can get something useful unless
        # the input depth, or problem parameters, are really bonkers.
        self.in_vals[IN_VAL_DEPTH] = 1e9

    @classmethod
    def new_for_fk10_fig9(cls, shlib_path):
        """Create a calculator initialized to reproduce Figure 9 from FK10.

        This is mostly to provide a handy way to create a new
        :class:`Calculator` instance that is initialized with reasonable
        values for all of its parameters.

        """
        inst = (
            cls(shlib_path)
            .set_thermal_background(2.1e7, 3e9)
            .set_bfield(48)
            .set_edist_powerlaw(0.016, 4.0, 3.7, 5e9 / 3)
            .set_freqs(100, 0.5, 50)
            .set_hybrid_parameters(12, 12)
            .set_ignore_q_terms(False)
            .set_obs_angle(50 * np.pi / 180)
            .set_padist_gaussian_loss_cone(0.5 * np.pi, 0.4)
            .set_trapezoidal_integration(15)
        )

        # haven't yet figure out how to deal with this part:
        inst.in_vals[0] = 1.33e18
        inst.in_vals[1] = 6e8
        return inst

    def compute_lowlevel(self, **kwargs):
        """Return the raw array computed by the FK10 code

        **Calling sequence**

        ``**kwargs``
          Passed on to :meth:`FK10Invoker.__call__`; you can specify
          a keyword *out_vals* to reuse an existing output array.
        Returns
          An array of shape ``(N, 5)``, where *N* has been specified
          by your frequency configuration. Layout as described in the
          main module documentation.

        """
        return self.func(self.in_vals, **kwargs)

    def set_bfield(self, B_G):
        """Set the strength of the local magnetic field.

        **Call signature**

        *B_G*
          The magnetic field strength, in Gauss
        Returns
          *self* for convenience in chaining.
        """
        if not (B_G > 0):
            raise ValueError("must have B_G > 0; got %r" % (B_G,))

        self.in_vals[IN_VAL_B] = B_G
        return self

    def set_bfield_for_s0(self, s0):
        """Set B to probe a certain harmonic number.

        **Call signature**

        *s0*
          The harmonic number to probe at the lowest frequency
        Returns
          *self* for convenience in chaining.

        This just proceeds from the relation ``nu = s nu_c = s e B / 2 pi m_e
        c``. Since *s* and *nu* scale with each other, if multiple frequencies
        are being probed, the harmonic numbers being probed will scale in the
        same way.

        """
        if not (s0 > 0):
            raise ValueError("must have s0 > 0; got %r" % (s0,))

        B0 = 2 * np.pi * cgs.me * cgs.c * self.in_vals[IN_VAL_FREQ0] / (cgs.e * s0)
        self.in_vals[IN_VAL_B] = B0
        return self

    def set_edist_powerlaw(self, emin_mev, emax_mev, delta, ne_cc):
        """Set the energy distribution function to a power law.

        **Call signature**

        *emin_mev*
          The minimum energy of the distribution, in MeV
        *emax_mev*
          The maximum energy of the distribution, in MeV
        *delta*
          The power-law index of the distribution
        *ne_cc*
          The number density of energetic electrons, in cm^-3.
        Returns
          *self* for convenience in chaining.
        """
        if not (emin_mev >= 0):
            raise ValueError("must have emin_mev >= 0; got %r" % (emin_mev,))
        if not (emax_mev >= emin_mev):
            raise ValueError(
                "must have emax_mev >= emin_mev; got %r, %r" % (emax_mev, emin_mev)
            )
        if not (delta >= 0):
            raise ValueError("must have delta >= 0; got %r, %r" % (delta,))
        if not (ne_cc >= 0):
            raise ValueError("must have ne_cc >= 0; got %r, %r" % (ne_cc,))

        self.in_vals[IN_VAL_EDIST] = EDIST_PLW
        self.in_vals[IN_VAL_EMIN] = emin_mev
        self.in_vals[IN_VAL_EMAX] = emax_mev
        self.in_vals[IN_VAL_DELTA1] = delta
        self.in_vals[IN_VAL_NB] = ne_cc
        return self

    def set_edist_powerlaw_gamma(self, gmin, gmax, delta, ne_cc):
        """Set the energy distribution function to a power law in the Lorentz factor

        **Call signature**

        *gmin*
          The minimum Lorentz factor of the distribution
        *gmax*
          The maximum Lorentz factor of the distribution
        *delta*
          The power-law index of the distribution
        *ne_cc*
          The number density of energetic electrons, in cm^-3.
        Returns
          *self* for convenience in chaining.
        """
        if not (gmin >= 1):
            raise ValueError("must have gmin >= 1; got %r" % (gmin,))
        if not (gmax >= gmin):
            raise ValueError("must have gmax >= gmin; got %r, %r" % (gmax, gmin))
        if not (delta >= 0):
            raise ValueError("must have delta >= 0; got %r, %r" % (delta,))
        if not (ne_cc >= 0):
            raise ValueError("must have ne_cc >= 0; got %r, %r" % (ne_cc,))

        self.in_vals[IN_VAL_EDIST] = EDIST_PLG
        self.in_vals[IN_VAL_EMIN] = (gmin - 1) * E0_MEV
        self.in_vals[IN_VAL_EMAX] = (gmax - 1) * E0_MEV
        self.in_vals[IN_VAL_DELTA1] = delta
        self.in_vals[IN_VAL_NB] = ne_cc
        return self

    def set_freqs(self, n, f_lo_ghz, f_hi_ghz):
        """Set the frequency grid on which to perform the calculations.

        **Call signature**

        *n*
          The number of frequency points to sample.
        *f_lo_ghz*
          The lowest frequency to sample, in GHz.
        *f_hi_ghz*
          The highest frequency to sample, in GHz.
        Returns
          *self* for convenience in chaining.

        """
        if not (f_lo_ghz >= 0):
            raise ValueError("must have f_lo_ghz >= 0; got %r" % (f_lo_ghz,))
        if not (f_hi_ghz >= f_lo_ghz):
            raise ValueError(
                "must have f_hi_ghz >= f_lo_ghz; got %r, %r" % (f_hi_ghz, f_lo_ghz)
            )
        if not n >= 1:
            raise ValueError("must have n >= 1; got %r" % (n,))

        self.in_vals[IN_VAL_NFREQ] = n
        self.in_vals[IN_VAL_FREQ0] = f_lo_ghz * 1e9  # GHz => Hz
        self.in_vals[IN_VAL_LOGDFREQ] = np.log10(f_hi_ghz / f_lo_ghz) / n
        return self

    def set_hybrid_parameters(self, s_C, s_WH, do_renorm=True):
        """Set the hybrid/renormalization control parameters.

        **Call signature**

        *s_C*
          The harmonic number above which the continuous approximation
          is used (with special behavior; see below).
        *s_WH*
          The harmonic number above which the Wild-Hill BEssel function
          approximations are used.
        *do_renorm* (default True)
          Whether to do any renormalization at all.
        Returns
          *self* for convenience in chaining.

        FK10 uses frequency parameters f^C_cr and f^WH_cr to control some of
        its optimizations. This function sets these parameters as multiples of
        the electron cyclotron frequency (f_Be in FK10 notation): e.g.,
        ``f^C_cr = s_C * f_Be``.

        At frequencies above f^C_cr, the "continuum" approximation is
        introduced, replacing the "exact" sum with an integral. At frequencies
        above f^WH_cr, the Wild-Hild approximations to the Bessel functions
        are used. In both cases, the activation of the optimizations can
        result in normalization shifts in the calculations. "Renormalization"
        computes these shifts (by doing both kinds of calculations at the
        transition frequencies) and attempts to correct them. (Some of the
        FK10 documentation seems to refer to renormalization as
        "R-optimization".)

        If f^C_cr is below the lowest frequency integrated, all calculations
        will be done in continuum mode. In this case, the sign of *s_C* sets
        whether Wild-Hill renormalization is applied. If *s_C* is negative and
        f^WH_cr is above the lowest frequency integration, renormalization is
        done. Otherwise, it is not.

        The documentation regarding f^WH_cr is confusing. It states that
        f^WH_cr only matters if (1) s_WH < s_C or (2) s_C < 0 and f^WH_cr >
        f_0. It is not obvious to me why s_WH > s_C should only matter if s_C
        < 0, but that's what's implied.

        In most examples in FK10, both of these parameters are set to 12.

        """
        self.in_vals[IN_VAL_FCCR] = s_C
        self.in_vals[IN_VAL_FWHCR] = s_WH
        self.in_vals[IN_VAL_RENORMFLAG] = 1 if do_renorm else 0
        return self

    def set_obs_angle(self, theta_rad):
        """Set the observer angle relative to the field.

        **Call signature**

        *theta_rad*
          The angle between the ray path and the local magnetic field,
          in radians.
        Returns
          *self* for convenience in chaining.
        """
        self.in_vals[IN_VAL_THETA] = theta_rad * 180 / np.pi  # rad => deg
        return self

    def set_one_freq(self, f_ghz):
        """Set the code to calculate results at just one frequency.

        **Call signature**

        *f_ghz*
          The frequency to sample, in GHz.
        Returns
          *self* for convenience in chaining.

        """
        if not (f_ghz >= 0):
            raise ValueError("must have f_lo_ghz >= 0; got %r" % (f_lo_ghz,))

        self.in_vals[IN_VAL_NFREQ] = 1
        self.in_vals[IN_VAL_FREQ0] = f_ghz * 1e9  # GHz -> Hz
        self.in_vals[IN_VAL_LOGDFREQ] = 1.0
        return self

    def set_padist_gaussian_loss_cone(self, boundary_rad, expwidth):
        """Set the pitch-angle distribution to a Gaussian loss cone.

        **Call signature**

        *boundary_rad*
          The angle inside which there are no losses, in radians.
        *expwidth*
          The characteristic width of the Gaussian loss profile
          *in direction-cosine units*.
        Returns
          *self* for convenience in chaining.

        See ``OnlineI.pdf`` in the Supplementary Data for a precise
        definition. (And note the distinction between α_c and μ_c since not
        everything is direction cosines.)

        """
        self.in_vals[IN_VAL_PADIST] = PADIST_GLC
        self.in_vals[IN_VAL_LCBDY] = boundary_rad * 180 / np.pi  # rad => deg
        self.in_vals[IN_VAL_DELTAMU] = expwidth
        return self

    def set_padist_isotropic(self):
        """Set the pitch-angle distribution to be isotropic.

        **Returns**
          *self* for convenience in chaining.

        """
        self.in_vals[IN_VAL_PADIST] = PADIST_ISO
        return self

    def set_ignore_q_terms(self, ignore_q_terms):
        """Set whether "Q" terms are ignored.

        **Call signature**

        *ignore_q_terms*
          If true, ignore "Q" terms in the integrations.
        Returns
          *self* for convenience in chaining.

        See Section 4.3 of FK10 and ``OnlineII.pdf`` in the Supplementary Data
        for a precise explanation. The default is to *not* ignore the terms.

        """
        # Note that we are ignoring the magic 1 value that only applies to the
        # "HomSrc_C" version of the library.
        self.in_vals[IN_VAL_QFLAG] = 0 if ignore_q_terms else 2
        return self

    def set_thermal_background(self, T_K, nth_cc):
        """Set the properties of the background thermal plasma.

        **Call signature**

        *T_K*
          The temperature of the background plasma, in Kelvin.
        *nth_cc*
          The number density of thermal electrons, in cm^-3.
        Returns
          *self* for convenience in chaining.

        Note that the parameters set here are the same as the ones that
        describe the thermal electron distribution, if you choose one of the
        electron energy distributions that explicitly models a thermal
        component ("thm", "tnt", "tnp", "tng", "kappa" in the code's
        terminology). For the power-law-y electron distributions, these
        parameters are used to calculate dispersion parameters (e.g.
        refractive indices) and a free-free contribution, but their
        synchrotron contribution is ignored.

        """
        if not (T_K >= 0):
            raise ValueError("must have T_K >= 0; got %r" % (T_K,))
        if not (nth_cc >= 0):
            raise ValueError("must have nth_cc >= 0; got %r, %r" % (nth_cc,))

        self.in_vals[IN_VAL_T0] = T_K
        self.in_vals[IN_VAL_N0] = nth_cc
        return self

    def set_trapezoidal_integration(self, n):
        """Set the code to use trapezoidal integration.

        **Call signature**

        *n*
          Use this many nodes
        Returns
          *self* for convenience in chaining.

        """
        if not (n >= 2):
            raise ValueError("must have n >= 2; got %r" % (n,))

        self.in_vals[IN_VAL_INTEG_METH] = n + 1
        return self

    def find_rt_coefficients(self, depth0=None):
        """Figure out emission and absorption coefficients for the current parameters.

        **Argument**

        *depth0* (default None)
          A first guess to use for a good integration depth, in cm. If None,
          the most recent value is used.

        **Return value**

        A tuple ``(j_O, alpha_O, j_X, alpha_X)``, where:

        *j_O*
          The O-mode emission coefficient, in erg/s/cm^3/Hz/sr.
        *alpha_O*
          The O-mode absorption coefficient, in cm^-1.
        *j_X*
          The X-mode emission coefficient, in erg/s/cm^3/Hz/sr.
        *alpha_X*
          The X-mode absorption coefficient, in cm^-1.

        The main outputs of the FK10 code are intensities and "damping
        factors" describing a radiative transfer integration of the emission
        from a homogeneous source. But there are times when we'd rather just
        know what the actual emission and absorption coefficients are. These
        can be backed out from the FK10 outputs, but only if the "damping
        factor" takes on an intermediate value not extremely close to either 0
        or 1. Unfortunately, there's no way for us to know a priori what
        choice of the "depth" parameter will yield a nice value for the
        damping factor. This routine automatically figures one out, by
        repeatedly running the calculation.

        To keep things simple, this routine requires that you only be solving
        for coefficients for one frequency at a time (e.g.,
        :meth:`set_one_freq`).

        """
        if self.in_vals[IN_VAL_NFREQ] != 1:
            raise Exception(
                "must have nfreq=1 to run Calculator.find_rt_coefficients()"
            )

        if depth0 is not None:
            depth = depth0
            self.in_vals[IN_VAL_DEPTH] = depth0
        else:
            depth = self.in_vals[IN_VAL_DEPTH]

        scale_factor = 100
        buf = np.empty((1, 5), dtype=np.float32)

        def classify(damping_factor):
            if damping_factor >= 0.99:
                return 1
            if damping_factor <= 0.01:
                return -1
            return 0

        DONE, SHRINK, GROW, ABORT = 0, 1, 2, 3

        actions = {
            (-1, -1): SHRINK,
            (-1, 0): SHRINK,
            (-1, 1): ABORT,
            (0, -1): SHRINK,
            (0, 0): DONE,
            (0, 1): GROW,
            (1, -1): ABORT,
            (1, 0): GROW,
            (1, 1): GROW,
        }

        last_change = DONE  # our first change will be treated as a change in direction

        for attempt_number in range(20):
            self.compute_lowlevel(out_values=buf)
            co = classify(buf[0, OUT_VAL_ODAMP])
            cx = classify(buf[0, OUT_VAL_XDAMP])
            action = actions[co, cx]
            ###print('Z', attempt_number, self.in_vals[IN_VAL_DEPTH], last_change, buf, co, cx, action)

            if action == DONE:
                break
            elif action == ABORT:
                raise Exception("depths of X and O modes are seriously incompatible")
            elif action == GROW:
                if last_change != GROW:
                    scale_factor *= 0.3
                depth *= scale_factor
                last_change = GROW
            elif action == SHRINK:
                if last_change != SHRINK:
                    scale_factor *= 0.3
                depth /= scale_factor
                last_change = SHRINK

            self.in_vals[IN_VAL_DEPTH] = depth
        else:
            # If we get here, we never explicitly quit the loop
            raise Exception("depth-finding algorithm did not converge!")

        # OK, we found some good depths! Now calculate the RT coefficients. I believe that
        # I'm doing this right ...

        sfu_to_specintens = (
            1e4 * cgs.cgsperjy * cgs.cmperau**2 / self.in_vals[IN_VAL_AREA]
        )

        damp_X = buf[0, OUT_VAL_XDAMP]
        alpha_X = -np.log(damp_X) / depth
        si_X = buf[0, OUT_VAL_XINT] * sfu_to_specintens
        j_X = si_X * alpha_X / (1 - damp_X)

        damp_O = buf[0, OUT_VAL_ODAMP]
        alpha_O = -np.log(damp_O) / depth
        si_O = buf[0, OUT_VAL_OINT] * sfu_to_specintens
        j_O = si_O * alpha_O / (1 - damp_O)

        return (j_O, alpha_O, j_X, alpha_X)

    def find_rt_coefficients_tot_intens(self, depth0=None):
        """Figure out total-intensity emission and absorption coefficients for the
        current parameters.

        **Argument**

        *depth0* (default None)
          A first guess to use for a good integration depth, in cm. If None,
          the most recent value is used.

        **Return value**

        A tuple ``(j_I, alpha_I)``, where:

        *j_I*
          The total intensity emission coefficient, in erg/s/cm^3/Hz/sr.
        *alpha_I*
          The total intensity absorption coefficient, in cm^-1.

        See :meth:`find_rt_coefficients` for an explanation how this routine
        works. This version merely postprocesses the results from that method
        to convert the coefficients to refer to total intensity.

        """
        j_O, alpha_O, j_X, alpha_X = self.find_rt_coefficients(depth0=depth0)
        j_I = j_O + j_X
        alpha_I = 0.5 * (alpha_O + alpha_X)  # uhh... right?
        return (j_I, alpha_I)
