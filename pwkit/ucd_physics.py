# -*- mode: python; coding: utf-8 -*-
# Copyright 2013-2014 Peter Williams <peter@newton.cx> and collaborators.
# Licensed under the MIT license.

"""pwkit.ucd_physics - Physical calculations for (ultra)cool dwarfs.

These functions generally implement various nontrivial physical relations
published in the literature. See docstrings for references.

Functions:

bcj_from_spt
  J-band bolometric correction from SpT.
bck_from_spt
  K-band bolometric correction from SpT.
load_bcah98_mass_radius
  Load Baraffe+ 1998 mass/radius data.
mass_from_j
  Mass from absolute J magnitude.
mk_radius_from_mass_bcah98
  Radius from mass, using BCAH98 models.
tauc_from_mass
  Convective turnover time from mass.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = str ('''bcj_from_spt bck_from_spt load_bcah98_mass_radius mass_from_j
                  mk_radius_from_mass_bcah98 tauc_from_mass''').split ()

# Implementation note: we use the numutil.broadcastize() decorator to be able
# to handle both scalar and vector arguments semi-transparently. I'd also like
# us to be able to handle Uvals and Lvals, which aren't going to be compatible
# with this approach. The latter will also present challenges for
# bounds-checking of inputs, so I'm going the numpy route for now. Not sure
# what to do about this in general.

import numpy as np

from . import cgs, msmt, numutil


# Bolometric luminosity estimation.

@numutil.broadcastize (1)
def bcj_from_spt (spt):
    """Calculate a bolometric correction constant for a J band magnitude based on
    a spectral type, using the fit of Wilking+ (1999AJ....117..469W).

    spt - Numerical spectral type. M0=0, M9=9, L0=10, ...

    Returns: the correction `bcj` such that `m_bol = j_abs + bcj`, or NaN if
    `spt` is out of range.

    Valid values of `spt` are between 0 and 10.

    """
    return np.where ((spt >= 0) & (spt <= 10),
                     1.53 + 0.148 * spt - 0.0105 * spt**2,
                     np.nan)


@numutil.broadcastize (1)
def bck_from_spt (spt):
    """Calculate a bolometric correction constant for a J band magnitude based on
    a spectral type, using the fits of Wilking+ (1999AJ....117..469W), Dahn+
    (2002AJ....124.1170D), and Nakajima+ (2004ApJ...607..499N).

    spt - Numerical spectral type. M0=0, M9=9, L0=10, ...

    Returns: the correction `bck` such that `m_bol = k_abs + bck`, or NaN if
    `spt` is out of range.

    Valid values of `spt` are between 2 and 30.

    """

    # NOTE: the way np.piecewise() is implemented, the last 'true' value in
    # the condition list is the one that takes precedence. This motivates the
    # construction of our condition list.
    #
    # XXX: I've restructured the implementation; this needs testing!

    spt = np.asfarray (spt) # we crash with integer inputs for some reason.
    return np.piecewise (spt,
                         [spt < 30,
                          spt < 19,
                          spt <= 14,
                          spt < 10,
                          (spt < 2) | (spt >= 30)],
                         [lambda s: 3.41 - 0.21 * (s - 20), # Nakajima
                          lambda s: 3.42 - 0.075 * (s - 14), # Dahn, Nakajima
                          lambda s: 3.42 + 0.075 * (s - 14), # Dahn, Nakajima
                          lambda s: 2.43 + 0.0895 * s, # Wilking; only ok for spt >= M2!
                          np.nan])


Mbol_sun = 4.7554
"""Absolute bolometric luminosity of the Sun. Copied from Eric Mamajek's star
notes:
https://sites.google.com/site/mamajeksstarnotes/basic-astronomical-data-for-the-sun

Quoted uncertainty is 0.0004.

Note that this bit isn't UCD-specific and could/should go elsewhere, say,
astutil.

NOTE! I haven't verified if this value is consistent with the one implicitly
adopted by the various relations that I use above! This could result in errors
of up to ~0.1 mag. Cf. Torres, 2010AJ....140.1158T.

"""
def lbol_from_mbol (mbol, format='cgs'):
    from .cgs import lsun

    x = 0.4 * (Mbol_sun - mbol)

    if format == 'cgs':
        return lsun * 10**x
    elif format == 'logsun':
        return x
    elif format == 'logcgs':
        return np.log10 (lsun) + x

    raise ValueError ('unrecognized output format %r' % format)


@numutil.broadcastize (4)
def lbol_from_spt_dist_mag (sptnum, dist_pc, jmag, kmag, format='cgs'):
    """Estimate a UCD's bolometric luminosity given some basic parameters.

    sptnum: the spectral type as a number; 8 -> M8; 10 -> L0 ; 20 -> T0
      Valid values range between 0 and 30, ie M0 to Y0.
    dist_pc: distance to the object in parsecs
    jmag: object's J-band magnitude or NaN (*not* None) if unavailable
    kmag: same with K-band magnitude
    format: either 'cgs', 'logcgs', or 'logsun', defining the form of the
      outputs. Logarithmic quantities are base 10.

    This routine can be used with vectors of measurements. The result will be
    NaN if a value cannot be computed. This routine implements the method
    documented in the Appendix of Williams et al., 2014ApJ...785....9W
    (doi:10.1088/0004-637X/785/1/9).

    """
    bcj = bcj_from_spt (sptnum)
    bck = bck_from_spt (sptnum)

    n = np.zeros (sptnum.shape, dtype=int)
    app_mbol = np.zeros (sptnum.shape)

    w = np.isfinite (bcj) & np.isfinite (jmag)
    app_mbol[w] += jmag[w] + bcj[w]
    n[w] += 1

    w = np.isfinite (bck) & np.isfinite (kmag)
    app_mbol[w] += kmag[w] + bck[w]
    n[w] += 1

    w = (n != 0)
    abs_mbol = (app_mbol[w] / n[w]) - 5 * (np.log10 (dist_pc[w]) - 1)
    # note: abs_mbol is filtered by `w`

    lbol = np.empty (sptnum.shape)
    lbol.fill (np.nan)
    lbol[w] = lbol_from_mbol (abs_mbol, format=format)
    return lbol


# Mass estimation.

def _delfosse_mass_from_j_helper (j_abs):
    x = 1e-3 * (1.6 + 6.01 * j_abs + 14.888 * j_abs**2 +
                -5.3557 * j_abs**3 + 0.28518 * j_abs**4)
    return 10**x * cgs.msun


@numutil.broadcastize (1)
def mass_from_j (j_abs):
    """Estimate mass in cgs from absolute J magnitude, using the relationship of
    Delfosse+ (2000A&A...364..217D).

    j_abs - The absolute J magnitude.

    Returns: the estimated mass in grams.

    If j_abs > 11, a fixed result of 0.1 Msun is returned. Values of j_abs <
    5.5 are illegal and get NaN. There is a discontinuity in the relation at
    j_abs = 11, which yields 0.0824 Msun.

    """
    j_abs = np.asfarray (j_abs)
    return np.piecewise (j_abs,
                         [j_abs > 11,
                          j_abs <= 11,
                          j_abs < 5.5],
                         [0.1 * cgs.msun,
                          _delfosse_mass_from_j_helper,
                          np.nan])


# Radius estimation.

def load_bcah98_mass_radius (tablelines, metallicity=0, heliumfrac=0.275,
                             age_gyr=5., age_tol=0.05):
    """Load mass and radius from the main data table for the famous models of
    Baraffe+ (1998A&A...337..403B).

    tablelines
      An iterable yielding lines from the table data file.
      I've named the file '1998A&A...337..403B_tbl1-3.dat'
      in some repositories (it's about 150K, not too bad).
    metallicity
      The metallicity of the model to select.
    heliumfrac
      The helium fraction of the model to select.
    age_gyr
      The age of the model to select, in Gyr.
    age_tol
      The tolerance on the matched age, in Gyr.

    Returns: (mass, radius), where both are Numpy arrays.

    The ages in the data table vary slightly at fixed metallicity and helium
    fraction. Therefore, there needs to be a tolerance parameter for matching
    the age.

    """
    mdata, rdata = [], []

    for line in tablelines:
        a = line.strip ().split ()

        thismetallicity = float (a[0])
        if thismetallicity != metallicity:
            continue

        thisheliumfrac = float (a[1])
        if thisheliumfrac != heliumfrac:
            continue

        thisage = float (a[4])
        if abs (thisage - age_gyr) > age_tol:
            continue

        mass = float (a[3]) * cgs.msun
        teff = float (a[5])
        mbol = float (a[7])

        # XXX to check: do they specify m_bol_sun = 4.64? IIRC, yes.
        lbol = 10**(0.4 * (4.64 - mbol)) * cgs.lsun
        area = lbol / (cgs.sigma * teff**4)
        r = np.sqrt (area / (4 * np.pi))

        mdata.append (mass)
        rdata.append (r)

    return np.asarray (mdata), np.asarray (rdata)


def mk_radius_from_mass_bcah98 (*args, **kwargs):
    """Create a function that maps (sub)stellar mass to radius, based on the
    famous models of Baraffe+ (1998A&A...337..403B).

    tablelines
      An iterable yielding lines from the table data file.
      I've named the file '1998A&A...337..403B_tbl1-3.dat'
      in some repositories (it's about 150K, not too bad).
    metallicity
      The metallicity of the model to select.
    heliumfrac
      The helium fraction of the model to select.
    age_gyr
      The age of the model to select, in Gyr.
    age_tol
      The tolerance on the matched age, in Gyr.

    Returns: a function mtor(mass_g), return a radius in cm as a function of a
    mass in grams. The mass must be between 0.05 and 0.7 Msun.

    The ages in the data table vary slightly at fixed metallicity and helium
    fraction. Therefore, there needs to be a tolerance parameter for matching
    the age.

    This function requires Scipy.

    """
    from scipy.interpolate import UnivariateSpline
    m, r = load_bcah98_mass_radius (*args, **kwargs)
    spl = UnivariateSpline (m, r, s=1)

    # This allows us to do range-checking with either scalars or vectors with
    # minimal gymnastics.
    @numutil.broadcastize (1)
    def interp (mass_g):
        if np.any (mass_g < 0.05 * cgs.msun) or np.any (mass_g > 0.7 * cgs.msun):
            raise ValueError ('mass_g must must be between 0.05 and 0.7 Msun')
        return spl (mass_g)

    return interp


# Estimation of the convective turnover time.

@numutil.broadcastize (1)
def tauc_from_mass (mass_g):
    """Estimate the convective turnover time from mass, using the method described
    in Cook+ (2014ApJ...785...10C).

    mass_g - UCD mass in grams.

    Returns: the convective turnover timescale in seconds.

    Masses larger than 1.3 Msun are out of range and yield NaN. If the mass is
    <0.1 Msun, the turnover time is fixed at 70 days.

    The Cook method was inspired by the description in McLean+
    (2012ApJ...746...23M). It is a hybrid of the method described in Reiners &
    Basri (2010ApJ...710..924R) and the data shown in Kiraga & Stepien
    (2007AcA....57..149K). However, this version imposes the 70-day cutoff in
    terms of mass, not spectral type, so that it is entirely defined in terms
    of a single quantity.

    There are discontinuities between the different break points! Any future
    use should tweak the coefficients to make everything smooth.

    """
    m = mass_g / cgs.msun
    return np.piecewise (m,
                         [m < 1.3,
                          m < 0.82,
                          m < 0.65,
                          m < 0.1],
                         [lambda x: 61.7 - 44.7 * x,
                          25.,
                          lambda x: 86.9 - 94.3 * x,
                          70.,
                          np.nan]) * 86400.
