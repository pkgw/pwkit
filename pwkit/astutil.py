# -*- mode: python; coding: utf-8 -*-
# Copyright 2012-2016 Peter Williams <peter@newton.cx> and collaborators.
# Licensed under the MIT License.

"""This module provides functions and constants for doing a variety of basic
calculations and conversions that come up in astronomy.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = str(
    """np pi twopi halfpi R2A A2R R2D D2R R2H H2R F2S S2F J2000 angcen orientcen
                  fmthours fmtdeglon fmtdeglat fmtradec parsehours parsedeglat
                  parsedeglon sphdist sphbear sphofs parang gaussian_convolve
                  gaussian_deconvolve load_skyfield_data AstrometryInfo app2abs abs2app"""
).split()

import numpy as np

from . import unicode_to_str, PKError
from .numutil import broadcastize


# Workaround for Sphinx bug 1641. Without this, the "autoattribute" directive
# does not work if the print_function future is used. Copied from
# https://github.com/sphinx-doc/sphinx/issues/1641#issuecomment-204323049.
# See commit message associated with this code for more detail.

try:
    import builtins

    print_ = getattr(builtins, "print")
except ImportError:
    import __builtin__

    print_ = getattr(__builtin__, "print")


# Constants.

pi = np.pi
twopi = 2 * pi
halfpi = 0.5 * pi

R2A = 3600 * 180 / pi
A2R = pi / (3600 * 180)
R2D = 180 / pi
D2R = pi / 180
R2H = 12 / pi
H2R = pi / 12
F2S = 1 / np.sqrt(8 * np.log(2))  # FWHM to sigma
S2F = np.sqrt(8 * np.log(2))

J2000 = 51544.5


# Angle and orientation (PA) normalization
#
# PA's seem to usually be given in the range [-90, 90]

angcen = lambda a: (((a + pi) % twopi) - pi)
orientcen = lambda a: (((a + halfpi) % pi) - halfpi)


# Formatting/parsing of lat/long/etc


def _fmtsexagesimal(base, norm, basemax, seps, precision=3):
    if norm == "none":
        pass
    elif norm == "raise":
        if base > basemax or base < 0:
            raise ValueError("illegal coordinate of %f" % base)
    elif norm == "wrap":
        base = base % basemax
    else:
        raise ValueError('unrecognized normalization type "%s"' % norm)

    if len(seps) < 2:
        # To ponder: we accept len(seps) > 3; seems OK.
        raise ValueError(
            "there must be at least two sexagesimal separators; "
            'got value "%s"' % seps
        )

    precision = max(int(precision), 0)
    if precision == 0:
        width = 2
    else:
        width = precision + 3

    basewidth = len(str(basemax))

    bs = int(np.floor(base))
    min = int(np.floor((base - bs) * 60))
    sec = round(3600 * (base - bs - min / 60.0), precision)

    if sec >= 60:
        # Can happen if we round up
        sec -= 60
        min += 1

        if min >= 60:
            min -= 60
            bs += 1

            if bs >= basemax:
                bs -= basemax

    if len(seps) > 2:
        sep2 = seps[2]
    else:
        sep2 = ""

    return "%0*d%s%02d%s%0*.*f%s" % (
        basewidth,
        bs,
        seps[0],
        min,
        seps[1],
        width,
        precision,
        sec,
        sep2,
    )


def fmthours(radians, norm="wrap", precision=3, seps="::"):
    """Format an angle as sexagesimal hours in a string.

    Arguments are:

    radians
      The angle, in radians.
    norm (default "wrap")
      The normalization mode, used for angles outside of the standard range
      of 0 to 2π. If "none", the value is formatted ignoring any potential
      problems. If "wrap", it is wrapped to lie within the standard range.
      If "raise", a :exc:`ValueError` is raised.
    precision (default 3)
      The number of decimal places in the "seconds" place to use in the
      formatted string.
    seps (default "::")
      A two- or three-item iterable, used to separate the hours, minutes, and
      seconds components. If a third element is present, it appears after the
      seconds component. Specifying "hms" yields something like "12h34m56s";
      specifying ``['', '']`` yields something like "123456".

    Returns a string.

    """
    return _fmtsexagesimal(radians * R2H, norm, 24, seps, precision=precision)


def fmtdeglon(radians, norm="wrap", precision=2, seps="::"):
    """Format a longitudinal angle as sexagesimal degrees in a string.

    Arguments are:

    radians
      The angle, in radians.
    norm (default "wrap")
      The normalization mode, used for angles outside of the standard range
      of 0 to 2π. If "none", the value is formatted ignoring any potential
      problems. If "wrap", it is wrapped to lie within the standard range.
      If "raise", a :exc:`ValueError` is raised.
    precision (default 2)
      The number of decimal places in the "arcseconds" place to use in the
      formatted string.
    seps (default "::")
      A two- or three-item iterable, used to separate the degrees, arcminutes,
      and arcseconds components. If a third element is present, it appears
      after the arcseconds component. Specifying "dms" yields something like
      "12d34m56s"; specifying ``['', '']`` yields something like "123456".

    Returns a string.

    """
    return _fmtsexagesimal(radians * R2D, norm, 360, seps, precision=precision)


def fmtdeglat(radians, norm="raise", precision=2, seps="::"):
    """Format a latitudinal angle as sexagesimal degrees in a string.

    Arguments are:

    radians
      The angle, in radians.
    norm (default "raise")
      The normalization mode, used for angles outside of the standard range
      of -π/2 to π/2. If "none", the value is formatted ignoring any potential
      problems. If "wrap", it is wrapped to lie within the standard range.
      If "raise", a :exc:`ValueError` is raised.
    precision (default 2)
      The number of decimal places in the "arcseconds" place to use in the
      formatted string.
    seps (default "::")
      A two- or three-item iterable, used to separate the degrees, arcminutes,
      and arcseconds components. If a third element is present, it appears
      after the arcseconds component. Specifying "dms" yields something like
      "+12d34m56s"; specifying ``['', '']`` yields something like "123456".

    Returns a string. The return value always includes a plus or minus sign.
    Note that the default of *norm* is different than in :func:`fmthours` and
    :func:`fmtdeglon` since it's not so clear what a "latitude" of 110 degrees
    (e.g.) means.

    """
    if norm == "none":
        pass
    elif norm == "raise":
        if radians > halfpi or radians < -halfpi:
            raise ValueError("illegal latitude of %f radians" % radians)
    elif norm == "wrap":
        radians = angcen(radians)
        if radians > halfpi:
            radians = pi - radians
        elif radians < -halfpi:
            radians = -pi - radians
    else:
        raise ValueError('unrecognized normalization type "%s"' % norm)

    if len(seps) < 2:
        # To ponder: we accept len(seps) > 3; seems OK.
        raise ValueError(
            "there must be at least two sexagesimal separators; "
            'got value "%s"' % seps
        )

    precision = max(int(precision), 0)
    if precision == 0:
        width = 2
    else:
        width = precision + 3

    degrees = radians * R2D

    if degrees >= 0:
        sgn = "+"
    else:
        sgn = "-"
        degrees = -degrees

    deg = int(np.floor(degrees))
    amin = int(np.floor((degrees - deg) * 60))
    asec = round(3600 * (degrees - deg - amin / 60.0), precision)

    if asec >= 60:
        # Can happen if we round up
        asec -= 60
        amin += 1

        if amin >= 60:
            amin -= 60
            deg += 1

    if len(seps) > 2:
        sep2 = seps[2]
    else:
        sep2 = ""

    return "%s%02d%s%02d%s%0*.*f%s" % (
        sgn,
        deg,
        seps[0],
        amin,
        seps[1],
        width,
        precision,
        asec,
        sep2,
    )


def fmtradec(rarad, decrad, precision=2, raseps="::", decseps="::", intersep=" "):
    """Format equatorial coordinates in a single sexagesimal string.

    Returns a string of the RA/lon coordinate, formatted as sexagesimal hours,
    then *intersep*, then the Dec/lat coordinate, formatted as degrees. This
    yields something like "12:34:56.78 -01:23:45.6". Arguments are:

    rarad
      The right ascension coordinate, in radians. More generically, this is
      the longitudinal coordinate; note that the ordering in this function
      differs than the other spherical functions, which generally prefer
      coordinates in "lat, lon" order.
    decrad
      The declination coordinate, in radians. More generically, this is the
      latitudinal coordinate.
    precision (default 2)
      The number of decimal places in the "arcseconds" place of the
      latitudinal (declination) coordinate. The longitudinal (right ascension)
      coordinate gets one additional place, since hours are bigger than
      degrees.
    raseps (default "::")
      A two- or three-item iterable, used to separate the hours, minutes, and
      seconds components of the RA/lon coordinate. If a third element is
      present, it appears after the seconds component. Specifying "hms" yields
      something like "12h34m56s"; specifying ``['', '']`` yields something
      like "123456".
    decseps (default "::")
      A two- or three-item iterable, used to separate the degrees, arcminutes,
      and arcseconds components of the Dec/lat coordinate.
    intersep (default " ")
      The string separating the RA/lon and Dec/lat coordinates

    """
    return (
        fmthours(rarad, precision=precision + 1, seps=raseps)
        + str(intersep)
        + fmtdeglat(decrad, precision=precision, seps=decseps)
    )


# Parsing routines are currently very lame.


def _parsesexagesimal(sxgstr, desc, negok):
    sxgstr_orig = sxgstr
    sgn = 1

    if sxgstr[0] == "-":
        if negok:
            sgn = -1
            sxgstr = sxgstr[1:]
        else:
            raise ValueError("illegal negative %s expression: %s" % (desc, sxgstr_orig))

    try:
        # TODO: other separators ...
        bs, mn, sec = sxgstr.split(":")
        bs = int(bs)
        mn = int(mn)
        sec = float(sec)
    except Exception:
        raise ValueError("unable to parse as %s: %s" % (desc, sxgstr_orig))

    if mn < 0 or mn > 59 or sec < 0 or sec >= 60.0:
        raise ValueError("illegal sexagesimal %s expression: %s" % (desc, sxgstr_orig))
    if bs < 0:  # two minus signs, or something
        raise ValueError("illegal negative %s expression: %s" % (desc, sxgstr_orig))

    return sgn * (bs + mn / 60.0 + sec / 3600.0)


def parsehours(hrstr):
    """Parse a string formatted as sexagesimal hours into an angle.

    This function converts a textual representation of an angle, measured in
    hours, into a floating point value measured in radians. The format of
    *hrstr* is very limited: it may not have leading or trailing whitespace,
    and the components of the sexagesimal representation must be separated by
    colons. The input must therefore resemble something like
    ``"12:34:56.78"``. A :exc:`ValueError` will be raised if the input does
    not resemble this template. Hours greater than 24 are not allowed, but
    negative values are.

    """
    hr = _parsesexagesimal(hrstr, "hours", False)
    if hr >= 24:
        raise ValueError("illegal hour specification: " + hrstr)
    return hr * H2R


def parsedeglat(latstr):
    """Parse a latitude formatted as sexagesimal degrees into an angle.

    This function converts a textual representation of a latitude, measured in
    degrees, into a floating point value measured in radians. The format of
    *latstr* is very limited: it may not have leading or trailing whitespace,
    and the components of the sexagesimal representation must be separated by
    colons. The input must therefore resemble something like
    ``"-00:12:34.5"``. A :exc:`ValueError` will be raised if the input does
    not resemble this template. Latitudes greater than 90 or less than -90
    degrees are not allowed.

    """
    deg = _parsesexagesimal(latstr, "latitude", True)
    if abs(deg) > 90:
        raise ValueError("illegal latitude specification: " + latstr)
    return deg * D2R


def parsedeglon(lonstr):
    """Parse a longitude formatted as sexagesimal degrees into an angle.

    This function converts a textual representation of a longitude, measured
    in degrees, into a floating point value measured in radians. The format of
    *lonstr* is very limited: it may not have leading or trailing whitespace,
    and the components of the sexagesimal representation must be separated by
    colons. The input must therefore resemble something like
    ``"270:12:34.5"``. A :exc:`ValueError` will be raised if the input does
    not resemble this template. Values of any sign and magnitude are allowed,
    and they are not normalized (e.g. to lie within the range [0, 2π]).

    """
    return _parsesexagesimal(lonstr, "longitude", True) * D2R


# Spherical trig


@broadcastize(4)
def sphdist(lat1, lon1, lat2, lon2):
    """Calculate the distance between two locations on a sphere.

    lat1
      The latitude of the first location.
    lon1
      The longitude of the first location.
    lat2
      The latitude of the second location.
    lon2
      The longitude of the second location.

    Returns the separation in radians. All arguments are in radians as well.
    The arguments may be vectors.

    Note that the ordering of the arguments maps to the nonstandard ordering
    ``(Dec, RA)`` in equatorial coordinates. In a spherical projection it maps
    to ``(Y, X)`` which may also be unexpected.

    The distance is computed with the "specialized Vincenty formula". Faster
    but more error-prone formulae are possible; see Wikipedia on Great-circle
    Distance.

    """
    cd = np.cos(lon2 - lon1)
    sd = np.sin(lon2 - lon1)
    c2 = np.cos(lat2)
    c1 = np.cos(lat1)
    s2 = np.sin(lat2)
    s1 = np.sin(lat1)
    a = np.sqrt((c2 * sd) ** 2 + (c1 * s2 - s1 * c2 * cd) ** 2)
    b = s1 * s2 + c1 * c2 * cd
    return np.arctan2(a, b)


@broadcastize(4)
def sphbear(lat1, lon1, lat2, lon2, tol=1e-15):
    """Calculate the bearing between two locations on a sphere.

    lat1
      The latitude of the first location.
    lon1
      The longitude of the first location.
    lat2
      The latitude of the second location.
    lon2
      The longitude of the second location.
    tol
      Tolerance for checking proximity to poles and rounding to zero.

    The bearing (AKA the position angle, PA) is the orientation of point 2
    with regards to point 1 relative to the longitudinal axis. Returns the
    bearing in radians. All arguments are in radians as well. The arguments
    may be vectors.

    Note that the ordering of the arguments maps to the nonstandard ordering
    ``(Dec, RA)`` in equatorial coordinates. In a spherical projection it maps
    to ``(Y, X)`` which may also be unexpected.

    The sign convention is astronomical: bearings range from -π to π, with
    negative values if point 2 is in the western hemisphere with regards to
    point 1, positive if it is in the eastern. (That is, “east from north”.)
    If point 1 is very near the pole, the bearing is undefined and the result
    is NaN.

    The *tol* argument is used for checking proximity to the poles and for
    rounding the bearing to precisely zero if it's extremely small.

    Derived from ``bear()`` in `angles.py from Prasanth Nair
    <https://github.com/phn/angles>`_. His version is BSD licensed. This one
    is sufficiently different that I think it counts as a separate
    implementation.

    """
    # cross product on outer axis:
    ocross = lambda a, b: np.cross(a, b, axisa=0, axisb=0, axisc=0)

    # if args have shape S, this has shape (3, S)
    v1 = np.asarray(
        [np.cos(lat1) * np.cos(lon1), np.cos(lat1) * np.sin(lon1), np.sin(lat1)]
    )

    v2 = np.asarray(
        [np.cos(lat2) * np.cos(lon2), np.cos(lat2) * np.sin(lon2), np.sin(lat2)]
    )

    is_bad = (v1[0] ** 2 + v1[1] ** 2) < tol

    p12 = ocross(v1, v2)  # ~"perpendicular to great circle containing points"
    p1z = np.asarray([v1[1], -v1[0], np.zeros_like(lat1)])  # ~"perp to base and Z axis"
    cm = np.sqrt((ocross(p12, p1z) ** 2).sum(axis=0))  # ~"angle between the vectors"
    bearing = np.arctan2(cm, np.sum(p12 * p1z, axis=0))
    bearing = np.where(p12[2] < 0, -bearing, bearing)  # convert to [-pi/2, pi/2]
    bearing = np.where(np.abs(bearing) < tol, 0, bearing)  # clamp
    bearing[np.where(is_bad)] = np.nan
    return bearing


def sphofs(lat1, lon1, r, pa, tol=1e-2, rmax=None):
    """Offset from one location on the sphere to another.

    This function is given a start location, expressed as a latitude and
    longitude, a distance to offset, and a direction to offset (expressed as a
    bearing, AKA position angle). It uses these to compute a final location.
    This function mirrors :func:`sphdist` and :func:`sphbear` such that::

      # If:
      r = sphdist (lat1, lon1, lat2a, lon2a)
      pa = sphbear (lat1, lon1, lat2a, lon2a)
      lat2b, lon2b = sphofs (lat1, lon1, r, pa)
      # Then lat2b = lat2a and lon2b = lon2a

    Arguments are:

    lat1
      The latitude of the start location.
    lon1
      The longitude of the start location.
    r
      The distance to offset by.
    pa
      The position angle (“PA” or bearing) to offset towards.
    tol
      The tolerance for the accuracy of the calculation.
    rmax
      The maximum allowed offset distance.

    Returns a pair ``(lat2, lon2)``. All arguments and the return values are
    measured in radians. The arguments may be vectors. The PA sign convention
    is astronomical, measuring orientation east from north.

    Note that the ordering of the arguments and return values maps to the
    nonstandard ordering ``(Dec, RA)`` in equatorial coordinates. In a
    spherical projection it maps to ``(Y, X)`` which may also be unexpected.

    The offset is computed naively as::

      lat2 = lat1 + r * cos (pa)
      lon2 = lon1 + r * sin (pa) / cos (lat2)

    This will fail for large offsets. Error checking can be done in two ways.
    If *tol* is not None, :func:`sphdist` is used to calculate the actual
    distance between the two locations, and if the magnitude of the fractional
    difference between that and *r* is larger than *tol*, :exc:`ValueError` is
    raised. This will add an overhead to the computation that may be
    significant if you're going to be calling this function a lot.

    Additionally, if *rmax* is not None, magnitudes of *r* greater than *rmax*
    are rejected. For reference, an *r* of 0.2 (~11 deg) gives a maximum
    fractional distance error of ~3%.

    """
    if rmax is not None and np.abs(r) > rmax:
        raise ValueError(
            "sphofs radius value %f is too big for " "our approximation" % r
        )

    lat2 = lat1 + r * np.cos(pa)
    lon2 = lon1 + r * np.sin(pa) / np.cos(lat2)

    if tol is not None:
        s = sphdist(lat1, lon1, lat2, lon2)
        if np.any(np.abs((s - r) / s) > tol):
            raise ValueError(
                "sphofs approximation broke down "
                "(%s %s %s %s %s %s %s)" % (lat1, lon1, lat2, lon2, r, s, pa)
            )

    return lat2, lon2


# Spherical trig tools that are more astronomy-specific. Note that precise
# positional calculations should generally use skyfield.


def parang(hourangle, declination, latitude):
    """Calculate the parallactic angle of a sky position.

    This computes the parallactic angle of a sky position expressed in terms
    of an hour angle and declination. Arguments:

    hourangle
      The hour angle of the location on the sky.
    declination
      The declination of the location on the sky.
    latitude
      The latitude of the observatory.

    Inputs and outputs are all in radians. Implementation adapted from GBTIDL
    ``parangle.pro``.

    """
    return -np.arctan2(
        -np.sin(hourangle),
        np.cos(declination) * np.tan(latitude)
        - np.sin(declination) * np.cos(hourangle),
    )


# 2D Gaussian (de)convolution


def gaussian_convolve(maj1, min1, pa1, maj2, min2, pa2):
    """Convolve two Gaussians analytically.

    Given the shapes of two 2-dimensional Gaussians, this function returns
    the shape of their convolution.

    Arguments:

    maj1
      Major axis of input Gaussian 1.
    min1
      Minor axis of input Gaussian 1.
    pa1
      Orientation angle of input Gaussian 1, in radians.
    maj2
      Major axis of input Gaussian 2.
    min2
      Minor axis of input Gaussian 2.
    pa2
      Orientation angle of input Gaussian 2, in radians.

    The return value is ``(maj3, min3, pa3)``, with the same format as the
    input arguments. The axes can be measured in any units, so long as they're
    consistent.

    Implementation copied from MIRIAD’s ``gaufac``.

    """
    c1 = np.cos(pa1)
    s1 = np.sin(pa1)
    c2 = np.cos(pa2)
    s2 = np.sin(pa2)

    a = (maj1 * c1) ** 2 + (min1 * s1) ** 2 + (maj2 * c2) ** 2 + (min2 * s2) ** 2
    b = (maj1 * s1) ** 2 + (min1 * c1) ** 2 + (maj2 * s2) ** 2 + (min2 * c2) ** 2
    g = 2 * ((min1**2 - maj1**2) * s1 * c1 + (min2**2 - maj2**2) * s2 * c2)

    s = a + b
    t = np.sqrt((a - b) ** 2 + g**2)
    maj3 = np.sqrt(0.5 * (s + t))
    min3 = np.sqrt(0.5 * (s - t))

    if abs(g) + abs(a - b) == 0:
        pa3 = 0.0
    else:
        pa3 = 0.5 * np.arctan2(-g, a - b)

    # "Amplitude of the resulting Gaussian":
    # f = pi / (4 * np.log (2)) * maj1 * min1 * maj2 * min2 \
    #    / np.sqrt (a * b - 0.25 * g**2)

    return maj3, min3, pa3


def gaussian_deconvolve(smaj, smin, spa, bmaj, bmin, bpa):
    """Deconvolve two Gaussians analytically.

    Given the shapes of 2-dimensional “source” and “beam” Gaussians, this
    returns a deconvolved “result” Gaussian such that the convolution of
    “beam” and “result” is “source”.

    Arguments:

    smaj
      Major axis of source Gaussian.
    smin
      Minor axis of source Gaussian.
    spa
      Orientation angle of source Gaussian, in radians.
    bmaj
      Major axis of beam Gaussian.
    bmin
      Minor axis of beam Gaussian.
    bpa
      Orientation angle of beam Gaussian, in radians.

    The return value is ``(rmaj, rmin, rpa, status)``. The first three values
    have the same format as the input arguments. The *status* result is one of
    "ok", "pointlike", or "fail". A "pointlike" status indicates that the
    source and beam shapes are difficult to distinguish; a "fail" status
    indicates that the two shapes seem to be mutually incompatible (e.g.,
    source and beam are very narrow and orthogonal).

    The axes can be measured in any units, so long as they're consistent.

    Ideally if::

      rmaj, rmin, rpa, status = gaussian_deconvolve (smaj, smin, spa, bmaj, bmin, bpa)

    then::

      smaj, smin, spa = gaussian_convolve (rmaj, rmin, rpa, bmaj, bmin, bpa)

    Implementation derived from MIRIAD’s ``gaudfac``. This function currently
    doesn't do a great job of dealing with pointlike sources, i.e. ones where
    “source” and “beam” are nearly indistinguishable.

    """
    # I've added extra code to ensure ``smaj >= bmaj``, ``smin >= bmin``, and
    # increased the coefficient in front of "limit" from 0.1 to 0.5. Feel a
    # little wary about that first change.

    from numpy import cos, sin, sqrt, min, abs, arctan2

    if smaj < bmaj:
        smaj = bmaj
    if smin < bmin:
        smin = bmin

    alpha = (
        (smaj * cos(spa)) ** 2
        + (smin * sin(spa)) ** 2
        - (bmaj * cos(bpa)) ** 2
        - (bmin * sin(bpa)) ** 2
    )
    beta = (
        (smaj * sin(spa)) ** 2
        + (smin * cos(spa)) ** 2
        - (bmaj * sin(bpa)) ** 2
        - (bmin * cos(bpa)) ** 2
    )
    gamma = 2 * (
        (smin**2 - smaj**2) * sin(spa) * cos(spa)
        - (bmin**2 - bmaj**2) * sin(bpa) * cos(bpa)
    )

    s = alpha + beta
    t = sqrt((alpha - beta) ** 2 + gamma**2)
    limit = 0.5 * min([smaj, smin, bmaj, bmin]) ** 2
    status = "ok"

    if alpha < 0 or beta < 0 or s < t:
        dmaj = dmin = dpa = 0

        if 0.5 * (s - t) < limit and alpha > -limit and beta > -limit:
            status = "pointlike"
        else:
            status = "fail"
    else:
        dmaj = sqrt(0.5 * (s + t))
        dmin = sqrt(0.5 * (s - t))

        if abs(gamma) + abs(alpha - beta) == 0:
            dpa = 0
        else:
            dpa = 0.5 * arctan2(-gamma, alpha - beta)

    return dmaj, dmin, dpa, status


# Given astrometric properties of a source, predict its position *with
# uncertainties* at a given date through Monte Carlo simulations with
# skyfield.


def load_skyfield_data():
    """Load data files used in Skyfield. This will download files from the
    internet if they haven't been downloaded before.

    Skyfield downloads files to the current directory by default, which is not
    ideal. Here we abuse astropy and use its cache directory to cache the data
    files per-user. If we start downloading files in other places in pwkit we
    should maybe make this system more generic. And the dep on astropy is not
    at all necessary.

    Skyfield will print out a progress bar as it downloads things.

    Returns ``(planets, ts)``, the standard Skyfield ephemeris and timescale
    data files.

    """
    import os.path
    from astropy.config import paths
    from skyfield.api import Loader

    cache_dir = os.path.join(paths.get_cache_dir(), "pwkit")
    loader = Loader(cache_dir)
    planets = loader("de421.bsp")
    ts = loader.timescale()
    return planets, ts


# Hack to implement epochs-of-position. For what it's worth, Skyfield is
# MIT-licensed like us.

try:
    from skyfield.api import Star, T0
except ImportError:

    def PromoEpochStar(**kwargs):
        raise NotImplementedError(
            'the "skyfield" package is required for this functionality'
        )

else:

    class PromoEpochStar(Star):
        """A customized version of the Skyfield Star class that accepts a new
        epoch-of-position parameter.

        Derived from the Skyfield source as of commit 49c2467b (2018 Mar 28).

        """

        def __init__(self, jd_of_position=T0, **kwargs):
            super(PromoEpochStar, self).__init__(**kwargs)
            self.jd_of_position = jd_of_position

        def __repr__(self):
            opts = []
            for (
                name
            ) in "ra_mas_per_year dec_mas_per_year parallax_mas radial_km_per_s jd_of_position names".split():
                value = getattr(self, name)
                if value:
                    opts.append(", {0}={1!r}".format(name, value))
            return "PromoEpochStar(ra_hours={0!r}, dec_degrees={1!r}{2})".format(
                self.ra.hours, self.dec.degrees, "".join(opts)
            )

        def _observe_from_bcrs(self, observer):
            from numpy import outer
            from skyfield.constants import C_AUDAY
            from skyfield.functions import length_of
            from skyfield.relativity import light_time_difference

            position, velocity = self._position_au, self._velocity_au_per_d
            t = observer.t
            dt = light_time_difference(position, observer.position.au)
            if t.shape:
                position = (
                    outer(velocity, t.tdb + dt - self.jd_of_position).T + position
                ).T
            else:
                position = position + velocity * (t.tdb + dt - self.jd_of_position)
            vector = position - observer.position.au
            distance = length_of(vector)
            light_time = distance / C_AUDAY
            return vector, (observer.velocity.au_per_d.T - velocity).T, t, light_time

    del Star, T0


_vizurl = "http://vizier.u-strasbg.fr/viz-bin/asu-tsv"


def get_2mass_epoch(tmra, tmdec, debug=False):
    """Given a 2MASS position, look up the epoch when it was observed.

    This function uses the CDS Vizier web service to look up information in
    the 2MASS point source database. Arguments are:

    tmra
      The source's J2000 right ascension, in radians.
    tmdec
      The source's J2000 declination, in radians.
    debug
      If True, the web server's response will be printed to :data:`sys.stdout`.

    The return value is an MJD. If the lookup fails, a message will be printed
    to :data:`sys.stderr` (unconditionally!) and the :data:`J2000` epoch will
    be returned.

    """
    import codecs

    try:
        from urllib.request import urlopen
    except ImportError:
        from urllib2 import urlopen
    postdata = b"""-mime=csv
-source=2MASS
-out=_q,JD
-c=%.6f %.6f
-c.eq=J2000""" % (
        tmra * R2D,
        tmdec * R2D,
    )

    jd = None

    for line in codecs.getreader("utf-8")(urlopen(_vizurl, postdata)):
        line = line.strip()
        if debug:
            print_("D: 2M >>", line)

        if line.startswith("1;"):
            jd = float(line[2:])

    if jd is None:
        import sys

        print_(
            "warning: 2MASS epoch lookup failed; astrometry could be very wrong!",
            file=sys.stderr,
        )
        return J2000

    return jd - 2400000.5


_simbadbase = "http://simbad.u-strasbg.fr/simbad/sim-script?script="
_simbaditems = (
    "COO(d;A) COO(d;D) COO(E) COO(B) PM(A) PM(D) PM(E) PLX(V) PLX(E) " "RV(V) RV(E)"
).split()


def get_simbad_astrometry_info(ident, items=_simbaditems, debug=False):
    """Fetch astrometric information from the Simbad web service.

    Given the name of a source as known to the CDS Simbad service, this
    function looks up its positional information and returns it in a
    dictionary. In most cases you should use an :class:`AstrometryInfo` object
    and its :meth:`~AstrometryInfo.fill_from_simbad` method instead of this
    function.

    Arguments:

    ident
      The Simbad name of the source to look up.
    items
      An iterable of data items to look up. The default fetches position,
      proper motion, parallax, and radial velocity information. Each item name
      resembles the string ``COO(d;A)`` or ``PLX(E)``. The allowed formats are
      defined `on this CDS page
      <http://simbad.u-strasbg.fr/Pages/guide/sim-fscript.htx>`_.
    debug
      If true, the response from the webserver will be printed.

    The return value is a dictionary with a key corresponding to the textual
    result returned for each requested item.

    """
    import codecs

    try:
        from urllib.parse import quote
    except ImportError:
        from urllib import quote
    try:
        from urllib.request import urlopen
    except ImportError:
        from urllib2 import urlopen

    s = "\\n".join("%s %%%s" % (i, i) for i in items)
    s = """output console=off script=off
format object "%s"
query id %s""" % (
        s,
        ident,
    )
    url = _simbadbase + quote(s)
    results = {}
    errtext = None

    for line in codecs.getreader("utf-8")(urlopen(url)):
        line = line.strip()
        if debug:
            print_("D: SA >>", line)

        if errtext is not None:
            errtext += line
        elif line.startswith("::error"):
            errtext = ""
        elif len(line):
            k, v = line.split(" ", 1)
            results[k] = v

    if errtext is not None:
        raise Exception("SIMBAD query error: " + errtext)
    return results


class AstrometryInfo(object):
    """Holds astrometric data and their uncertainties, and can predict
    positions with uncertainties.

    """

    ra = None
    "The J2000 right ascension of the object, measured in radians."

    dec = None
    "The J2000 declination of the object, measured in radians."

    pos_u_maj = None
    "Major axis of the error ellipse for the object position, in radians."

    pos_u_min = None
    "Minor axis of the error ellipse for the object position, in radians."

    pos_u_pa = None
    """Position angle (really orientation) of the error ellipse for the object
    position, east from north, in radians.

    """
    pos_epoch = None
    """The epoch of position, that is, the date when the position was measured, in
    MJD[TT].

    """
    promo_ra = None
    """The proper motion in right ascension, in milliarcsec per year. XXX:
    cos(dec) confusion!

    """
    promo_dec = None
    """The object's proper motion in declination, in milliarcsec per year."""

    promo_u_maj = None
    """Major axis of the error ellipse for the object's proper motion, in
    milliarcsec per year.

    """
    promo_u_min = None
    """Minor axis of the error ellipse for the object's proper motion, in
    milliarcsec per year.

    """
    promo_u_pa = None
    """Position angle (really orientation) of the error ellipse for the object
    proper motion, east from north, in radians.

    """
    parallax = None
    "The object's parallax, in milliarcsec."

    u_parallax = None
    "Uncertainty in the object's parallax, in milliarcsec."

    vradial = None
    "The object's radial velocity, in km/s. NOTE: not relevant in our usage."

    u_vradial = None
    """The uncertainty in the object's radial velocity, in km/s. NOTE: not
    relevant in our usage.

    """

    def __init__(self, simbadident=None, **kwargs):
        if simbadident is not None:
            self.fill_from_simbad(simbadident)

        for k, v in kwargs.items():
            setattr(self, k, v)

    def _partial_info(self, val0, *rest):
        if not len(rest):
            return False

        first = val0 is None
        for v in rest:
            if (v is None) != first:
                return True
        return False

    def verify(self, complain=True):
        """Validate that the attributes are self-consistent.

        This function does some basic checks of the object attributes to
        ensure that astrometric calculations can legally be performed. If the
        *complain* keyword is true, messages may be printed to
        :data:`sys.stderr` if non-fatal issues are encountered.

        Returns *self*.

        """
        import sys

        if self.ra is None:
            raise ValueError('AstrometryInfo missing "ra"')
        if self.dec is None:
            raise ValueError('AstrometryInfo missing "dec"')

        if self._partial_info(self.promo_ra, self.promo_dec):
            raise ValueError("partial proper-motion info in AstrometryInfo")

        if self._partial_info(self.pos_u_maj, self.pos_u_min, self.pos_u_pa):
            raise ValueError("partial positional uncertainty info in AstrometryInfo")

        if self._partial_info(self.promo_u_maj, self.promo_u_min, self.promo_u_pa):
            raise ValueError("partial proper-motion uncertainty info in AstrometryInfo")

        if self.pos_u_maj is None:
            if complain:
                print_(
                    "AstrometryInfo: no positional uncertainty info", file=sys.stderr
                )
        elif self.pos_u_maj < self.pos_u_min:
            # Based on experience with PM, this may be possible
            if complain:
                print_(
                    "AstrometryInfo: swapped positional uncertainty "
                    "major/minor axes",
                    file=sys.stderr,
                )
            self.pos_u_maj, self.pos_u_min = self.pos_u_min, self.pos_u_maj
            self.pos_u_pa += 0.5 * np.pi

        if self.pos_epoch is None:
            if complain:
                print_(
                    "AstrometryInfo: assuming epoch of position is J2000.0",
                    file=sys.stderr,
                )

        if self.promo_ra is None:
            if complain:
                print_("AstrometryInfo: assuming zero proper motion", file=sys.stderr)
        elif self.promo_u_maj is None:
            if complain:
                print_(
                    "AstrometryInfo: no uncertainty on proper motion", file=sys.stderr
                )
        elif self.promo_u_maj < self.promo_u_min:
            # I've seen this: V* V374 Peg
            if complain:
                print_(
                    "AstrometryInfo: swapped proper motion uncertainty "
                    "major/minor axes",
                    file=sys.stderr,
                )
            self.promo_u_maj, self.promo_u_min = self.promo_u_min, self.promo_u_maj
            self.promo_u_pa += 0.5 * np.pi

        if self.parallax is None:
            if complain:
                print_("AstrometryInfo: assuming zero parallax", file=sys.stderr)
        else:
            if self.parallax < 0.0:
                raise ValueError("negative parallax in AstrometryInfo")
            if self.u_parallax is None:
                if complain:
                    print_(
                        "AstrometryInfo: no uncertainty on parallax", file=sys.stderr
                    )

        if self.vradial is None:
            pass  # not worth complaining
        elif self.u_vradial is None:
            if complain:
                print_("AstrometryInfo: no uncertainty on v_radial", file=sys.stderr)

        return self  # chain-friendly

    def predict_without_uncertainties(self, mjd, complain=True):
        """Predict the object position at a given MJD.

        The return value is a tuple ``(ra, dec)``, in radians, giving the
        predicted position of the object at *mjd*. Unlike :meth:`predict`, the
        astrometric uncertainties are ignored. This function is therefore
        deterministic but potentially misleading.

        If *complain* is True, print out warnings for incomplete information.

        This function relies on the external :mod:`skyfield` package.

        """
        import sys

        self.verify(complain=complain)

        planets, ts = load_skyfield_data()  # might download stuff from the internet
        earth = planets["earth"]
        t = ts.tdb(jd=mjd + 2400000.5)

        # "Best" position. The implementation here is a bit weird to keep
        # parallelism with predict().

        args = {
            "ra_hours": self.ra * R2H,
            "dec_degrees": self.dec * R2D,
        }

        if self.pos_epoch is not None:
            args["jd_of_position"] = self.pos_epoch + 2400000.5

        if self.promo_ra is not None:
            args["ra_mas_per_year"] = self.promo_ra
            args["dec_mas_per_year"] = self.promo_dec
        if self.parallax is not None:
            args["parallax_mas"] = self.parallax
        if self.vradial is not None:
            args["radial_km_per_s"] = self.vradial

        bestra, bestdec, _ = earth.at(t).observe(PromoEpochStar(**args)).radec()
        return bestra.radians, bestdec.radians

    def predict(self, mjd, complain=True, n=20000):
        """Predict the object position at a given MJD.

        The return value is a tuple ``(ra, dec, major, minor, pa)``, all in
        radians. These are the predicted position of the object and its
        uncertainty at *mjd*. If *complain* is True, print out warnings for
        incomplete information. *n* is the number of Monte Carlo samples to
        draw for computing the positional uncertainty.

        The uncertainty ellipse parameters are sigmas, not FWHM. These may be
        converted with the :data:`S2F` constant.

        This function relies on the external :mod:`skyfield` package.

        """
        import sys
        from . import ellipses

        self.verify(complain=complain)

        planets, ts = load_skyfield_data()  # might download stuff from the internet
        earth = planets["earth"]
        t = ts.tdb(jd=mjd + 2400000.5)

        # "Best" position.

        args = {
            "ra_hours": self.ra * R2H,
            "dec_degrees": self.dec * R2D,
        }

        if self.pos_epoch is not None:
            args["jd_of_position"] = self.pos_epoch + 2400000.5

        if self.promo_ra is not None:
            args["ra_mas_per_year"] = self.promo_ra
            args["dec_mas_per_year"] = self.promo_dec
        if self.parallax is not None:
            args["parallax_mas"] = self.parallax
        if self.vradial is not None:
            args["radial_km_per_s"] = self.vradial

        bestra, bestdec, _ = earth.at(t).observe(PromoEpochStar(**args)).radec()
        bestra = bestra.radians
        bestdec = bestdec.radians

        # Monte Carlo to get an uncertainty. As always, astronomy position
        # angle convention requires that we treat declination as X and RA as
        # Y. First, we check sanity and generate randomized parameters:

        if (
            self.pos_u_maj is None
            and self.promo_u_maj is None
            and self.u_parallax is None
        ):
            if complain:
                print_(
                    "AstrometryInfo.predict(): no uncertainties "
                    "available; cannot Monte Carlo!",
                    file=sys.stderr,
                )
            return (bestra, bestdec, 0.0, 0.0, 0.0)

        if self.pos_u_maj is not None:
            sd, sr, cdr = ellipses.ellbiv(self.pos_u_maj, self.pos_u_min, self.pos_u_pa)
            decs, ras = ellipses.bivrandom(self.dec, self.ra, sd, sr, cdr, n).T
        else:
            ras = np.zeros(n) + self.ra
            decs = np.zeros(n) + self.dec

        if self.promo_ra is None:
            pmras = np.zeros(n)
            pmdecs = np.zeros(n)
        elif self.promo_u_maj is not None:
            sd, sr, cdr = ellipses.ellbiv(
                self.promo_u_maj, self.promo_u_min, self.promo_u_pa
            )
            pmdecs, pmras = ellipses.bivrandom(
                self.promo_dec, self.promo_ra, sd, sr, cdr, n
            ).T
        else:
            pmras = np.zeros(n) + self.promo_ra
            pmdecs = np.zeros(n) + self.promo_dec

        if self.parallax is None:
            parallaxes = np.zeros(n)
        elif self.u_parallax is not None:
            parallaxes = np.random.normal(self.parallax, self.u_parallax, n)
        else:
            parallaxes = np.zeros(n) + self.parallax

        if self.vradial is None:
            vradials = np.zeros(n)
        elif self.u_vradial is not None:
            vradials = np.random.normal(self.vradial, self.u_vradial, n)
        else:
            vradials = np.zeros(n) + self.vradial

        # Now we compute the positions and summarize as an ellipse:

        results = np.empty((n, 2))

        for i in range(n):
            args["ra_hours"] = ras[i] * R2H
            args["dec_degrees"] = decs[i] * R2D
            args["ra_mas_per_year"] = pmras[i]
            args["dec_mas_per_year"] = pmdecs[i]
            args["parallax_mas"] = parallaxes[i]
            args["radial_km_per_s"] = vradials[i]
            ara, adec, _ = earth.at(t).observe(PromoEpochStar(**args)).radec()
            results[i] = adec.radians, ara.radians

        maj, min, pa = ellipses.bivell(*ellipses.databiv(results))

        # All done.

        return bestra, bestdec, maj, min, pa

    def print_prediction(self, ptup, precision=2):
        """Print a summary of a predicted position.

        The argument *ptup* is a tuple returned by :meth:`predict`. It is
        printed to :data:`sys.stdout` in a reasonable format that uses Unicode
        characters.

        """
        from . import ellipses

        bestra, bestdec, maj, min, pa = ptup

        f = ellipses.sigmascale(1)
        maj *= R2A
        min *= R2A
        pa *= R2D

        print_("position =", fmtradec(bestra, bestdec, precision=precision))
        print_(
            'err(1σ)  = %.*f" × %.*f" @ %.0f°'
            % (precision, maj * f, precision, min * f, pa)
        )

    def fill_from_simbad(self, ident, debug=False):
        """Fill in astrometric information using the Simbad web service.

        This uses the CDS Simbad web service to look up astrometric
        information for the source name *ident* and fills in attributes
        appropriately. Values from Simbad are not always reliable.

        Returns *self*.

        """
        info = get_simbad_astrometry_info(ident, debug=debug)
        posref = "unknown"

        for k, v in info.items():
            if "~" in v:
                continue  # no info

            if k == "COO(d;A)":
                self.ra = float(v) * D2R
            elif k == "COO(d;D)":
                self.dec = float(v) * D2R
            elif k == "COO(E)":
                a = v.split()
                self.pos_u_maj = float(a[0]) * A2R * 1e-3  # mas -> rad
                self.pos_u_min = float(a[1]) * A2R * 1e-3
                self.pos_u_pa = float(a[2]) * D2R
            elif k == "COO(B)":
                posref = v
            elif k == "PM(A)":
                self.promo_ra = float(v)  # mas/yr
            elif k == "PM(D)":
                self.promo_dec = float(v)  # mas/yr
            elif k == "PM(E)":
                a = v.split()
                self.promo_u_maj = float(a[0])  # mas/yr
                self.promo_u_min = float(a[1])
                self.promo_u_pa = float(a[2]) * D2R  # rad!
            elif k == "PLX(V)":
                self.parallax = float(v)  # mas
            elif k == "PLX(E)":
                self.u_parallax = float(v)  # mas
            elif k == "RV(V)":
                self.vradial = float(v)  # km/s
            elif k == "RV(E)":
                self.u_vradial = float(v)  # km/s

        if self.ra is None:
            raise Exception('no position returned by Simbad for "%s"' % ident)
        if self.u_parallax == 0:
            self.u_parallax = None
        if self.u_vradial == 0:
            self.u_vradial = None

        # Get the right epoch of position when possible

        if posref == "2003yCat.2246....0C":
            self.pos_epoch = get_2mass_epoch(self.ra, self.dec, debug)
        elif posref == "2018yCat.1345....0G":
            self.pos_epoch = 57205.875  # J2015.5 for Gaia DR2

        return self  # eases chaining

    def fill_from_allwise(self, ident, catalog_ident="II/328/allwise"):
        """Fill in astrometric information from the AllWISE catalog using Astroquery.

        This uses the :mod:`astroquery` module to query the AllWISE
        (2013wise.rept....1C) source catalog through the Vizier
        (2000A&AS..143...23O) web service. It then fills in the instance with
        the relevant information. Arguments are:

        ident
          The AllWISE catalog identifier of the form ``"J112254.70+255021.9"``.
        catalog_ident
          The Vizier designation of the catalog to query. The default is
          "II/328/allwise", the current version of the AllWISE catalog.

        Raises :exc:`~pwkit.PKError` if something unexpected happens that
        doesn't itself result in an exception within :mod:`astroquery`.

        You should probably prefer :meth:`fill_from_simbad` for objects that
        are known to the CDS Simbad service, but not all objects in the
        AllWISE catalog are so known.

        If you use this function, you should `acknowledge AllWISE
        <https://wise2.ipac.caltech.edu/docs/release/allwise/>`_ and `Vizier
        <http://cds.u-strasbg.fr/vizier-org/licences_vizier.html>`_.

        Returns *self*.

        """
        from astroquery.vizier import Vizier
        import numpy.ma.core as ma_core

        # We should match exactly one table and one row within that table, but
        # for robustness we ignore additional results if they happen to
        # appear. Strangely, querying for an invalid identifier yields a table
        # with two rows that are filled with masked out data.

        table_list = Vizier.query_constraints(catalog=catalog_ident, AllWISE=ident)
        if not len(table_list):
            raise PKError(
                "Vizier query returned no tables (catalog=%r AllWISE=%r)",
                catalog_ident,
                ident,
            )

        table = table_list[0]
        if not len(table):
            raise PKError(
                "Vizier query returned empty %s table (catalog=%r AllWISE=%r)",
                table.meta["name"],
                catalog_ident,
                ident,
            )

        row = table[0]
        if isinstance(row["_RAJ2000"], ma_core.MaskedConstant):
            raise PKError(
                "Vizier query returned flagged row in %s table; your AllWISE "
                "identifier likely does not exist (it should be of the form "
                '"J112254.70+255021.9"; catalog=%r AllWISE=%r)',
                table.meta["name"],
                catalog_ident,
                ident,
            )

        # OK, we can actually do this.

        self.ra = row["RA_pm"] * D2R
        self.dec = row["DE_pm"] * D2R

        if row["e_RA_pm"] > row["e_DE_pm"]:
            self.pos_u_maj = row["e_RA_pm"] * A2R
            self.pos_u_min = row["e_DE_pm"] * A2R
            self.pos_u_pa = halfpi
        else:
            self.pos_u_maj = row["e_DE_pm"] * A2R
            self.pos_u_min = row["e_RA_pm"] * A2R
            self.pos_u_pa = 0

        self.pos_epoch = 55400.0  # hardcoded in the catalog
        self.promo_ra = row["pmRA"]
        self.promo_dec = row["pmDE"]

        if row["e_pmRA"] > row["e_pmDE"]:
            self.promo_u_maj = row["e_pmRA"] * 1.0
            self.promo_u_min = row["e_pmDE"] * 1.0
            self.promo_u_pa = halfpi
        else:
            self.promo_u_maj = row["e_pmDE"] * 1.0
            self.promo_u_min = row["e_pmRA"] * 1.0
            self.promo_u_pa = 0.0

        return self  # eases chaining

    def __unicode__(self):
        self.verify(complain=False)
        a = []
        a.append("Position: " + fmtradec(self.ra, self.dec))
        if self.pos_u_maj is None:
            a.append("No uncertainty info for position.")
        else:
            a.append(
                'Pos. uncert: %.3f" × %.3f" @ %.0f°'
                % (self.pos_u_maj * R2A, self.pos_u_min * R2A, self.pos_u_pa * R2D)
            )
        if self.pos_epoch is None:
            a.append("No epoch of position.")
        else:
            a.append("Epoch of position: MJD %.3f" % self.pos_epoch)
        if self.promo_ra is None:
            a.append("No proper motion.")
        else:
            a.append(
                "Proper motion: %.3f, %.3f mas/yr" % (self.promo_ra, self.promo_dec)
            )
        if self.promo_u_maj is None:
            a.append("No uncertainty info for proper motion.")
        else:
            a.append(
                "Promo. uncert: %.1f × %.1f mas/yr @ %.0f°"
                % (self.promo_u_maj, self.promo_u_min, self.promo_u_pa * R2D)
            )
        if self.parallax is None:
            a.append("No parallax information.")
        elif self.u_parallax is not None:
            a.append("Parallax: %.1f ± %.1f mas" % (self.parallax, self.u_parallax))
        else:
            a.append("Parallax: %.1f mas, unknown uncert." % self.parallax)
        if self.vradial is None:
            a.append("No radial velocity information.")
        elif self.u_vradial is not None:
            a.append(
                "Radial velocity: %.2f ± %.2f km/s" % (self.vradial, self.u_vradial)
            )
        else:
            a.append("Radial velocity: %.1f km/s, unknown uncert." % self.vradial)
        return "\n".join(a)

    __str__ = unicode_to_str


# Other astronomical calculations


def app2abs(app_mag, dist_pc):
    """Convert an apparent magnitude to an absolute magnitude, given a source's
    (luminosity) distance in parsecs.

    Arguments:

    app_mag
      Apparent magnitude.
    dist_pc
      Distance, in parsecs.

    Returns the absolute magnitude. The arguments may be vectors.

    """
    return app_mag - 5 * (np.log10(dist_pc) - 1)


def abs2app(abs_mag, dist_pc):
    """Convert an absolute magnitude to an apparent magnitude, given a source's
    (luminosity) distance in parsecs.

    Arguments:

    abs_mag
      Absolute magnitude.
    dist_pc
      Distance, in parsecs.

    Returns the apparent magnitude. The arguments may be vectors.

    """
    return abs_mag + 5 * (np.log10(dist_pc) - 1)
