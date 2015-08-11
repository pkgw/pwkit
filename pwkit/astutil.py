# -*- mode: python; coding: utf-8 -*-
# Copyright 2012-2014 Peter Williams <peter@newton.cx> and collaborators.
# Licensed under the MIT License.

"""pwkit.astutil - miscellaneous astronomical constants and utilities

Constants:

np                  - The numpy module.
pi                  - Pi.
twopi               - 2 * Pi.
halfpi              - 0.5 * Pi.
R2A                 - arcsecs = radians * R2A
A2R                 - radians = arcsecs * A2R
R2D                 - degrees = radians * R2D
D2R                 - radians = degrees * D2R
R2H                 - hours = radians * R2H
H2R                 - radians = hours * H2R
F2S                 - sigma = fwhm * F2S (for Gaussians)
S2F                 - fwhm = sigma * S2F
J2000               - J2000 as an MJD: 51544.5

Functions:

angcen              - Center an angle in radians in [-π, π].
fmtdeglat           - Format radian latitude (dec) as sexagesimal degrees.
fmtdeglon           - Format radian longitude as sexagesimal degrees.
fmthours            - Format radian longitude (RA) as sexagesimal hours.
fmtradec            - Format radian ra/dec as text.
gaussian_convolve   - Convolve a Gaussian profile with another.
gaussian_deconvolve - Deconvolve a Gaussian from profile from another.
orientcen           - Center an orientation in radians in [-π/2, π/2]
parang              - Parallactic angle from HA, dec, lat.
parsedeglat         - Parse sexagesimal degrees (dec) into a latitude.
parsedeglon         - Parse sexagesimal degrees into a longitude.
parsehours          - Parse sexagesimal hours (RA) into a longitude.
sphbear             - Calculate the bearing (~PA) from one lat/lon to another.
sphdist             - Calculate the distance between two lat/lons
sphofs              - Calculate lat/lon from an initial lat/lon and an offset.

Classes:

AstrometryInfo      - Hold astrometric parameters and predict a source location
                      with Monte Carlo uncertainties. (Requires precastro.)

"""
from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = b'''np pi twopi halfpi R2A A2R R2D D2R R2H H2R F2S S2F J2000 angcen orientcen
           fmthours fmtdeglon fmtdeglat fmtradec parsehours parsedeglat
           parsedeglon sphdist sphbear sphofs parang gaussian_convolve
           gaussian_deconvolve AstrometryInfo'''.split ()

import numpy as np

from . import text_type, unicode_to_str
from .numutil import broadcastize


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
F2S = 1 / np.sqrt (8 * np.log (2)) # FWHM to sigma
S2F = np.sqrt (8 * np.log (2))

J2000 = 51544.5


# Angle and orientation (PA) normalization
#
# PA's seem to usually be given in the range [-90, 90]

angcen = lambda a: (((a + pi) % twopi) - pi)
orientcen = lambda a: (((a + halfpi) % pi) - halfpi)


# Formatting/parsing of lat/long/etc

def _fmtsexagesimal (base, norm, basemax, seps, precision=3):
    if norm == 'none':
        pass
    elif norm == 'raise':
        if base > basemax or base < 0:
            raise ValueError ('illegal coordinate of %f' % base)
    elif norm == 'wrap':
        base = base % basemax
    else:
        raise ValueError ('unrecognized normalization type "%s"' % norm)

    if len (seps) < 2:
        # To ponder: we accept len(seps) > 3; seems OK.
        raise ValueError ('there must be at least two sexagesimal separators; '
                          'got value "%s"' % seps)

    precision = max (int (precision), 0)
    if precision == 0:
        width = 2
    else:
        width = precision + 3

    basewidth = len (text_type (basemax))

    bs = int (np.floor (base))
    min = int (np.floor ((base - bs) * 60))
    sec = round (3600 * (base - bs - min / 60.), precision)

    if sec >= 60:
        # Can happen if we round up
        sec -= 60
        min += 1

        if min >= 60:
            min -= 60
            bs += 1

            if bs >= basemax:
                bs -= basemax

    if len (seps) > 2:
        sep2 = seps[2]
    else:
        sep2 = ''

    return '%0*d%s%02d%s%0*.*f%s' % \
        (basewidth, bs, seps[0], min, seps[1], width, precision, sec, sep2)


def fmthours (radians, norm='wrap', precision=3, seps='::'):
    """(radians, norm='wrap', precision=3) -> string

    norm[alization] can be one of 'none', 'raise', or 'wrap'

    """
    return _fmtsexagesimal (radians * R2H, norm, 24, seps, precision=precision)


def fmtdeglon (radians, norm='wrap', precision=2, seps='::'):
    """(radians, norm='wrap', precision=2) -> string

    norm[alization] can be one of 'none', 'raise', or 'wrap'

    """
    return _fmtsexagesimal (radians * R2D, norm, 360, seps, precision=precision)


def fmtdeglat (radians, norm='raise', precision=2, seps='::'):
    """(radians, norm='raise', precision=2) -> string

    norm[alization] can be one of 'none', 'raise', or 'wrap'

    """
    if norm == 'none':
        pass
    elif norm == 'raise':
        if radians > halfpi or radians < -halfpi:
            raise ValueError ('illegal latitude of %f radians' % radians)
    elif norm == 'wrap':
        radians = angcen (radians)
        if radians > halfpi:
            radians = pi - radians
        elif radians < -halfpi:
            radians = -pi - radians
    else:
        raise ValueError ('unrecognized normalization type "%s"' % norm)

    if len (seps) < 2:
        # To ponder: we accept len(seps) > 3; seems OK.
        raise ValueError ('there must be at least two sexagesimal separators; '
                          'got value "%s"' % seps)

    precision = max (int (precision), 0)
    if precision == 0:
        width = 2
    else:
        width = precision + 3

    degrees = radians * R2D

    if degrees >= 0:
        sgn = '+'
    else:
        sgn = '-'
        degrees = -degrees

    deg = int (np.floor (degrees))
    amin = int (np.floor ((degrees - deg) * 60))
    asec = round (3600 * (degrees - deg - amin / 60.), precision)

    if asec >= 60:
        # Can happen if we round up
        asec -= 60
        amin += 1

        if amin >= 60:
            amin -= 60
            deg += 1

    if len (seps) > 2:
        sep2 = seps[2]
    else:
        sep2 = ''

    return '%s%02d%s%02d%s%0*.*f%s' % \
        (sgn, deg, seps[0], amin, seps[1], width, precision, asec, sep2)


def fmtradec (rarad, decrad, precision=2, raseps='::', decseps='::', intersep=' '):
    return (fmthours (rarad, precision=precision + 1, seps=raseps) +
            text_type (intersep) +
            fmtdeglat (decrad, precision=precision, seps=decseps))



# Parsing routines are currently very lame.

def _parsesexagesimal (sxgstr, desc, negok):
    sxgstr_orig = sxgstr
    sgn = 1

    if sxgstr[0] == '-':
        if negok:
            sgn = -1
            sxgstr = sxgstr[1:]
        else:
            raise ValueError ('illegal negative %s expression: %s' % (desc, sxgstr_orig))

    try:
        # TODO: other separators ...
        bs, mn, sec = sxgstr.split (':')
        bs = int (bs)
        mn = int (mn)
        sec = float (sec)
    except Exception:
        raise ValueError ('unable to parse as %s: %s' % (desc, sxgstr_orig))

    if mn < 0 or mn > 59 or sec < 0 or sec >= 60.:
        raise ValueError ('illegal sexagesimal %s expression: %s' % (desc, sxgstr_orig))
    if bs < 0: # two minus signs, or something
        raise ValueError ('illegal negative %s expression: %s' % (desc, sxgstr_orig))

    return sgn * (bs + mn / 60. + sec / 3600.)


def parsehours (hrstr):
    hr = _parsesexagesimal (hrstr, 'hours', False)
    if hr >= 24:
        raise ValueError ('illegal hour specification: ' + hrstr)
    return hr * H2R


def parsedeglat (latstr):
    deg = _parsesexagesimal (latstr, 'latitude', True)
    if abs (deg) > 90:
        raise ValueError ('illegal latitude specification: ' + latstr)
    return deg * D2R


def parsedeglon (lonstr):
    return _parsesexagesimal (lonstr, 'longitude', True) * D2R


# Spherical trig

@broadcastize (4)
def sphdist (lat1, lon1, lat2, lon2):
    """Args are: lat1, lon1, lat2, lon2 -- consistent with the usual coordinates
    in images, but note that this maps to (Dec, RA) or (Y, X), so be careful
    with this.

    The distance is computed with the "specialized Vincenty formula". Faster
    but more error-prone formulae are possible; see Wikipedia on Great-circle
    Distance.

    """
    cd = np.cos (lon2 - lon1)
    sd = np.sin (lon2 - lon1)
    c2 = np.cos (lat2)
    c1 = np.cos (lat1)
    s2 = np.sin (lat2)
    s1 = np.sin (lat1)
    a = np.sqrt ((c2 * sd)**2 + (c1 * s2 - s1 * c2 * cd)**2)
    b = s1 * s2 + c1 * c2 * cd
    return np.arctan2 (a, b)


@broadcastize (4)
def sphbear (lat1, lon1, lat2, lon2, tol=1e-15):
    """Args are (lat1, lon1, lat2, lon2, tol=1e-15) --
    consistent with the usual coordinates in images, but note that
    this maps to (Dec, RA) or (Y, X). All in radians. Returns the
    bearing (AKA position angle, PA) of point 2 with regards to point
    1.

    The sign convention is astronomical: bearing ranges from -pi to pi,
    with negative values if point 2 is in the western hemisphere w.r.t.
    point 1, positive if it is in the eastern.

    If point1 is very near the pole, the bearing is undefined and the result
    is NaN.

    tol is used for checking pole nearness and for rounding the bearing to
    precisely zero if it's extremely small.

    Derived from bear() in angles.py from Prasanth Nair,
    https://github.com/phn/angles . His version is BSD licensed. This one is
    sufficiently different that I think it counts as a separate
    implementation.

    """
    # cross product on outer axis:
    ocross = lambda a, b: np.cross (a, b, axisa=0, axisb=0, axisc=0)

    # if args have shape S, this has shape (3, S)
    v1 = np.asarray ([np.cos (lat1) * np.cos (lon1),
                      np.cos (lat1) * np.sin (lon1),
                      np.sin (lat1)])

    v2 = np.asarray ([np.cos (lat2) * np.cos (lon2),
                      np.cos (lat2) * np.sin (lon2),
                      np.sin (lat2)])

    is_bad = (v1[0]**2 + v1[1]**2) < tol

    p12 = ocross (v1, v2) # ~"perpendicular to great circle containing points"
    p1z = np.asarray ([v1[1], -v1[0], np.zeros_like (lat1)]) # ~"perp to base and Z axis"
    cm = np.sqrt ((ocross (p12, p1z)**2).sum (axis=0)) # ~"angle between the vectors"
    bearing = np.arctan2 (cm, np.sum (p12 * p1z, axis=0))
    bearing = np.where (p12[2] < 0, -bearing, bearing) # convert to [-pi/2, pi/2]
    bearing = np.where (np.abs (bearing) < tol, 0, bearing) # clamp
    bearing[np.where (is_bad)] = np.nan
    return bearing


def sphofs (lat1, lon1, r, pa, tol=1e-2, rmax=None):
    """Args are: lat1, lon1, r, pa -- consistent with
    the usual coordinates in images, but note that this maps
    to (Dec, RA) or (Y, X). PA is East from North. Returns
    lat2, lon2.

    Error checking can be done in two ways. If tol is not
    None, sphdist() is used to calculate the actual distance
    between the two locations, and if the magnitude of the
    fractional difference between that and *r* is larger than
    tol, an exception is raised. This will add an overhead
    to the computation that may be significant if you're
    going to be calling this function a whole lot.

    If rmax is not None, magnitudes of *r* greater than that
    value are rejected. For reference, an *r* of 0.2 (~11 deg)
    gives a maximum fractional distance error of ~3%.
    """

    if rmax is not None and np.abs (r) > rmax:
        raise ValueError ('sphofs radius value %f is too big for '
                          'our approximation' % r)

    lat2 = lat1 + r * np.cos (pa)
    lon2 = lon1 + r * np.sin (pa) / np.cos (lat2)

    if tol is not None:
        s = sphdist (lat1, lon1, lat2, lon2)
        if np.any (np.abs ((s - r) / s) > tol):
            raise ValueError ('sphofs approximation broke down '
                              '(%s %s %s %s %s %s %s)' % (lat1, lon1,
                                                          lat2, lon2,
                                                          r, s, pa))

    return lat2, lon2


# Spherical trig tools that are more astronomy-specific. Note that precise
# positional calculations should generally use precastro and (indirectly)
# NOVAS.

def parang (hourangle, declination, latitude):
    """Calculate the parallactic angle of a sky position.

    Inputs and outputs are all in radians. Implementation adapted
    from GBTIDL parangle.pro."""

    return -np.arctan2 (-np.sin (hourangle),
                        np.cos (declination) * np.tan (latitude)
                        - np.sin (declination) * np.cos (hourangle))


# 2D Gaussian (de)convolution

def gaussian_convolve (maj1, min1, pa1, maj2, min2, pa2):
    """Arguments:

    maj1 - major axis of input Gaussian 1
    min1 - etc
    pa1  - Gaussian 1 PA in radians.
    maj2 - major axis of input Gaussian 2
    min2 -
    pa2  - Gaussian 1 PA in radians.

    Returns: (maj3, min3, pa3).

    Axes can be in any units so long as they're consistent.

    """
    # copied from miriad/src/subs/gaupar.for:gaufac()
    c1 = np.cos (pa1)
    s1 = np.sin (pa1)
    c2 = np.cos (pa2)
    s2 = np.sin (pa2)

    a = (maj1*c1)**2 + (min1*s1)**2 + (maj2*c2)**2 + (min2*s2)**2
    b = (maj1*s1)**2 + (min1*c1)**2 + (maj2*s2)**2 + (min2*c2)**2
    g = 2 * ((min1**2 - maj1**2) * s1 * c1 + (min2**2 - maj2**2) * s2 * c2)

    s = a + b
    t = np.sqrt ((a - b)**2 + g**2)
    maj3 = np.sqrt (0.5 * (s + t))
    min3 = np.sqrt (0.5 * (s - t))

    if abs (g) + abs (a - b) == 0:
        pa3 = 0.
    else:
        pa3 = 0.5 * np.arctan2 (-g, a - b)

    # "Amplitude of the resulting Gaussian":
    # f = pi / (4 * np.log (2)) * maj1 * min1 * maj2 * min2 \
    #    / np.sqrt (a * b - 0.25 * g**2)

    return maj3, min3, pa3


def gaussian_deconvolve (smaj, smin, spa, bmaj, bmin, bpa):
    """Deconvolve a source with regard to a PSF.

    smaj - source major axis.
    smin - source minor axis.
    spa  - source PA in radians.
    bmaj - beam/PSF major axis.
    bmin - beam/PSF minor axis.
    bpa  - beam/PSF PA in radians.

    Returns: (dmaj, dmin, dpa, status). Units are consistent with the inputs.
    `status` is one of 'ok', 'pointlike', 'fail'.

    Ideally if:

      tmaj, tmin, tpa, status = gaussian_deconvolve (cmaj, cmin, cpa, bmaj, bmin, bpa)

    then:

      cmaj, cmin, cpa = gaussian_convolve (tmaj, tmin, tpa, bmaj, bmin, bpa)

    Derived from miriad gaupar.for:GauDfac()

    We currently don't do a great job of dealing with pointlike sources. I've
    added extra code ensure smaj >= bmaj, smin >= bmin, and increased
    coefficient in front of "limit" from 0.1 to 0.5. Feel a little wary about
    that first change.

    """
    from numpy import cos, sin, sqrt, min, abs, arctan2

    if smaj < bmaj:
        smaj = bmaj
    if smin < bmin:
        smin = bmin

    alpha = ((smaj * cos (spa))**2 + (smin * sin (spa))**2 -
             (bmaj * cos (bpa))**2 - (bmin * sin (bpa))**2)
    beta = ((smaj * sin (spa))**2 + (smin * cos (spa))**2 -
            (bmaj * sin (bpa))**2 - (bmin * cos (bpa))**2)
    gamma = 2 * ((smin**2 - smaj**2) * sin (spa) * cos (spa) -
                 (bmin**2 - bmaj**2) * sin (bpa) * cos (bpa))

    s = alpha + beta
    t = sqrt ((alpha - beta)**2 + gamma**2)
    limit = 0.5 * min ([smaj, smin, bmaj, bmin])**2
    #limit = 0.1 * min ([smaj, smin, bmaj, bmin])**2
    status = 'ok'

    if alpha < 0 or beta < 0 or s < t:
        dmaj = dmin = dpa = 0

        if 0.5 * (s - t) < limit and alpha > -limit and beta > -limit:
            status = 'pointlike'
        else:
            status = 'fail'
    else:
        dmaj = sqrt (0.5 * (s + t))
        dmin = sqrt (0.5 * (s - t))

        if abs (gamma) + abs (alpha - beta) == 0:
            dpa = 0
        else:
            dpa = 0.5 * arctan2 (-gamma, alpha - beta)

    return dmaj, dmin, dpa, status


# Given astrometric properties of a source, predict its position *with
# uncertainties* at a given date through Monte Carlo simulations involving
# precastro.

_vizurl = 'http://vizier.u-strasbg.fr/viz-bin/asu-tsv'

def get_2mass_epoch (tmra, tmdec, debug=False):
    """Given a 2MASS ra/dec in radians, fetch the epoch when it was observed
    as an MJD."""
    from urllib2 import urlopen
    postdata = '''-mime=csv
-source=2MASS
-out=_q,JD
-c=%.6f %.6f
-c.eq=J2000''' % (tmra * R2D, tmdec * R2D)

    jd = None

    for line in urlopen (_vizurl, postdata):
        line = line.strip ()
        if debug:
            print ('D: 2M >>', line)

        if line.startswith ('1;'):
            jd = float (line[2:])

    if jd is None:
        import sys
        print ('warning: 2MASS epoch lookup failed; astrometry could be very wrong!',
               file=sys.stderr)
        return J2000

    return jd - 2400000.5


_simbadbase = 'http://simbad.u-strasbg.fr/simbad/sim-script?script='
_simbaditems = ('COO(d;A) COO(d;D) COO(E) COO(B) PM(A) PM(D) PM(E) PLX(V) PLX(E) '
                'RV(V) RV(E)').split ()

def get_simbad_astrometry_info (ident, items=_simbaditems, debug=False):
    from urllib import quote
    from urllib2 import urlopen

    s = '\\n'.join ('%s %%%s' % (i, i) for i in items)
    s = '''output console=off script=off
format object "%s"
query id %s''' % (s, ident)
    url = _simbadbase + quote (s)
    results = {}
    errtext = None

    for line in urlopen (url):
        line = line.strip ()
        if debug:
            print ('D: SA >>', line)

        if errtext is not None:
            errtext += line
        elif line.startswith ('::error'):
            errtext = ''
        elif len (line):
            k, v = line.split (' ', 1)
            results[k] = v

    if errtext is not None:
        raise Exception ('SIMBAD query error: ' + errtext)
    return results


class AstrometryInfo (object):
    """Holds astrometric data and their uncertainties, and can compute
    predicted positions with uncertainties.

    Fields:

    ra          - J2000 right ascension of the object, in radians
    dec         - J2000 declination of the object, in radians
    pos_u_maj   - Major axis of positional error ellipse, in radians
    pos_u_min   - Minor axis of positional error ellipse, in radians
    pos_u_pa    - Position angle of positional error ellipse, East from
                  North, in radians
    pos_epoch   - Epoch of position; that is, date when the position was
                  measured, as an MJD[TT].
    promo_ra    - Proper motion in right ascension, in mas/yr.
                  XXX: terminology for cos(delta) factor
    promo_dec   - Proper motion in declination, in mas/yr.
    promo_u_maj - Major axis of proper motion error ellipse, in mas/yr
    promo_u_min - Minor axis of proper motion error ellipse, in mas/yr
    promo_u_pa  - Position angle of proper motion error ellipse, East from
                  North, in radians
    parallax    - Parallax of the target, in mas.
    u_parallax  - Uncertainty in the parallax, in mas.
    vradial     - Radial velocity of the object, in km/s.
                  XXX: totally pointless?
    u_vradial   - Uncertainty in the radial velocity, in km/s.
                  XXX: totally pointless?

    Methods:

    fill_from_simbad (ident, debug=False) - does what it says
    verify(complain=True) - make sure fields are consistent and valid
    predict(mjd, complain=True) - predict position with uncertainty at given MJD
        returns (ra, dec, maj, min, pa), all in radians
    print_prediction(ptup) - prints prediction to stdout prettily
    """

    ra = None
    dec = None
    pos_u_maj = None
    pos_u_min = None
    pos_u_pa = None
    pos_epoch = None
    promo_ra = None
    promo_dec = None
    promo_u_maj = None
    promo_u_min = None
    promo_u_pa = None
    parallax = None
    u_parallax = None
    vradial = None
    u_vradial = None


    def __init__ (self, simbadident=None, **kwargs):
        if simbadident is not None:
            self.fill_from_simbad (simbadident)

        for k, v in kwargs.iteritems ():
            setattr (self, k, v)


    def _partial_info (self, val0, *rest):
        if not len (rest):
            return False

        first = val0 is None
        for v in rest:
            if (v is None) != first:
                return True
        return False


    def verify (self, complain=True):
        import sys

        if self.ra is None:
            raise ValueError ('AstrometryInfo missing "ra"')
        if self.dec is None:
            raise ValueError ('AstrometryInfo missing "dec"')

        if self._partial_info (self.promo_ra, self.promo_dec):
            raise ValueError ('partial proper-motion info in AstrometryInfo')

        if self._partial_info (self.pos_u_maj, self.pos_u_min, self.pos_u_pa):
            raise ValueError ('partial positional uncertainty info in AstrometryInfo')

        if self._partial_info (self.promo_u_maj, self.promo_u_min, self.promo_u_pa):
            raise ValueError ('partial proper-motion uncertainty info in AstrometryInfo')

        if self.pos_u_maj is None:
            if complain:
                print ('AstrometryInfo: no positional uncertainty info', file=sys.stderr)
        elif self.pos_u_maj < self.pos_u_min:
            # Based on experience with PM, this may be possible
            if complain:
                print ('AstrometryInfo: swapped positional uncertainty '
                       'major/minor axes', file=sys.stderr)
            self.pos_u_maj, self.pos_u_min = self.pos_u_min, self.pos_u_maj
            self.pos_u_pa += 0.5 * np.pi

        if self.promo_ra is None:
            if complain:
                print ('AstrometryInfo: assuming zero proper motion', file=sys.stderr)
        elif self.promo_u_maj is None:
            if complain:
                print ('AstrometryInfo: no uncertainty on proper motion', file=sys.stderr)
        elif self.promo_u_maj < self.promo_u_min:
            # I've seen this: V* V374 Peg
            if complain:
                print ('AstrometryInfo: swapped proper motion uncertainty '
                       'major/minor axes', file=sys.stderr)
            self.promo_u_maj, self.promo_u_min = self.promo_u_min, self.promo_u_maj
            self.promo_u_pa += 0.5 * np.pi

        if self.parallax is None:
            if complain:
                print ('AstrometryInfo: assuming zero parallax', file=sys.stderr)
        else:
            if self.parallax < 0.:
                raise ValueError ('negative parallax in AstrometryInfo')
            if self.u_parallax is None:
                if complain:
                    print ('AstrometryInfo: no uncertainty on parallax', file=sys.stderr)

        if self.vradial is None:
            pass # not worth complaining
        elif self.u_vradial is None:
            if complain:
                print ('AstrometryInfo: no uncertainty on v_radial', file=sys.stderr)

        return self # chain-friendly


    def predict (self, mjd, complain=True, n=20000):
        """Returns (ra, dec, major, minor, pa), all in radians. These are the
        predicted position of the object and its uncertainty at `mjd`. If
        `complain` is True, print out warnings for incomplete information. `n`
        is the number of Monte Carlo samples to draw for computing the
        positional uncertainty.

        """
        import precastro, sys
        from . import ellipses
        o = precastro.SiderealObject ()
        self.verify (complain=complain)

        # "Best" position.

        o.ra = self.ra
        o.dec = self.dec

        if self.pos_epoch is not None:
            o.promoepoch = self.pos_epoch + 2400000.5
        else:
            if complain:
                print ('AstrometryInfo.predict(): assuming epoch of '
                       'position is J2000.0', file=sys.stderr)
            o.promoepoch = 2451545.0 # J2000.0

        if self.promo_ra is not None:
            o.promora = self.promo_ra
            o.promodec = self.promo_dec
        if self.parallax is not None:
            o.parallax = self.parallax
        if self.vradial is not None:
            o.vradial = self.vradial

        bestra, bestdec = o.astropos (mjd + 2400000.5)

        # Monte Carlo to get an uncertainty. As always, astronomy position
        # angle convention requires that we treat declination as X and RA as
        # Y. First, we check sanity and generate randomized parameters:

        if self.pos_u_maj is None and self.promo_u_maj is None and self.u_parallax is None:
            if complain:
                print ('AstrometryInfo.predict(): no uncertainties '
                       'available; cannot Monte Carlo!', file=sys.stderr)
            return (bestra, bestdec, 0., 0., 0.)

        if self.pos_u_maj is not None:
            sd, sr, cdr = ellipses.ellbiv (self.pos_u_maj, self.pos_u_min, self.pos_u_pa)
            decs, ras = ellipses.bivrandom (self.dec, self.ra, sd, sr, cdr, n).T
        else:
            ras = np.zeros (n) + self.ra
            decs = np.zeros (n) + self.dec

        if self.promo_ra is None:
            pmras = np.zeros (n)
            pmdecs = np.zeros (n)
        elif self.promo_u_maj is not None:
            sd, sr, cdr = ellipses.ellbiv (self.promo_u_maj, self.promo_u_min, self.promo_u_pa)
            pmdecs, pmras = ellipses.bivrandom (self.promo_dec, self.promo_ra, sd, sr, cdr, n).T
        else:
            pmras = np.zeros (n) + self.promo_ra
            pmdecs = np.zeros (n) + self.promo_dec

        if self.parallax is None:
            parallaxes = np.zeros (n)
        elif self.u_parallax is not None:
            parallaxes = np.random.normal (self.parallax, self.u_parallax, n)
        else:
            parallaxes = np.zeros (n) + self.parallax

        if self.vradial is not None:
            vradials = np.random.normal (self.vradial, self.u_vradial, n)
        else:
            vradials = np.zeros (n)

        # Now we compute the positions and summarize as an ellipse:

        results = np.empty ((n, 2))

        for i in xrange (n):
            o.ra = ras[i]
            o.dec = decs[i]
            o.promora = pmras[i]
            o.promodec = pmdecs[i]
            o.parallax = parallaxes[i]
            o.vradial = vradials[i]

            ara, adec = o.astropos (mjd + 2400000.5)
            results[i] = adec, ara

        maj, min, pa = ellipses.bivell (*ellipses.databiv (results))

        # All done.

        return bestra, bestdec, maj, min, pa


    def print_prediction (self, ptup, precision=2):
        """The argument is the tuple returned by predict(). Prints it to stdout."""
        from . import ellipses
        bestra, bestdec, maj, min, pa = ptup

        f = ellipses.sigmascale (1)
        maj *= R2A
        min *= R2A
        pa *= R2D

        print ('position =', fmtradec (bestra, bestdec, precision=precision))
        print ('err(1σ)  = %.*f" × %.*f" @ %.0f°' % (precision, maj * f, precision,
                                                     min * f, pa))


    def fill_from_simbad (self, ident, debug=False):
        """Fills in astrometric information based on Simbad/Sesame. Returns `self`.

        """
        import sys
        info = get_simbad_astrometry_info (ident, debug=debug)
        posref = 'unknown'

        for k, v in info.iteritems ():
            if '~' in v:
                continue # no info

            if k == 'COO(d;A)':
                self.ra = float (v) * D2R
            elif k == 'COO(d;D)':
                self.dec = float (v) * D2R
            elif k == 'COO(E)':
                a = v.split ()
                self.pos_u_maj = float (a[0]) * A2R * 1e-3 # mas -> rad
                self.pos_u_min = float (a[1]) * A2R * 1e-3
                self.pos_u_pa = float (a[2]) * D2R
            elif k == 'COO(B)':
                posref = v
            elif k == 'PM(A)':
                self.promo_ra = float (v) # mas/yr
            elif k == 'PM(D)':
                self.promo_dec = float (v) # mas/yr
            elif k == 'PM(E)':
                a = v.split ()
                self.promo_u_maj = float (a[0]) # mas/yr
                self.promo_u_min = float (a[1])
                self.promo_u_pa = float (a[2]) * D2R # rad!
            elif k == 'PLX(V)':
                self.parallax = float (v) # mas
            elif k == 'PLX(E)':
                self.u_parallax = float (v) # mas
            elif k == 'RV(V)':
                self.vradial = float (v) # km/s
            elif k == 'RV(E)':
                self.u_vradial = float (v) #km/s

        if self.ra is None:
            raise Exception ('no position returned by Simbad for "%s"' % ident)
        if self.u_parallax == 0:
            self.u_parallax = None
        if self.u_vradial == 0:
            self.u_vradial = None

        # Get the right epoch of position for 2MASS positions

        if posref == '2003yCat.2246....0C':
            self.pos_epoch = get_2mass_epoch (self.ra, self.dec, debug)

        return self # eases chaining


    def __unicode__ (self):
        self.verify (complain=False)
        a = []
        a.append (u'Position: ' + fmtradec (self.ra, self.dec))
        if self.pos_u_maj is None:
            a.append (u'No uncertainty info for position.')
        else:
            a.append (u'Pos. uncert: %.3f" × %.3f" @ %.0f°' %
                      (self.pos_u_maj * R2A, self.pos_u_min * R2A,
                       self.pos_u_pa * R2D))
        if self.pos_epoch is None:
            a.append (u'No epoch of position.')
        else:
            a.append (u'Epoch of position: MJD %.3f' % self.pos_epoch)
        if self.promo_ra is None:
            a.append (u'No proper motion.')
        else:
            a.append (u'Proper motion: %.3f, %.3f mas/yr' % (self.promo_ra, self.promo_dec))
        if self.promo_u_maj is None:
            a.append (u'No uncertainty info for proper motion.')
        else:
            a.append (u'Promo. uncert: %.1f × %.1f mas/yr @ %.0f°' %
                      (self.promo_u_maj, self.promo_u_min,
                       self.promo_u_pa * R2D))
        if self.parallax is None:
            a.append (u'No parallax information.')
        elif self.u_parallax is not None:
            a.append (u'Parallax: %.1f ± %.1f mas' % (self.parallax, self.u_parallax))
        else:
            a.append (u'Parallax: %.1f mas, unknown uncert.' % self.parallax)
        if self.vradial is None:
            a.append (u'No radial velocity information.')
        elif self.u_vradial is not None:
            a.append (u'Radial velocity: %.2f ± %.2f km/s' % (self.vradial, self.u_vradial))
        else:
            a.append (u'Radial velocity: %.1f km/s, unknown uncert.' % self.vradial)
        return u'\n'.join (a)


    __str__ = unicode_to_str
