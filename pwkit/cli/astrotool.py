# -*- mode: python; coding: utf-8 -*-
# Copyright 2014 Peter Williams <peter@newton.cx> and collaborators.
# Licensed under the MIT License.

"""pwkit.cli.astrotool - the 'astrotool' program."""

from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = [b'commandline']

import math, numpy as np, os, sys

from .. import PKError
from ..astutil import *
from ..cgs import *
from . import multitool
from . import *


def fparse (text):
    try:
        # note: totally reckless about globals/locals passed, etc.
        # nice to have * from astutil and cgs available.
        v = eval (text)
    except Exception as e:
        die ('cannot evaluate "%s": %s (%s)', text, str (e), e.__class__.__name__)

    try:
        f = float (v)
    except Exception as e:
        die ('evaluted "%s", but could not floatify result %r: %s (%s)', text, v,
             str (e), e.__class__.__name__)

    return f


def fmt (v):
    f1 = '%.4g' % v
    f2 = '%.3e' % v

    if f1 == f2:
        reprs = [f1]
    else:
        reprs = [f1, f2]

    if v > 0:
        reprs.append ('10**(%.2f)' % log10 (v))

    return ' = '.join (reprs)


# The commands.

class Abs2app (multitool.Command):
    name = 'abs2app'
    argspec = '<absmag [mag]> <dist [pc]>'
    summary = 'Convert absolute to apparent magnitude.'

    def invoke (self, app, args):
        if len (args) != 2:
            raise multitool.UsageError ('abs2app expected exactly 2 arguments')

        absmag = fparse (args[0])
        dist = fparse (args[1])
        appmag = absmag + 5 * (log10 (dist) - 1)
        print (fmt (appmag))


class App2abs (multitool.Command):
    name = 'app2abs'
    argspec = '<appmag [mag]> <dist [pc]>'
    summary = 'Convert apparent to absolute magnitude.'

    def invoke (self, app, args):
        if len (args) != 2:
            raise multitool.UsageError ('app2abs expected exactly 2 arguments')

        appmag = fparse (args[0])
        dist = fparse (args[1])
        absmag = appmag - 5 * (log10 (dist) - 1)
        print (fmt (absmag))


class C2m (multitool.Command):
    name = 'c2m'
    argspec = '<year> <month> <frac.day> [hour] [min] [sec]'
    summary = 'Convert a calendar date to MJD[TAI].'

    def invoke (self, app, args):
        """c2m  - calendar to MJD[TAI]"""

        if len (args) not in (3, 6):
            raise multitool.UsageError ('c2m expected exactly 3 or 6 arguments')

        try:
            import precastro
        except ImportError:
            die ('need the "precastro" module')

        year = int (args[0])
        month = int (args[1])

        if len (args) == 3:
            day = float (args[2])
            iday = int (math.floor (day))
            fday = day - iday
        else:
            iday = int (args[2])
            hour = int (args[3])
            minute = int (args[4])
            second = float (args[5])
            fday = (hour + minute / 60. + second / 3600.) / 24

        t = precastro.Time ().fromfdcal (year, month, iday, fday, 'TAI')
        print ('%.4f' % t.asMJD ())


class Calc (multitool.Command):
    name = 'calc'
    argspec = '{expr...}'
    summary = 'Evaluate and print an expression.'

    def invoke (self, app, args):
        if not len (args):
            raise multitool.UsageError ('calc expected arguments')

        print (fmt (fparse (' '.join (args))))


class Csep (multitool.Command):
    name = 'csep'
    argspec = '<RA 1> <dec 1> <RA 2> <dec 2>'
    summary = 'Print separation between two positions; args in sexagesimal.'

    def invoke (self, app, args):
        if len (args) != 4:
            raise multitool.UsageError ('csep expected 4 arguments')

        try:
            lat1 = parsedeglat (args[1])
            lon1 = parsehours (args[0])
            lat2 = parsedeglat (args[3])
            lon2 = parsehours (args[2])
        except Exception as e:
            die (e)

        print ('degree:', fmt (sphdist (lat1, lon1, lat2, lon2) * R2D))
        print ('arcsec:', fmt (sphdist (lat1, lon1, lat2, lon2) * R2A))


class D2sh (multitool.Command):
    name = 'd2sh'
    argspec = '{expr}'
    summary = 'Convert decimal degrees to sexagesimal hours (RA).'

    def invoke (self, app, args):
        if not len (args):
            raise multitool.UsageError ('d2sh expected arguments')

        deglon = fparse (' '.join (args))
        print (fmthours (deglon * D2R))


class D2sl (multitool.Command):
    name = 'd2sl'
    argspec = '{expr}'
    summary = 'Convert decimal degrees to a sexagesimal latitude (declination).'

    def invoke (self, app, args):
        if not len (args):
            raise multitool.UsageError ('d2sl expected arguments')

        deglat = fparse (' '.join (args))
        print (fmtdeglat (deglat * D2R))


class Ephem (multitool.Command):
    name = 'ephem'
    argspec = '<name> <mjd>'
    summary = 'Compute position of ephemeris object at a given time.'

    def invoke (self, app, args):
        # This is obviously not going to be super high accuracy.

        if len (args) != 2:
            raise multitool.UsageError ('ephem expected exactly 2 arguments')

        try:
            import precastro
        except ImportError:
            die ('need the "precastro" module')

        try:
            obj = precastro.EphemObject (args[0])
        except Exception as e:
            die (e)

        mjd = fparse (args[1])
        print (fmtradec (*obj.astropos (mjd + 2400000.5)))


class Flux2lum (multitool.Command):
    name = 'flux2lum'
    argspec = '<flux [cgs]> <dist [pc]>'
    summary = 'Compute luminosity from flux and distance.'

    def invoke (self, app, args):
        if len (args) != 2:
            raise multitool.UsageError ('flux2lum expected exactly 2 arguments')

        flux = fparse (args[0])
        dist = fparse (args[1])
        lum = flux * 4 * pi * (dist * cmperpc)**2
        print (fmt (lum))


class Lum2flux (multitool.Command):
    name = 'lum2flux'
    argspec = '<lum [cgs]> <dist [pc]>'
    summary = 'Compute flux from luminosity and distance.'

    def invoke (self, app, args):
        if len (args) != 2:
            raise multitool.UsageError ('lum2flux expected exactly 2 arguments')

        lum = fparse (args[0])
        dist = fparse (args[1])
        flux = lum / (4 * pi * (dist * cmperpc)**2)
        print (fmt (flux))


class M2c (multitool.Command):
    name = 'm2c'
    argspec = '<MJD>'
    summary = 'Convert MJD[TAI] to a calendar date.'

    def invoke (self, app, args):
        if len (args) != 1:
            raise multitool.UsageError ('m2c expected exactly 1 argument')

        try:
            import precastro
        except ImportError:
            die ('need the "precastro" module')

        mjd = float (args[0])
        t = precastro.Time ().fromMJD (mjd, 'TAI')
        print (t.fmtcalendar ())


class Sastrom (multitool.Command):
    name = 'sastrom'
    argspec = '[-v] <source name> <MJD>'
    summary = 'Compute source location using Simbad data.'

    def invoke (self, app, args):
        verbose = pop_option ('v', args)

        if len (args) != 2:
            raise multitool.UsageError ('simbadastrom expected 2 arguments')

        ident = args[0]
        mjd = float (args[1])

        info = AstrometryInfo ()
        info.fill_from_simbad (ident, debug=verbose)
        p = info.predict (mjd)
        print ('%s at %.3f:' % (ident, mjd))
        print ()
        info.print_prediction (p)


class Sesame (multitool.Command):
    name = 'sesame'
    argspec = '{source name}'
    summary = 'Print source information from Sesame.'

    def invoke (self, app, args):
        if not len (args):
            raise multitool.UsageError ('sesame expected an argument')

        try:
            import precastro
        except ImportError:
            die ('need the "precastro" module')

        src = ' '.join (args)
        obj = precastro.SiderealObject ()

        try:
            obj.fromsesame (src)
        except Exception as e:
            die ('couldn\'t look up "%s": %s (%s)', src, e, e.__class__.__name__)

        print (obj.describe ())


class Senscale (multitool.Command):
    name = 'senscale'
    argspec = '<sens 1> <time 1> <time 2>'
    summary = 'Scale a sensitivity to a different integration time.'

    def invoke (self, app, args):
        if len (args) != 3:
            raise multitool.UsageError ('senscale expected 3 arguments')

        s1 = fparse (args[0])
        t1 = fparse (args[1])
        t2 = fparse (args[2])
        s2 = s1 * np.sqrt (t1 / t2)
        print (fmt (s2))


class Sh2d (multitool.Command):
    name = 'sh2d'
    argspec = '<sexagesimal hours>'
    summary = 'Convert sexagesimal hours to decimal degrees.'
    more_help = """The argument should look like "12:20:14.6"."""

    def invoke (self, app, args):
        if len (args) != 1:
            raise multitool.UsageError ('sh2d expected 1 argument')

        try:
            rarad = parsehours (args[0])
        except Exception as e:
            die ('couldn\'t parse "%s" as sexagesimal hours: %s (%s)',
                 args[0], e, e.__class__.__name__)

        print ('%.8f' % (rarad * R2D))


class Sl2d (multitool.Command):
    name = 'sl2d'
    argspec = '<sexagesimal latitude in degrees>'
    summary = 'Convert sexagesimal latitude (ie, declination) to decimal degrees.'
    more_help = """The argument should look like "47:20:14.6". The leading sign is optional."""

    def invoke (self, app, args):
        if len (args) != 1:
            raise multitool.UsageError ('sl2d expected 1 argument')

        try:
            decrad = parsedeglat (args[0])
        except Exception as e:
            die ('couldn\'t parse "%s" as sexagesimal latitude in degrees: %s (%s)',
                 args[0], e, e.__class__.__name__)

        print ('%.8f' % (decrad * R2D))


class Ssep (multitool.Command):
    name = 'ssep'
    argspec = '<source name> <source name>'
    summary = 'Print separation between two sources identified by name.'

    def invoke (self, app, args):
        if len (args) != 2:
            raise multitool.UsageError ('ssep expected 2 arguments')

        try:
            import precastro
        except ImportError:
            die ('need the "precastro" module')

        try:
            obj1 = precastro.SiderealObject ().fromsesame (args[0])
        except Exception as e:
            die ('couldn\'t look up "%s": %s (%s)', args[0], e, e.__class__.__name__)

        try:
            obj2 = precastro.SiderealObject ().fromsesame (args[1])
        except Exception as e:
            die ('couldn\'t look up "%s": %s (%s)', args[1], e, e.__class__.__name__)

        print ('degree:', fmt (sphdist (obj1.dec, obj1.ra, obj2.dec, obj2.ra) * R2D))
        print ('arcsec:', fmt (sphdist (obj1.dec, obj1.ra, obj2.dec, obj2.ra) * R2A))


# The driver.

class Astrotool (multitool.Multitool):
    cli_name = 'astrotool'
    help_summary = 'Perform miscellaneous astronomical calculations.'

def commandline ():
    multitool.invoke_tool (globals ())
