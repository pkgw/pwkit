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
from . import *


class UsageError (PKError):
    pass


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

def cmd_abs2app (args):
    if len (args) != 2:
        raise UsageError ('abs2app expected exactly 2 arguments')

    absmag = fparse (args[0])
    dist = fparse (args[1])
    appmag = absmag + 5 * (log10 (dist) - 1)
    print (fmt (appmag))
cmd_abs2app.argspec = '<absmag [mag]> <dist [pc]>'
cmd_abs2app.summary = 'Convert absolute to apparent magnitude.'

def cmd_app2abs (args):
    if len (args) != 2:
        raise UsageError ('app2abs expected exactly 2 arguments')

    appmag = fparse (args[0])
    dist = fparse (args[1])
    absmag = appmag - 5 * (log10 (dist) - 1)
    print (fmt (absmag))
cmd_app2abs.argspec = '<appmag [mag]> <dist [pc]>'
cmd_app2abs.summary = 'Convert apparent to absolute magnitude.'


def cmd_c2m (args):
    """c2m  - calendar to MJD[TAI]"""

    if len (args) not in (3, 6):
        raise UsageError ('c2m expected exactly 3 or 6 arguments')

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
cmd_c2m.argspec = '<year> <month> <frac.day> [hour] [min] [sec]'
cmd_c2m.summary = 'Convert a calendar date to MJD[TAI].'


def cmd_calc (args):
    if not len (args):
        raise UsageError ('calc expected arguments')

    print (fmt (fparse (' '.join (args))))
cmd_calc.argspec = '{expr...}'
cmd_calc.summary = 'Evaluate and print an expression.'


def cmd_csep (args):
    if len (args) != 4:
        raise UsageError ('csep expected 4 arguments')

    try:
        lat1 = parsedeglat (args[1])
        lon1 = parsehours (args[0])
        lat2 = parsedeglat (args[3])
        lon2 = parsehours (args[2])
    except Exception as e:
        die (e)

    print ('degree:', fmt (sphdist (lat1, lon1, lat2, lon2) * R2D))
    print ('arcsec:', fmt (sphdist (lat1, lon1, lat2, lon2) * R2A))
cmd_csep.argspec = '<RA 1> <dec 1> <RA 2> <dec 2>'
cmd_csep.summary = 'Print separation between two positions; args in sexagesimal.'


def cmd_d2sh (args):
    if not len (args):
        raise UsageError ('d2sh expected arguments')

    deglon = fparse (' '.join (args))
    print (fmthours (deglon * D2R))
cmd_d2sh.argspec = '{expr}'
cmd_d2sh.summary = 'Convert decimal degrees to sexagesimal hours (RA).'


def cmd_d2sl (args):
    if not len (args):
        raise UsageError ('d2sl expected arguments')

    deglat = fparse (' '.join (args))
    print (fmtdeglat (deglat * D2R))
cmd_d2sl.argspec = '{expr}'
cmd_d2sl.summary = 'Convert decimal degrees to a sexagesimal latitude (declination).'


def cmd_ephem (args):
    # This is obviously not going to be super high accuracy.

    if len (args) != 2:
        raise UsageError ('ephem expected exactly 2 arguments')

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
cmd_ephem.argspec = '<name> <mjd>'
cmd_ephem.summary = 'Compute position of ephemeris object at a given time.'


def cmd_flux2lum (args):
    if len (args) != 2:
        raise UsageError ('flux2lum expected exactly 2 arguments')

    flux = fparse (args[0])
    dist = fparse (args[1])
    lum = flux * 4 * pi * (dist * cmperpc)**2
    print (fmt (lum))
cmd_flux2lum.argspec = '<flux [cgs]> <dist [pc]>'
cmd_flux2lum.summary = 'Compute luminosity from flux and distance.'


def cmd_lum2flux (args):
    if len (args) != 2:
        raise UsageError ('lum2flux expected exactly 2 arguments')

    lum = fparse (args[0])
    dist = fparse (args[1])
    flux = lum / (4 * pi * (dist * cmperpc)**2)
    print (fmt (flux))
cmd_lum2flux.argspec = '<lum [cgs]> <dist [pc]>'
cmd_lum2flux.summary = 'Compute flux from luminosity and distance.'


def cmd_m2c (args):
    if len (args) != 1:
        raise UsageError ('m2c expected exactly 1 argument')

    try:
        import precastro
    except ImportError:
        die ('need the "precastro" module')

    mjd = float (args[0])
    t = precastro.Time ().fromMJD (mjd, 'TAI')
    print (t.fmtcalendar ())
cmd_m2c.argspec = '<MJD>'
cmd_m2c.summary = 'Convert MJD[TAI] to a calendar date.'


def cmd_sastrom (args):
    verbose = pop_option ('v', args)

    if len (args) != 2:
        raise UsageError ('simbadastrom expected 2 arguments')

    ident = args[0]
    mjd = float (args[1])

    info = AstrometryInfo ()
    info.fill_from_simbad (ident, debug=verbose)
    p = info.predict (mjd)
    print ('%s at %.3f:' % (ident, mjd))
    print ()
    info.print_prediction (p)
cmd_sastrom.argspec = '[-v] <source name> <MJD>'
cmd_sastrom.summary = 'Compute source location using Simbad data.'


def cmd_sesame (args):
    if not len (args):
        raise UsageError ('sesame expected an argument')

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
cmd_sesame.argspec = '{source name}'
cmd_sesame.summary = 'Print source information from Sesame.'


def cmd_senscale (args):
    if len (args) != 3:
        raise UsageError ('senscale expected 3 arguments')

    s1 = fparse (args[0])
    t1 = fparse (args[1])
    t2 = fparse (args[2])
    s2 = s1 * np.sqrt (t1 / t2)
    print (fmt (s2))
cmd_senscale.argspec = '<sens 1> <time 1> <time 2>'
cmd_senscale.summary = 'Scale a sensitivity to a different integration time.'


def cmd_sh2d (args):
    if len (args) != 1:
        raise UsageError ('sh2d expected 1 argument')

    try:
        rarad = parsehours (args[0])
    except Exception as e:
        die ('couldn\'t parse "%s" as sexagesimal hours: %s (%s)',
             args[0], e, e.__class__.__name__)

    print ('%.8f' % (rarad * R2D))
cmd_sh2d.argspec = '<sexagesimal hours>'
cmd_sh2d.summary = 'Convert sexagesimal hours to decimal degrees.'
cmd_sh2d.moredocs = """The argument should look like "12:20:14.6"."""


def cmd_sl2d (args):
    if len (args) != 1:
        raise UsageError ('sl2d expected 1 argument')

    try:
        decrad = parsedeglat (args[0])
    except Exception as e:
        die ('couldn\'t parse "%s" as sexagesimal latitude in degrees: %s (%s)',
             args[0], e, e.__class__.__name__)

    print ('%.8f' % (decrad * R2D))
cmd_sl2d.argspec = '<sexagesimal latitude in degrees>'
cmd_sl2d.summary = 'Convert sexagesimal latitude (ie, declination) to decimal degrees.'
cmd_sl2d.moredocs = """The argument should look like "47:20:14.6". The leading sign is optional."""


def cmd_ssep (args):
    if len (args) != 2:
        raise UsageError ('ssep expected 2 arguments')

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
cmd_ssep.argspec = '<source name> <source name>'
cmd_ssep.summary = 'Print separation between two sources identified by name.'


# The driver.

def _fullusage ():
    usagestr = """astrotool <command> [arguments...]

This is a tool that does quick astronomical calculations that I find myself
performing frequently.

Subcommands are:

"""

    g = globals ()
    cnames = sorted (n for n in g.iterkeys () if n.startswith ('cmd_'))

    for cname in cnames:
        usagestr += '  astrotool %-8s - %s\n' % (cname[4:], g[cname].summary)

    usagestr += """
Most commands will give help if run with no arguments."""

    return usagestr

usagestr = _fullusage ()


def commandline (argv=None):
    if argv is None:
        argv = sys.argv
        propagate_sigint ()
        unicode_stdio ()

    check_usage (usagestr, argv, usageifnoargs='long')

    if len (argv) < 2:
        wrong_usage (usagestr, 'need to specify a command')

    cmdname = argv[1]
    func = globals ().get ('cmd_' + cmdname)

    if func is None:
        wrong_usage (usagestr, 'no such command "%s"', cmdname)

    args = argv[2:]
    if not len (args) and not hasattr (func, 'no_args_is_ok'):
        print ('usage: astrotool', cmdname, func.argspec)
        print ()
        print (func.summary)
        if hasattr (func, 'moredocs'):
            print ()
            print (func.moredocs)
        return

    try:
        func (args)
    except UsageError as e:
        print ('error:', e, '\n\nusage: astrotool', cmdname, func.argspec,
               file=sys.stderr)
