# -*- mode: python; coding: utf-8 -*-
# Copyright 2014 Peter Williams <peter@newton.cx> and collaborators.
# Licensed under the MIT License.

"""pwkit.cli.astrotool - the 'astrotool' program."""

from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = str ('commandline').split ()

import math, numpy as np, os, sys
from six.moves import range

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

    def invoke (self, args, **kwargs):
        if len (args) != 2:
            raise multitool.UsageError ('abs2app expected exactly 2 arguments')

        absmag = fparse (args[0])
        dist = fparse (args[1])
        print (fmt (abs2app (absmag, dist)))


class App2abs (multitool.Command):
    name = 'app2abs'
    argspec = '<appmag [mag]> <dist [pc]>'
    summary = 'Convert apparent to absolute magnitude.'

    def invoke (self, args, **kwargs):
        if len (args) != 2:
            raise multitool.UsageError ('app2abs expected exactly 2 arguments')

        appmag = fparse (args[0])
        dist = fparse (args[1])
        print (fmt (app2abs (appmag, dist)))


class C2m (multitool.Command):
    name = 'c2m'
    argspec = '<year> <month> <frac.day> [hour] [min] [sec]'
    summary = 'Convert a UTC calendar date to MJD[TAI].'
    more_help = '''Note that fractional UTC days are not well-specified because of
the possibility of leap seconds.'''

    def invoke (self, args, **kwargs):
        """c2m  - UTC calendar to MJD[TAI]"""

        if len (args) not in (3, 6):
            raise multitool.UsageError ('c2m expected exactly 3 or 6 arguments')

        year = int (args[0])
        month = int (args[1])

        import astropy.time

        if len (args) == 3:
            day = float (args[2])
            iday = int (math.floor (day))
            r = 24 * (day - iday)
            hour = int (np.floor (r))
            r = 60 * (r - hour)
            minute = int (np.floor (r))
            second = 60 * (r - minute)
        else:
            iday = int (args[2])
            hour = int (args[3])
            minute = int (args[4])
            second = float (args[5])

        s = '%d-%02d-%02d %02d:%02d:%02.8f' % (year, month, iday,
                                               hour, minute, second)
        t = astropy.time.Time (s, format='iso', scale='utc')
        print ('%.4f' % t.tai.mjd)


class Calc (multitool.Command):
    name = 'calc'
    argspec = '{expr...}'
    summary = 'Evaluate and print an expression.'

    def invoke (self, args, **kwargs):
        if not len (args):
            raise multitool.UsageError ('calc expected arguments')

        print (fmt (fparse (' '.join (args))))


class Csep (multitool.Command):
    name = 'csep'
    argspec = '<RA 1> <dec 1> <RA 2> <dec 2>'
    summary = 'Print separation between two positions; args in sexagesimal.'

    def invoke (self, args, **kwargs):
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

    def invoke (self, args, **kwargs):
        if not len (args):
            raise multitool.UsageError ('d2sh expected arguments')

        deglon = fparse (' '.join (args))
        print (fmthours (deglon * D2R))


class D2sl (multitool.Command):
    name = 'd2sl'
    argspec = '{expr}'
    summary = 'Convert decimal degrees to a sexagesimal latitude (declination).'

    def invoke (self, args, **kwargs):
        if not len (args):
            raise multitool.UsageError ('d2sl expected arguments')

        deglat = fparse (' '.join (args))
        print (fmtdeglat (deglat * D2R))


class Ephem (multitool.Command):
    name = 'ephem'
    argspec = '<name> <mjd>'
    summary = 'Compute position of ephemeris object at a given time.'

    def invoke (self, args, **kwargs):
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

    def invoke (self, args, **kwargs):
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

    def invoke (self, args, **kwargs):
        if len (args) != 2:
            raise multitool.UsageError ('lum2flux expected exactly 2 arguments')

        lum = fparse (args[0])
        dist = fparse (args[1])
        flux = lum / (4 * pi * (dist * cmperpc)**2)
        print (fmt (flux))


class M2c (multitool.Command):
    name = 'm2c'
    argspec = '<MJD>'
    summary = 'Convert MJD[TAI] to a UTC calendar date.'

    def invoke (self, args, **kwargs):
        if len (args) != 1:
            raise multitool.UsageError ('m2c expected exactly 1 argument')

        mjd = float (args[0])

        import astropy.time
        t = astropy.time.Time (mjd, format='mjd', scale='tai')
        print (t.utc.iso, 'UTC')


class Sastrom (multitool.Command):
    name = 'sastrom'
    argspec = '[-v] <source name> <MJD>'
    summary = 'Compute source location using Simbad data.'

    def invoke (self, args, **kwargs):
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

    def invoke (self, args, **kwargs):
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

    def invoke (self, args, **kwargs):
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

    def invoke (self, args, **kwargs):
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

    def invoke (self, args, **kwargs):
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

    def invoke (self, args, **kwargs):
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


class Summfits (multitool.Command):
    name = 'summfits'
    argspec = '<path>'
    summary = 'Summarize contents of a FITS file.'

    _commentary_card_names = frozenset (('HISTORY', 'COMMENT'))
    _skip_headers = frozenset (('XTENSION', 'BITPIX', 'SIMPLE', 'EXTEND',
                                'EXTNAME', 'PCOUNT', 'GCOUNT', 'TFIELDS'))
    _skip_prefixes = frozenset (('NAXIS', 'TTYPE', 'TFORM', 'TDIM', 'TUNIT'))

    def invoke (self, args, **kwargs):
        if len (args) != 1:
            raise multitool.UsageError ('summfits expected 1 argument')

        try:
            from astropy.io import fits
        except ImportError:
            try:
                import pyfits as fits
                warn ('falling back to untested "pyfits" backend')
            except ImportError:
                die ('need either the "astropy.io.fits" or the "pyfits" modules')

        fitspath = args[0]

        try:
            hdulist = fits.open (fitspath)
        except Exception as e:
            die ('couldn\'t open "%s" as a FITS file: %s', fitspath, e)

        def output (depth, fmt, *args):
            print ('  ' * depth, fmt % args, sep='')

        output (0, '%s: %d HDUs', fitspath, len (hdulist))
        hduidxwidth = len (str (len (hdulist)))
        hdunamewidth = 0

        for hdu in hdulist:
            hdunamewidth = max (hdunamewidth, len (hdu.name))

        for hduidx, hdu in enumerate (hdulist):
            # TODO: handle more HDU kinds. See
            # astropy/io/fits/hdu/__init__.py. Looks like possibilities are
            # Primary, Image, Table, BinTable, Groups, CompImage,
            # nonstandard/Fits, and Streaming.

            if hdu.data is None:
                kind = 'empty'
            elif hasattr (hdu, 'columns'):
                kind = 'table'
            elif hdu.is_image:
                kind = 'image'
            else:
                kind = '???'

            output (1, 'HDU %*d = %*s: kind=%s size=%d ver=%s level=%s',
                    hduidxwidth, hduidx, hdunamewidth, hdu.name, kind, hdu.size,
                    hdu.ver, hdu.level)

            output (2, '%d headers (some omitted)', len (hdu.header))

            for k in hdu.header.keys ():
                if k in self._commentary_card_names or k in self._skip_headers:
                    continue
                for pfx in self._skip_prefixes:
                    if k.startswith (pfx):
                        break
                else:
                    # We did not break out of the loop -> shouldn't be skipped.
                    output (3, '%-8s = %r # %s', k, hdu.header[k], hdu.header.comments[k])

            for ck in self._commentary_card_names:
                # hacky linewrapping logic here
                if ck not in hdu.header:
                    continue

                buf = ''
                h = hdu.header[ck]
                output (2, '%s commentary', ck)

                for ccidx in range (len (h)):
                    s = h[ccidx]
                    buf += s

                    if len (s) in (71, 72):
                        continue # assume hard linewrapping

                    output (3, '%s', buf)
                    buf = ''

                if len (buf):
                    output (3, '%s', buf)

            if kind == 'table':
                colidxwidth = len (str (len (hdu.columns)))
                colnamewidth = 0

                output (2, '%d rows, %d columns', hdu.data.size, len (hdu.columns))

                for col in hdu.columns:
                    colnamewidth = max (colnamewidth, len (col.name))

                for colidx, col in enumerate (hdu.columns):
                    output (3, 'col %*d = %*s: format=%s unit=%s', colidxwidth,
                            colidx, colnamewidth, col.name, col.format, col.unit)
            elif kind == 'image':
                output (2, 'data shape=%r dtype=%s', hdu.data.shape, hdu.data.dtype)

# The driver.

from .multitool import HelpCommand

class Astrotool (multitool.Multitool):
    cli_name = 'astrotool'
    summary = 'Perform miscellaneous astronomical calculations.'

def commandline ():
    multitool.invoke_tool (globals ())
