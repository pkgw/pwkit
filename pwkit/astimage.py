# -*- mode: python; coding: utf-8 -*-
# Copyright 2012-2014 Peter Williams <peter@newton.cx> and collaborators.
# Licensed under the MIT License.

"""pwkit.astimage -- generic loading of (radio) astronomical images

Use `open (path, mode)` to open an astronomical image, regardless of its file
format.

The emphasis of this module is on getting 90%-good-enough semantics and a
really, genuinely, uniform interface. This can be tough to achieve.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

# Developer notes:
"""Note that CASA allegedly supports HDF5, FITS, and MIRIAD format images too.
Frankly, I don't trust it, I don't like its API, and I don't want to rely on
some variant of casacore being installed.

TODO: axis types (ugh standardizing these would be a bear)
      Some kind of way to get generic formatting of RA/Dec, glat/glon,
      etc would be nice.

TODO: image units (ie, "set units to Jy/px"; standardization also a pain)

"""
__all__ = str ('''UnsupportedError AstroImage MIRIADImage PyrapImage FITSImage SimpleImage
                  open''').split ()

import six
from six.moves import range
import numpy as np
from numpy import pi

from . import PKError
from .astutil import D2R, R2D


class UnsupportedError (PKError):
    pass


def _load_fits_module ():
    try:
        from astropy.io import fits
        return fits
    except ImportError:
        pass

    try:
        import pyfits
        import warnings

        # YARRRRGGHH. Some versionfs of Pyfits override functions in the
        # warnings module for no good reason. (The motivation seems to be
        # compat with Python <2.6.) Part of what it does is to make warnings
        # be printed to stdout, rather than stderr, which breaks things if
        # you're printing specially-formatted output to stdout and a random
        # warning comes up. Hypothetically. At least the original routines are
        # backed up.

        if hasattr (pyfits.core, '_formatwarning'):
            warnings.formatwarning = pyfits.core._formatwarning

        if hasattr (pyfits.core, '_showwarning'):
            warnings.showwarning = pyfits.core._showwarning

        return pyfits
    except ImportError:
        pass

    raise UnsupportedError ('cannot open FITS images without either the '
                            '"astropy.io.fits" or "pyfits" modules')


def _load_wcs_module ():
    try:
        from astropy import wcs
        return wcs
    except ImportError:
        pass

    try:
        import pywcs
        return pywcs
    except ImportError:
        pass

    raise UnsupportedError ('cannot open this image without either the '
                            '"astropy.wcs" or "pywcs" modules')


def _create_wcs (fitsheader):
    """For compatibility between astropy and pywcs."""
    wcsmodule = _load_wcs_module ()
    is_pywcs = hasattr (wcsmodule, 'UnitConverter')

    wcs = wcsmodule.WCS (fitsheader)
    wcs.wcs.set ()
    wcs.wcs.fix () # I'm interested in MJD computation via datfix()

    if hasattr (wcs, 'wcs_pix2sky'):
        wcs.wcs_pix2world = wcs.wcs_pix2sky
        wcs.wcs_world2pix = wcs.wcs_sky2pix

    return wcs


class AstroImage (object):
    """An astronomical image.

    path
      The filesystem path of the image.
    mode
      Its access mode: 'r' for read, 'rw' for read/write.
    shape
      The data shape, like numpy.ndarray.shape.
    bmaj
      If not None, the restoring beam FWHM major axis in radians.
    bmin
      If not None, the restoring beam FWHM minor axis in radians.
    bpa
      If not None, the restoring beam position angle (east from celestial
      north) in radians.
    units
      Lower-case string describing image units (e.g., jy/beam, jy/pixel).
      Not standardized between formats.
    pclat
      Latitude (usually dec) of the pointing center in radians.
    pclon
      Longitude (usually RA) of the pointing center in radians.
    charfreq
      Characteristic observing frequency of the image in GHz.
    mjd
      Mean MJD of the observations.
    axdescs
      If not None, list of strings describing the axis types.
      Not standardized.
    size
      The number of pixels in the image (=shape.prod ()).

    Methods:

    close
      Close the image.
    read
      Read all of the data.
    write
      Rewrite all of the data.
    toworld
      Convert pixel coordinates to world coordinates.
    topixel
      Convert world coordinates to pixel coordinates.
    simple
      Convert to a 2D lat/lon image.
    subimage
      Extract a sub-cube of the image.
    save_copy
      Save a copy of the image.
    save_as_fits
      Save a copy of the image in FITS format.
    delete
      Delete the on-disk image.

    """
    path = None
    mode = None
    _handle = None

    _latax = None # index of the spatial latitude axis
    _lonax = None # ditto for longitude
    _specax = None # ditto for spectral axis (may be freq or velocity!)

    shape = None
    "An integer ndarray of the image shape"

    bmaj = None
    "If not None, the restoring beam FWHM major axis in radians"

    bmin = None
    "If not None, the restoring beam FWHM minor axis in radians"

    bpa = None
    """If not None, the restoring beam position angle (east
    from celestial north) in radians"""

    units = None
    "Lower-case string describing image units (e.g., jy/beam, jy/pixel)"

    pclat = None
    "Latitude of the pointing center in radians"

    pclon = None
    "Longitude of the pointing center in radians"

    charfreq = None
    "Characteristic observing frequency of the image in GHz"
    # NOTE: we get this from evaluating the spectral axis in its middle
    # pixel, not the reference value.

    mjd = None
    "Mean MJD of the observations"

    axdescs = None
    """If not None, list of strings describing the axis types;
    no standard format."""

    def __init__ (self, path, mode):
        self.path = path
        self.mode = mode


    def __del__ (self):
        self.close ()


    def close (self):
        if self._handle is not None:
            self._close_impl ()
            self._handle = None


    def __enter__ (self):
        return self


    def __exit__ (self, etype, evalue, traceback):
        self.close ()
        return False # raise any exception that may have happened


    def _checkOpen (self):
        if self._handle is None:
            raise UnsupportedError ('this operation cannot be performed on the '
                                    'closed image at "%s"', self.path)


    def _checkWriteable (self):
        if self.mode == 'r':
            raise UnsupportedError ('this operation cannot be performed on the '
                                    'read-only image at "%s"', self.path)


    @property
    def size (self):
        return np.prod (self.shape)


    def read (self, squeeze=False, flip=False):
        raise NotImplementedError ()


    def write (self, data):
        raise NotImplementedError ()


    def toworld (self, pixel):
        raise NotImplementedError ()


    def topixel (self, world):
        raise NotImplementedError ()


    def simple (self):
        if self._latax == 0 and self._lonax == 1 and self.shape.size == 2:
            return self # noop
        return SimpleImage (self)


    def subimage (self, pixofs, shape):
        """Extract a sub-cube of this image.

        Both `pixofs` and `shape` should be integer arrays with as many
        elements as this image has axes. Thinking of this operation as taking
        a Python slice of an N-dimensional cube, the i'th axis of the
        sub-image is slices from `pixofs[i]` to `pixofs[i] + shape[i]`.

        """
        return NaiveSubImage (self, pixofs, shape)


    def save_copy (self, path, overwrite=False, openmode=None):
        raise NotImplementedError ()


    def save_as_fits (self, path, overwrite=False, openmode=None):
        raise NotImplementedError ()


    def delete (self):
        raise NotImplementedError ()


def maybescale (x, a):
    if x is None:
        return None
    return a * x


def maybelower (x):
    if x is None:
        return None
    return x.lower ()


# We use astropy.wcs/WCSLIB/pywcs for coordinates for both FITS and MIRIAD
# images. It does two things that we don't like. First of all, it stores axes
# in Fortran style, with the first axis being the most rapidly varying.
# Secondly, it does all of its angular work in degrees, not radians (why??).
# We fix these up as best we can.

def _get_wcs_scale (wcs, naxis):
    wcsmodule = _load_wcs_module ()
    wcscale = np.ones (naxis)

    is_pywcs = hasattr (wcsmodule, 'UnitConverter')
    if not is_pywcs:
        from astropy.units import UnitsError, rad

    for i in range (naxis):
        u = wcs.wcs.cunit[wcscale.size - 1 - i]

        if is_pywcs:
            try:
                uc = wcsmodule.UnitConverter (u.strip (), 'rad')
                wcscale[i] = uc.scale
            except SyntaxError: # !! pywcs 1.10
                pass # not an angle unit; don't futz.
            except ValueError: # pywcs 1.11
                pass
        else:
            try:
                wcscale[i] = u.to (rad)
            except UnitsError:
                pass # not an angle unit

    return wcscale


def _wcs_toworld (wcs, pixel, wcscale, naxis):
    # TODO: we don't allow the usage of "SIP" or "Paper IV"
    # transformations, let alone a concatenation of these, because
    # they're not invertible.

    pixel = np.asarray (pixel)
    if pixel.shape != (naxis, ):
        raise ValueError ('pixel coordinate must be a %d-element vector', naxis)

    pixel = pixel.reshape ((1, naxis))[:,::-1]
    world = wcs.wcs_pix2world (pixel, 0)
    return world[0,::-1] * wcscale


def _wcs_topixel (wcs, world, wcscale, naxis):
    world = np.asarray (world)
    if world.shape != (naxis, ):
        raise ValueError ('world coordinate must be a %d-element vector', naxis)

    world = (world / wcscale)[::-1].reshape ((1, naxis))
    pixel = wcs.wcs_world2pix (world, 0)
    return pixel[0,::-1]


def _wcs_axes (wcs, naxis):
    lat = lon = spec = None

    if wcs.wcs.lat >= 0:
        lat = naxis - 1 - wcs.wcs.lat
    if wcs.wcs.lng >= 0:
        lon = naxis - 1 - wcs.wcs.lng
    if wcs.wcs.spec >= 0:
        spec = naxis - 1 - wcs.wcs.spec

    return lat, lon, spec


def _wcs_get_freq (wcs, specval):
    from mirtask._miriad_c import mirwcs_compute_freq
    assert wcs.wcs.spec >= 0
    spectype = wcs.wcs.ctype[wcs.wcs.spec][:4]
    return mirwcs_compute_freq (spectype, specval, wcs.wcs.restfrq) * 1e-9


class MIRIADImage (AstroImage):
    """A MIRIAD format image. Requires the `mirtask` module from miriad-python."""

    _modemap = {'r': 'rw', # no true read-only option
                'rw': 'rw'
                }

    def __init__ (self, path, mode):
        try:
            from mirtask import XYDataSet
        except ImportError:
            raise UnsupportedError ('cannot open MIRIAD images without the '
                                    'Python module "mirtask"')

        super (MIRIADImage, self).__init__ (path, mode)

        self._handle = h = XYDataSet (path, self._modemap[mode])
        self._wcs, warnings = h.wcs ()

        if hasattr (self._wcs, 'wcs_pix2sky'):
            self._wcs.wcs_pix2world = self._wcs.wcs_pix2sky
            self._wcs.wcs_world2pix = self._wcs.wcs_sky2pix

        for w in warnings:
            # Whatever.
            import sys
            print ('irregularity in coordinates of "%s": %s' % (self.path, w),
                   file=sys.stderr)

        naxis = h.getScalarItem ('naxis', 0)
        self.shape = np.empty (naxis, dtype=np.int)
        self.axdescs = []

        for i in range (naxis):
            q = naxis - i
            self.shape[i] = h.getScalarItem ('naxis%d' % q, 1)
            self.axdescs.append (h.getScalarItem ('ctype%d' % q, '???'))

        self.units = maybelower (h.getScalarItem ('bunit'))

        self.bmaj = h.getScalarItem ('bmaj')
        if self.bmaj is not None:
            self.bmin = h.getScalarItem ('bmin', self.bmaj)
            self.bpa = h.getScalarItem ('bpa', 0) * D2R

        self.pclat = h.getScalarItem ('obsdec')
        if self.pclat is not None:
            self.pclon = h.getScalarItem ('obsra')
        else:
            try:
                import mirtask.mostable
            except ImportError:
                pass
            else:
                mt = mirtask.mostable.readDataSet (h)[0] # ignore WCS warnings here
                if mt.radec.shape[0] == 1:
                    self.pclat = mt.radec[0,1]
                    self.pclon = mt.radec[0,0]

        self._wcscale = _get_wcs_scale (self._wcs, self.shape.size)
        self._latax, self._lonax, self._specax = _wcs_axes (self._wcs, self.shape.size)

        if self._specax is not None:
            try:
                from mirtask._miriad_c import mirwcs_compute_freq
            except ImportError:
                pass
            else:
                specval = self.toworld (0.5 * (self.shape - 1))[self._specax]
                self.charfreq = _wcs_get_freq (self._wcs, specval)

        jd = h.getScalarItem ('obstime')
        if jd is not None:
            self.mjd = jd - 2400000.5


    def _close_impl (self):
        self._handle.close ()


    def read (self, squeeze=False, flip=False):
        self._checkOpen ()
        nonplane = self.shape[:-2]

        if nonplane.size == 0:
            data = self._handle.readPlane ([], topIsZero=flip)
        else:
            data = np.ma.empty (self.shape, dtype=np.float32)
            data.mask = np.zeros (self.shape, dtype=np.bool)
            n = np.prod (nonplane)
            fdata = data.reshape ((n, self.shape[-2], self.shape[-1]))

            for i in range (n):
                # Must convert from C to Fortran indexing convention
                axes = np.unravel_index (i, nonplane)[::-1]
                self._handle.readPlane (axes, fdata[i], topIsZero=flip)

        if squeeze:
            data = data.squeeze ()

        return data


    def write (self, data):
        data = np.ma.asarray (data)

        if data.shape != tuple (self.shape):
            raise ValueError ('"data" is wrong shape: got %s, want %s' \
                                  % (data.shape, tuple (self.shape)))

        self._checkOpen ()
        self._checkWriteable ()
        nonplane = self.shape[:-2]

        if nonplane.size == 0:
            self._handle.writePlane (data, [])
        else:
            n = np.prod (nonplane)
            fdata = data.reshape ((n, self.shape[-2], self.shape[-1]))

            for i in range (n):
                axes = np.unravel_index (i, nonplane)
                self._handle.writePlane (fdata[i], axes)

        return self


    def toworld (self, pixel):
        # self._wcs is still valid if we've been closed, so no need
        # to _checkOpen().

        if self._wcs is None:
            raise UnsupportedError ('world coordinate information is required '
                                    'but not present in "%s"', self.path)

        return _wcs_toworld (self._wcs, pixel, self._wcscale, self.shape.size)


    def topixel (self, world):
        if self._wcs is None:
            raise UnsupportedError ('world coordinate information is required '
                                    'but not present in "%s"', self.path)

        return _wcs_topixel (self._wcs, world, self._wcscale, self.shape.size)


    def save_copy (self, path, overwrite=False, openmode=None):
        import shutil, os.path

        # FIXME: race conditions and such in overwrite checks.
        # Too lazy to do a better job.

        if os.path.exists (path):
            if overwrite:
                if os.path.isdir (path):
                    shutil.rmtree (path)
                else:
                    os.unlink (path)
            else:
                raise UnsupportedError ('refusing to copy "%s" to "%s": '
                                        'destination already exists' % (self.path, path))

        shutil.copytree (self.path, path, symlinks=False)

        if openmode is None:
            return None
        return open (path, openmode)


    def save_as_fits (self, path, overwrite=False, openmode=None):
        from mirexec import TaskFits
        import os.path

        if os.path.exists (path):
            if overwrite:
                os.unlink (path)
            else:
                raise UnsupportedError ('refusing to export "%s" to "%s": '
                                        'destination already exists' % (self.path, path))

        TaskFits (op='xyout', in_=self.path, out=path).runsilent ()

        if openmode is None:
            return None
        return FITSImage (path, openmode)


    def delete (self):
        if self._handle is not None:
            raise UnsupportedError ('cannot delete the image at "%s" without '
                                    'first closing it', self.path)
        self._checkWriteable ()

        import shutil, os.path

        if os.path.isdir (self.path):
            shutil.rmtree (self.path)
        else:
            os.unlink (self.path) # may be a symlink; rmtree rejects this


# CASA images. We need either casac or pyrap.

class _CasaUnsupportedImage (AstroImage):
    def __init__ (self, path, mode):
        raise UnsupportedError ('no modules are available for reading CASA images')


def _pyrap_convert (d, unitstr):
    from pyrap.quanta import quantity
    return quantity (d['value'], d['unit']).get_value (unitstr)


class PyrapImage (AstroImage):
    """A CASA-format image loaded with the 'pyrap' Python module."""

    def __init__ (self, path, mode):
        try:
            from pyrap.images import image
        except ImportError:
            raise UnsupportedError ('cannot open CASAcore images in Pyrap mode without '
                                    'the Python module "pyrap.images"')

        super (PyrapImage, self).__init__ (path, mode)

        # no mode specifiable
        self._handle = image (path)

        allinfo = self._handle.info ()
        self.units = maybelower (allinfo.get ('unit'))
        self.shape = np.asarray (self._handle.shape (), dtype=np.int)
        self.axdescs = []

        if 'coordinates' in allinfo:
            pc = allinfo['coordinates'].get ('pointingcenter')
            # initial=True signifies that the pointing center information
            # hasn't actually been initialized.
            if pc is not None and not pc['initial']:
                # This bit of info doesn't have any metadata about units or
                # whatever; appears to be fixed as RA/Dec in radians.
                self.pclat = pc['value'][1]
                self.pclon = pc['value'][0]

        ii = self._handle.imageinfo ()

        if 'restoringbeam' in ii:
            self.bmaj = _pyrap_convert (ii['restoringbeam']['major'], 'rad')
            self.bmin = _pyrap_convert (ii['restoringbeam']['minor'], 'rad')
            self.bpa = _pyrap_convert (ii['restoringbeam']['positionangle'], 'rad')

        # Make sure that angular units are always measured in radians,
        # because anything else is ridiculous.

        from pyrap.quanta import quantity
        self._wcscale = wcscale = np.ones (self.shape.size)
        c = self._handle.coordinates ()
        radian = quantity (1., 'rad')

        for item in c.get_axes ():
            if isinstance (item, six.string_types):
                self.axdescs.append (item.replace (' ', '_'))
            else:
                for subitem in item:
                    self.axdescs.append (subitem.replace (' ', '_'))

        def getconversion (text):
            q = quantity (1., text)
            if q.conforms (radian):
                return q.get_value ('rad')
            return 1

        i = 0

        for item in c.get_unit ():
            if isinstance (item, six.string_types):
                wcscale[i] = getconversion (item)
                i += 1
            elif len (item) == 0:
                wcscale[i] = 1 # null unit
                i += 1
            else:
                for subitem in item:
                    wcscale[i] = getconversion (subitem)
                    i += 1

        # Figure out which axes are lat/long/spec. We have some
        # paranoia code the give up in case there are multiple axes
        # that appear to be of the same type. This stuff could
        # be cleaned up.

        lat = lon = spec = -1

        try:
            logspecidx = c.get_names ().index ('spectral')
        except ValueError:
            specaxname = None
        else:
            specaxname = c.get_axes ()[logspecidx]

        for i, name in enumerate (self.axdescs):
            # These symbolic direction names obtained from
            # casacore/coordinates/Coordinates/DirectionCoordinate.cc
            # Would be nice to have a better system for determining
            # this a la what wcslib provides.
            if name == specaxname:
                spec = i
            elif name in ('Right_Ascension', 'Hour_Angle', 'Longitude'):
                if lon == -1:
                    lon = i
                else:
                    lon = -2
            elif name in ('Declination', 'Latitude'):
                if lat == -1:
                    lat = i
                else:
                    lat = -2

        if lat >= 0:
            self._latax = lat
        if lon >= 0:
            self._lonax = lon
        if spec >= 0:
            self._specax = spec

        # Phew, that was gross.

        if self._specax is not None:
            sd = c.get_coordinate ('spectral').dict ()
            wi = sd.get ('wcs')
            if wi is not None:
                try:
                    from mirtask._miriad_c import mirwcs_compute_freq
                except ImportError:
                    pass
                else:
                    spectype = wi['ctype'].replace ('\x00', '')[:4]
                    restfreq = sd.get ('restfreq', 0.)
                    specval = self.toworld (0.5 * (self.shape - 1))[self._specax]
                    self.charfreq = mirwcs_compute_freq (spectype, specval, restfreq) * 1e-9

        # TODO: any unit weirdness or whatever here?
        self.mjd = c.get_obsdate ()['m0']['value']


    def _close_impl (self):
        # No explicit close method provided here. Annoying.
        del self._handle


    def read (self, squeeze=False, flip=False):
        self._checkOpen ()
        data = self._handle.get ()

        if flip:
            data = data[...,::-1,:]
        if squeeze:
            data = data.squeeze ()
        return data


    def write (self, data):
        data = np.ma.asarray (data)

        if data.shape != tuple (self.shape):
            raise ValueError ('data is wrong shape: got %s, want %s' \
                                  % (data.shape, tuple (self.shape)))

        self._checkOpen ()
        self._checkWriteable ()
        self._handle.put (data)
        return self


    def toworld (self, pixel):
        self._checkOpen ()
        pixel = np.asarray (pixel)
        return self._wcscale * np.asarray (self._handle.toworld (pixel))


    def topixel (self, world):
        self._checkOpen ()
        world = np.asarray (world)
        return np.asarray (self._handle.topixel (world / self._wcscale))


    def save_copy (self, path, overwrite=False, openmode=None):
        self._checkOpen ()
        self._handle.saveas (path, overwrite=overwrite)

        if openmode is None:
            return None
        return open (path, openmode)


    def save_as_fits (self, path, overwrite=False, openmode=None):
        self._checkOpen ()
        self._handle.tofits (path, overwrite=overwrite)

        if openmode is None:
            return None
        return FITSImage (path, openmode)


    def delete (self):
        if self._handle is not None:
            raise UnsupportedError ('cannot delete the image at "%s" without '
                                    'first closing it', self.path)
        self._checkWriteable ()

        import shutil, os.path

        if os.path.isdir (self.path):
            shutil.rmtree (self.path)
        else:
            os.unlink (self.path) # may be a symlink; rmtree rejects this


# 'casac' images.

def _casac_convert (d, unitstr):
    from .environments.casa.util import tools
    qa = tools.quanta ()
    x = qa.quantity (d['value'], d['unit'])
    return qa.convert (x, unitstr)['value']


def _casac_findwcoord (cs, kind):
    v = cs.findcoordinate (kind)
    if 'world' in v:
        return np.atleast_1d (v['world'])
    return v[2]

class CasaCImage (AstroImage):
    """A CASA-format image loaded with the 'casac' Python module."""

    # casac uses Fortran-style axis ordering, with innermost first. casac
    # coordinate systems have two axis orderings, "world" and "pixel", and
    # generally default to world. We want to keep things in pixel terms so we
    # have to undo the mapping.

    def __init__ (self, path, mode):
        super (CasaCImage, self).__init__ (path, mode)
        from .environments.casa.util import tools
        self._handle = tools.image ()
        self._handle.open (path) # no mode specifiable.
        self.shape = np.asarray (self._handle.shape ())[::-1].copy ()

        cs = self._handle.coordsys ()
        naxis = self.shape.size
        # for world-to-pixel, we reverse the values in the array:
        w2p = self._wax2pax = naxis - 1 - np.asarray (cs.axesmap (toworld=False))
        # for pixel-to-world, we reverse the array ordering:
        p2w = self._pax2wax = np.asarray (cs.axesmap (toworld=True))[::-1].copy ()
        assert p2w.size == self.shape.size

        self._specax = w2p[_casac_findwcoord (cs, b'spectral')[0]]
        #self._polax = w2p[_casac_findwcoord (cs, b'stokes')[0]]
        self._lonax = w2p[_casac_findwcoord (cs, b'direction')[0]]
        self._latax = w2p[_casac_findwcoord (cs, b'direction')[1]]

        self.axdescs = [None] * naxis
        names = cs.names ()
        for i in range (naxis):
            self.axdescs[i] = names[p2w[i]]

        self.charfreq = _casac_convert (cs.referencevalue (format=b'm', type=b'spectral')
                                        ['measure']['spectral']['frequency']['m0'], 'GHz')
        self.units = maybelower (self._handle.brightnessunit ())

        rb = self._handle.restoringbeam ()
        if 'major' in rb:
            self.bmaj = _casac_convert (rb['major'], 'rad')
            self.bmin = _casac_convert (rb['minor'], 'rad')
            self.bpa = _casac_convert (rb['positionangle'], 'rad')

        # pclat, pclon?

        # this is the *start* of the observation, annoyingly. The timescale is
        # available but we ignore it. Generally is in UTC :-(
        self.mjd = cs.epoch ()['m0']['value']


    def _close_impl (self):
        self._handle.close ()


    def read (self, squeeze=False, flip=False):
        self._checkOpen ()

        # Older casac doesn't accept ndarrays, and we need to be careful to
        # change from Numpy types to Python ints.
        blc = [0] * self.shape.size
        trc = [int (x) for x in (self.shape[::-1] - 1)]

        data = self._handle.getchunk (blc, trc, dropdeg=squeeze, getmask=False)
        data = data.T # does the right thing and gives us C-contiguous data
        mask = self._handle.getchunk (blc, trc, dropdeg=squeeze, getmask=True)
        np.logical_not (mask, mask) # CASA image masking is opposite of CASA UV flagging: T -> OK
        mask = mask.T

        data = np.ma.MaskedArray (data, mask=mask)

        if flip:
            data = data[...,::-1,:]

        return data


    def write (self, data):
        self._checkOpen ()
        self._checkWriteable ()

        data = np.ma.asarray (data)
        if data.shape != tuple (self.shape):
            raise ValueError ('data is wrong shape: got %s, want %s' \
                                  % (data.shape, tuple (self.shape)))

        data = data.T # back to CASA convention
        # putchunk can't do the mask:
        self._handle.putregion (pixels=data.data, pixelmask=data.mask,
                                usemask=True)
        return self


    def toworld (self, pixel):
        # TODO: CASA quantities seem to be spat out in radians and Hz,
        # which work well enough for us. But this might not be
        # reliable. And perhaps we'll want to enforce units for
        # frequency and/or velocity axes.

        self._checkOpen ()
        pixel = np.asarray (pixel)[::-1] # reverse to CASA's ordering
        qtys = self._handle.toworld (pixel, format=b'q')['quantity']

        # Our "world" coordinates are still in what CASA would call
        # its "pixel" ordering. This will probably all go down in
        # flames if anyone ever reorders or removes axes.

        naxis = self.shape.size
        world = np.empty (naxis)
        for i in range (naxis):
            s = '*%d' % (self._pax2wax[i] + 1)
            world[i] = qtys[s]['value']

        return world


    def topixel (self, world):
        self._checkOpen ()
        myworld = np.asarray (world)
        ncwa = self._wax2pax.size # num of CASA world axes
        casaworld = np.zeros (ncwa)

        for i in range (ncwa):
            casaworld[self._pax2wax[i]] = myworld[i]

        casapixel = self._handle.topixel (casaworld)['numeric']
        return casapixel[::-1].copy ()


    def save_copy (self, path, overwrite=False, openmode=None):
        self._checkOpen ()

        # In theory we could be more efficient and reuse this tool if openmode
        # is not None. In practice, I'd have to mess with __init__() and who
        # cares?

        from .environments.casa.util import tools
        ia = tools.image ()
        ia.newimagefromimage (self.path, path, overwrite=overwrite)
        ia.close ()

        if openmode is None:
            return None
        return open (path, openmode)


    def save_as_fits (self, path, overwrite=False, openmode=None):
        self._checkOpen ()

        self._handle.tofits (path, overwrite=overwrite)

        if openmode is None:
            return None
        return FITSImage (path, openmode)


    def delete (self):
        if self._handle is not None:
            raise UnsupportedError ('cannot delete the image at "%s" without '
                                    'first closing it', self.path)
        self._checkWriteable ()

        import shutil, os.path

        if os.path.isdir (self.path):
            shutil.rmtree (self.path)
        else:
            os.unlink (self.path) # may be a symlink; rmtree rejects this


try:
    import casac
    CASAImage = CasaCImage
except ImportError:
    try:
        from pyrap.images import image
        CASAImage = PyrapImage
    except ImportError:
        CASAImage = _CasaUnsupportedImage


class FITSImage (AstroImage):
    """A FITS format image."""

    _modemap = {'r': 'readonly',
                'rw': 'update' # ???
                }

    def __init__ (self, path, mode):
        fitsmodule = _load_fits_module ()
        wcsmodule = _load_wcs_module ()

        super (FITSImage, self).__init__ (path, mode)

        self._handle = fitsmodule.open (path, self._modemap[mode])
        header = self._handle[0].header
        self._wcs = _create_wcs (header)

        self.units = maybelower (header.get ('bunit'))

        naxis = header.get ('naxis', 0)
        self.shape = np.empty (naxis, dtype=np.int)
        self.axdescs = []

        for i in range (naxis):
            q = naxis - i
            self.shape[i] = header.get ('naxis%d' % q, 1)
            self.axdescs.append (header.get ('ctype%d' % q, '???'))

        self.bmaj = maybescale (header.get ('bmaj'), D2R)
        if self.bmaj is None:
            bmindefault = None
        else:
            bmindefault = self.bmaj * R2D
        self.bmin = maybescale (header.get ('bmin', bmindefault), D2R)
        self.bpa = maybescale (header.get ('bpa', 0), D2R)

        self.pclat = maybescale (header.get ('obsdec'), D2R)
        self.pclon = maybescale (header.get ('obsra'), D2R)

        self._wcscale = _get_wcs_scale (self._wcs, self.shape.size)
        self._latax, self._lonax, self._specax = _wcs_axes (self._wcs, self.shape.size)

        if self._specax is not None:
            try:
                from mirtask._miriad_c import mirwcs_compute_freq
            except ImportError:
                pass
            else:
                specval = self.toworld (0.5 * (self.shape - 1))[self._specax]
                self.charfreq = _wcs_get_freq (self._wcs, specval)

        if np.isfinite (self._wcs.wcs.mjdavg):
            self.mjd = self._wcs.wcs.mjdavg
        elif np.isfinite (self._wcs.wcs.mjdobs):
            # close enough
            self.mjd = self._wcs.wcs.mjdobs


    def _close_impl (self):
        self._handle.close ()


    def read (self, squeeze=False, flip=False):
        self._checkOpen ()
        data = np.ma.asarray (self._handle[0].data)
        # Are there other standards for expressing masking in FITS?
        data.mask = -np.isfinite (data.data)

        if flip:
            data = data[...,::-1,:]
        if squeeze:
            data = data.squeeze ()
        return data


    def write (self, data):
        data = np.ma.asarray (data)

        if data.shape != tuple (self.shape):
            raise ValueError ('data is wrong shape: got %s, want %s' \
                                  % (data.shape, tuple (self.shape)))

        self._checkOpen ()
        self._checkWriteable ()
        self._handle[0].data[:] = data
        self._handle.flush ()
        return self


    def toworld (self, pixel):
        if self._wcs is None:
            raise UnsupportedError ('world coordinate information is required '
                                    'but not present in "%s"', self.path)
        return _wcs_toworld (self._wcs, pixel, self._wcscale, self.shape.size)


    def topixel (self, world):
        if self._wcs is None:
            raise UnsupportedError ('world coordinate information is required '
                                    'but not present in "%s"', self.path)
        return _wcs_topixel (self._wcs, world, self._wcscale, self.shape.size)


    def save_copy (self, path, overwrite=False, openmode=None):
        self._checkOpen ()
        self._handle.writeto (path, output_verify='fix', clobber=overwrite)

        if openmode is None:
            return None
        return open (path, openmode)


    def save_as_fits (self, path, overwrite=False, openmode=None):
        return self.save_copy (path, overwrite=overwrite, openmode=openmode)


    def delete (self):
        if self._handle is not None:
            raise UnsupportedError ('cannot delete the image at "%s" without '
                                    'first closing it', self.path)
        self._checkWriteable ()

        os.unlink (self.path)


class SimpleImage (AstroImage):
    """A 2D, latitude/longitude image, referenced to a parent image."""

    def __init__ (self, parent):
        platax, plonax = parent._latax, parent._lonax

        if platax is None or plonax is None or platax == plonax:
            raise UnsupportedError ('the image "%s" does not have both latitude '
                                    'and longitude axes', parent.path)

        self._handle = parent
        self._platax = platax
        self._plonax = plonax

        self._latax = 0
        self._lonax = 1

        checkworld1 = parent.toworld (parent.shape * 0.) # need float!
        checkworld2 = parent.toworld (parent.shape - 1.) # (for pyrap)
        self._topixelok = True

        for i in range (parent.shape.size):
            # Two things to check. Firstly, that all non-lat/lon axes have
            # only one pixel; this limitation can be relaxed if we add a
            # mechanism for choosing which non-spatial pixels to work with.
            #
            # Secondly, check that non-lat/lon world coordinates don't vary
            # over the image; otherwise topixel() will be broken.
            if i in (platax, plonax):
                continue
            if parent.shape[i] != 1:
                raise UnsupportedError ('cannot simplify an image with '
                                        'nondegenerate nonspatial axes')
            if checkworld2[i] == 0:
                if checkworld1[i] == 0:
                    pass
                elif np.abs (checkworld1[i]) > 1e-6:
                    self._topixelok = False
            elif np.abs (1 - checkworld1[i] / checkworld2[i]) > 1e-6:
                self._topixelok = False

        self.path = '<subimage of %s>' % parent.path
        self.shape = np.asarray ([parent.shape[platax], parent.shape[plonax]])
        self.axdescs = [parent.axdescs[platax], parent.axdescs[plonax]]
        self.bmaj = parent.bmaj
        self.bmin = parent.bmin
        self.bpa = parent.bpa
        self.units = parent.units
        self.pclat = parent.pclat
        self.pclon = parent.pclon
        self.charfreq = parent.charfreq
        self.mjd = parent.mjd

        self._pctmpl = np.zeros (parent.shape.size)
        self._wctmpl = parent.toworld (self._pctmpl)


    def _close_impl (self):
        pass


    def read (self, squeeze=False, flip=False):
        self._checkOpen ()
        data = self._handle.read (flip=flip)
        idx = [0] * self._pctmpl.size
        idx[self._platax] = slice (None)
        idx[self._plonax] = slice (None)
        data = data[tuple (idx)]

        if self._platax > self._plonax:
            # Ensure that order is (lat, lon). Note that unlike the
            # above operations, this forces a copy of data.
            data = data.T

        if squeeze:
            data = data.squeeze () # could be 1-by-N ...

        return data


    def write (self, data):
        data = np.ma.asarray (data)

        if data.shape != tuple (self.shape):
            raise ValueError ('data is wrong shape: got %s, want %s' \
                                  % (data.shape, tuple (self.shape)))

        self._checkOpen ()
        self._checkWriteable ()

        fulldata = np.ma.empty (self._handle.shape, dtype=data.dtype)
        idx = [0] * self._pctmpl.size
        idx[self._platax] = slice (None)
        idx[self._plonax] = slice (None)

        if self._platax > self._plonax:
            fulldata[tuple (idx)] = data.T
        else:
            fulldata[tuple (idx)] = data

        self._handle.write (fulldata)
        return self


    def toworld (self, pixel):
        pixel = np.asarray (pixel)
        if pixel.shape != (2,):
            raise ValueError ('"pixel" must be a single pair of numbers')

        self._checkOpen ()
        p = self._pctmpl.copy ()
        p[self._platax] = pixel[0]
        p[self._plonax] = pixel[1]
        w = self._handle.toworld (p)
        world = np.empty (2)
        world[0] = w[self._platax]
        world[1] = w[self._plonax]
        return world


    def topixel (self, world):
        world = np.asarray (world)
        if world.shape != (2,):
            raise ValueError ('"world" must be a single pair of numbers')

        self._checkOpen ()
        if not self._topixelok:
            raise UnsupportedError ('mixing in the coordinate system of '
                                    'this subimage prevents mapping from '
                                    'world to pixel coordinates')

        w = self._wctmpl.copy ()
        w[self._platax] = world[0]
        w[self._plonax] = world[1]
        p = self._handle.topixel (w)
        pixel = np.empty (2)
        pixel[0] = p[self._platax]
        pixel[1] = p[self._plonax]
        return pixel


    def simple (self):
        return self


    def save_copy (self, path, overwrite=False, openmode=None):
        raise UnsupportedError ('cannot save a copy of a subimage')


    def save_as_fits (self, path, overwrite=False, openmode=None):
        raise UnsupportedError ('cannot save subimage as FITS')


class NaiveSubImage (AstroImage):
    """A subset of a parent image, implemented naively."""

    def __init__ (self, parent, pixofs, shape):
        pixofs = np.asarray (pixofs)
        shape = np.asarray (shape)

        if pixofs.shape != parent.shape.shape:
            raise UnsupportedError ('sub-image pixofs must match parent shape')
        if shape.shape != parent.shape.shape:
            raise UnsupportedError ('sub-image shape must match parent shape')
        if np.any (pixofs < 0):
            raise UnsupportedError ('sub-image pixofs must be nonnegative; '
                                    'got %r' % pixofs)
        if np.any (pixofs + shape > parent.shape):
            raise UnsupportedError ('sub-image may not extend past parent; '
                                    'got pixofs=%r subshape=%r vs shape=%r'
                                    % (pixofs, shape, parent.shape))

        self._handle = parent
        self._pixofs = pixofs
        self.shape = shape

        self.path = '<subimage of %s>' % parent.path
        self.axdescs = parent.axdescs
        self.bmaj = parent.bmaj
        self.bmin = parent.bmin
        self.bpa = parent.bpa
        self.units = parent.units
        self.pclat = parent.pclat
        self.pclon = parent.pclon
        self.charfreq = parent.charfreq
        self.mjd = parent.mjd


    def _close_impl (self):
        pass


    def read (self, squeeze=False, flip=False):
        self._checkOpen ()
        data = self._handle.read (flip=flip)
        slices = tuple (slice (self._pixofs[i], self._pixofs[i] + self.shape[i])
                        for i in range (self.shape.size))
        data = data[slices]

        if squeeze:
            data = data.squeeze ()

        return data


    def write (self, data):
        raise UnsupportedError ('writing a subimage is not supported')


    def toworld (self, pixel):
        self._checkOpen ()
        return self._handle.toworld (pixel + self._pixofs)


    def topixel (self, world):
        self._checkOpen ()
        return self._handle.topixel (world) - self._pixofs


    def save_copy (self, path, overwrite=False, openmode=None):
        raise UnsupportedError ('cannot save a copy of a subimage')


    def save_as_fits (self, path, overwrite=False, openmode=None):
        raise UnsupportedError ('cannot save subimage as FITS')


def open (path, mode, eat_warnings=False):
    import io
    from os.path import exists, join, isdir

    if eat_warnings:
        import warnings
        with warnings.catch_warnings ():
            warnings.filterwarnings ('ignore', module='astropy.*')
            return open (path, mode, eat_warnings=False)

    if mode not in ('r', 'rw'):
        raise ValueError ('mode must be "r" or "rw"; got "%s"' % mode)

    if exists (join (path, 'image')):
        return MIRIADImage (path, mode)

    if exists (join (path, 'table.dat')):
        return CASAImage (path, mode)

    if isdir (path):
        raise UnsupportedError ('cannot infer format of image "%s"' % path)

    with io.open (path, 'rb') as f:
        sniff = f.read (9)

    if sniff.startswith ('SIMPLE  ='):
        return FITSImage (path, mode)

    raise UnsupportedError ('cannot infer format of image "%s"' % path)
