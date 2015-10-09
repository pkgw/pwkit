# -*- mode: python; coding: utf-8 -*-
# Copyright 2012-2014 Peter Williams <peter@newton.cx> and collaborators
# Licensed under the MIT License.

"""pwkit.data_gui_helpers - helpers for GUIs looking at data arrays

Classes:

Clipper      - Map data into [0,1]
ColorMapper  - Map data onto RGB colors using `pwkit.colormaps`
Stretcher    - Map data within [0,1] using a stretch like sqrt, etc.

Functions:

data_to_argb32       - Turn arbitrary data values into ARGB32 colors.
data_to_imagesurface - Turn arbitrary data values into a Cairo ImageSurface.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = str ('data_to_argb32 Clipper ColorMapper LazyComputer Stretcher').split ()

import numpy as np
from six.moves import range

from . import colormaps


DEFAULT_TILESIZE = 128

class LazyComputer (object):
    buffer = None
    tilesize = None
    valid = None

    def set_buffer (self, buffer):
        self.buffer = buffer
        return self


    def alloc_buffer (self, template):
        if np.ma.is_masked (template):
            self.buffer = np.ma.empty (template.shape)
            self.buffer.mask = template.mask
        else:
            self.buffer = np.empty (template.shape)
        return self


    def set_tile_size (self, tilesize=DEFAULT_TILESIZE):
        self.tilesize = tilesize
        h, w = self.buffer.shape
        nxt = (w + tilesize - 1) // tilesize
        nyt = (h + tilesize - 1) // tilesize
        self.valid = np.zeros ((nyt, nxt))
        return self


    def ensure_region_updated (self, data, xoffset, yoffset, width, height):
        ts = self.tilesize
        buf = self.buffer
        valid = self.valid
        func = self._make_func (np.ma.is_masked (data))

        tilej = xoffset // ts
        tilei = yoffset // ts
        nxt = (xoffset + width + ts - 1) // ts - tilej
        nyt = (yoffset + height + ts - 1) // ts - tilei

        tyofs = tilei
        pyofs = tilei * ts

        for i in range (nyt):
            txofs = tilej
            pxofs = tilej * ts

            for j in range (nxt):
                if not valid[tyofs,txofs]:
                    func (data[pyofs:pyofs+ts,pxofs:pxofs+ts],
                          buf[pyofs:pyofs+ts,pxofs:pxofs+ts])
                    valid[tyofs,txofs] = 1

                pxofs += ts
                txofs += 1
            pyofs += ts
            tyofs += 1

        return self


    def ensure_all_updated (self, data):
        return self.ensure_region_updated (data, 0, 0, data.shape[1], data.shape[0])


    def invalidate (self):
        self.valid.fill (0)
        return self


class Clipper (LazyComputer):
    dmin = None
    dmax = None

    def default_bounds (self, data):
        dmin, dmax = data.min (), data.max ()

        if not np.isfinite (dmin):
            dmin = data[np.ma.where (np.isfinite (data))].min ()
        if not np.isfinite (dmax):
            dmax = data[np.ma.where (np.isfinite (data))].max ()

        self.dmin = dmin
        self.dmax = dmax
        return self


    def _make_func (self, ismasked):
        dmin = self.dmin
        scale = 1. / (self.dmax - dmin)

        if ismasked:
            def func (src, dest):
                np.subtract (src, dmin, dest)
                np.multiply (dest, scale, dest)
                np.clip (dest, 0, 1, dest)
                dest.mask[:] = src.mask
        else:
            def func (src, dest):
                np.subtract (src, dmin, dest)
                np.multiply (dest, scale, dest)
                np.clip (dest, 0, 1, dest)

        return func


class Stretcher (LazyComputer):
    """Assumes that its inputs are in [0, 1]. Maps its outputs to the same
    range.

    """

    def passthrough (src, dest):
        dest[:] = src

    modes = {
        'linear': passthrough,
        'sqrt': np.sqrt,
        'square': np.square,
    }

    def __init__ (self, mode):
        if mode not in self.modes:
            raise ValueError ('unrecognized Stretcher mode %r', mode)

        self.mode = mode

    def _make_func (self, ismasked):
        return self.modes[self.mode]


class ColorMapper (LazyComputer):
    mapper = None

    def __init__ (self, mapname):
        if mapname is not None:
            self.mapper = colormaps.factory_map[mapname] ()


    def alloc_buffer (self, template):
        self.buffer = np.empty (template.shape, dtype=np.uint32)
        self.buffer.fill (0xFF000000)
        return self


    def _make_func (self, ismasked):
        mapper = self.mapper

        if not ismasked:
            def func (src, dest):
                mapped = mapper (src)
                dest.fill (0xFF000000)
                effscratch = (mapped[:,:,0] * 0xFF).astype (np.uint32)
                np.left_shift (effscratch, 16, effscratch)
                np.bitwise_or (dest, effscratch, dest)
                effscratch = (mapped[:,:,1] * 0xFF).astype (np.uint32)
                np.left_shift (effscratch, 8, effscratch)
                np.bitwise_or (dest, effscratch, dest)
                effscratch = (mapped[:,:,2] * 0xFF).astype (np.uint32)
                np.bitwise_or (dest, effscratch, dest)
        else:
            scratch2 = np.zeros ((self.tilesize, self.tilesize), dtype=np.uint32)

            def func (src, dest):
                effscratch2 = scratch2[:dest.shape[0],:dest.shape[1]]
                mapped = mapper (src)

                dest.fill (0xFF000000)
                effscratch = (mapped[:,:,0] * 0xFF).astype (np.uint32)
                np.left_shift (effscratch, 16, effscratch)
                np.bitwise_or (dest, effscratch, dest)
                effscratch = (mapped[:,:,1] * 0xFF).astype (np.uint32)
                np.left_shift (effscratch, 8, effscratch)
                np.bitwise_or (dest, effscratch, dest)
                effscratch = (mapped[:,:,2] * 0xFF).astype (np.uint32)
                np.bitwise_or (dest, effscratch, dest)

                np.invert (src.mask, effscratch2)
                np.multiply (dest, effscratch2, dest)

        return func


def data_to_argb32 (data, cmin=None, cmax=None, stretch='linear', cmap='black_to_blue'):
    """Turn arbitrary data values into ARGB32 colors.

    There are three steps to this process: clipping the data values to a
    maximum and minimum; stretching the spacing between those values; and
    converting their amplitudes into colors with some kind of color map.

    `data`    - Input data; can (and should) be a MaskedArray if some values are
                invalid.
    `cmin`    - The data clip minimum; all values <= cmin are treated
                identically. If None (the default), `data.min ()` is used.
    `cmax`    - The data clip maximum; all values >= cmax are treated
                identically. If None (the default), `data.max ()` is used.
    `stretch` - The stretch function name; 'linear', 'sqrt', or 'square'; see
                the Stretcher class.
    `cmap`    - The color map name; defaults to 'black_to_blue'. See the
                `pwkit.colormaps` module for more choices.

    Returns a Numpy array of the same shape as `data` with dtype `np.uint32`,
    which represents the ARGB32 colorized version of the data. If your
    colormap is restricted to a single R or G or B channel, you can make color
    images by bitwise-or'ing together different such arrays.

    """
    # This could be more efficient, but whatever. This lets us share code with
    # the ndshow module.

    clipper = Clipper ()
    clipper.alloc_buffer (data)
    clipper.set_tile_size ()
    clipper.dmin = cmin if cmin is not None else data.min ()
    clipper.dmax = cmax if cmax is not None else data.max ()
    clipper.ensure_all_updated (data)

    stretcher = Stretcher (stretch)
    stretcher.alloc_buffer (clipper.buffer)
    stretcher.set_tile_size ()
    stretcher.ensure_all_updated (clipper.buffer)

    mapper = ColorMapper (cmap)
    mapper.alloc_buffer (stretcher.buffer)
    mapper.set_tile_size ()
    mapper.ensure_all_updated (stretcher.buffer)

    return mapper.buffer


def data_to_imagesurface (data, **kwargs):
    """Turn arbitrary data values into a Cairo ImageSurface.

    The method and arguments are the same as data_to_argb32, except that the
    data array will be treated as 2D, and higher dimensionalities are not
    allowed. The return value is a Cairo ImageSurface object.

    Combined with the write_to_png() method on ImageSurfaces, this is an easy
    way to quickly visualize 2D data.

    """
    import cairo

    data = np.atleast_2d (data)
    if data.ndim != 2:
        raise ValueError ('input array may not have more than 2 dimensions')

    argb32 = data_to_argb32 (data, **kwargs)

    format = cairo.FORMAT_ARGB32
    height, width = argb32.shape
    stride = cairo.ImageSurface.format_stride_for_width (format, width)

    if argb32.strides[0] != stride:
        raise ValueError ('stride of data array not compatible with ARGB32')

    return cairo.ImageSurface.create_for_data (argb32, format,
                                               width, height, stride)
