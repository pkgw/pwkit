# -*- mode: python; coding: utf-8 -*-
# Copyright 2012-2014 Peter Williams <peter@newton.cx> and collaborators
# Licensed under the MIT License.

"""pwkit.data_gui_helpers - helpers for GUIs looking at data arrays

Classes:

Clipper      - Map data into [0,1]
ColorMapper  - Map data onto RGB colorrs using `pwkit.colormaps`

"""
from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = ('Clipper ColorMapper LazyComputer').split ()

import numpy as np

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

        for i in xrange (nyt):
            txofs = tilej
            pxofs = tilej * ts

            for j in xrange (nxt):
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
