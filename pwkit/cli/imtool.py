# -*- mode: python; coding: utf-8 -*-
# Copyright 2014 Peter Williams <peter@newton.cx> and collaborators.
# Licensed under the MIT License.

"""pwkit.cli.imtool - the 'imtool' program.

FIXME: the structure of this is all very redundant with astrotool.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = [b'commandline']

import numpy as np, sys

from .. import PKError
from . import *
from .. import astimage


class UsageError (PKError):
    pass


def load_ndshow ():
    # one day: flexible backend

    try:
        from .. import ndshow_gtk2 as ndshow
    except ImportError as e:
        die ('cannot load graphics backend for viewing images: %s', e)

    return ndshow


# The commands.

def _blink_load (path, fft, maxnorm):
    try:
        img = astimage.open (path, 'r')
    except Exception as e:
        die ('can\'t open path “%s”: %s', path, e)

    try:
        img = img.simple ()
    except Exception as e:
        print ('blink: can\'t convert “%s” to simple 2D sky image; taking '
               'first plane' % path, file=sys.stderr)
        data = img.read (flip=True)[tuple (np.zeros (img.shape.size - 2))]
        toworld = None
    else:
        data = img.read (flip=True)
        toworld = img.toworld

    if fft:
        from numpy.fft import ifftshift, fft2, fftshift
        data = np.abs (ifftshift (fft2 (fftshift (data.filled (0)))))
        data = np.ma.MaskedArray (data)
        toworld = None

    if maxnorm:
        data /= np.ma.max (data)

    return data, toworld


def cmd_blink (args):
    fft = pop_option ('f', args)
    maxnorm = pop_option ('m', args)
    ndshow = load_ndshow ()

    images = []
    toworlds = []

    for path in args:
        image, toworld = _blink_load (path, fft, maxnorm)
        images.append (image)
        toworlds.append (toworld)

    if not len (images):
        return

    shape = images[0].shape
    for i, im in enumerate (images[1:]):
        if im.shape != shape:
            die ('shape of “%s” (%s) does not agree with that '
                 'of “%s” (%s)',
                 paths[i+1], '×'.join (map (str, im.shape)),
                 paths[0], '×'.join (map (str, shape)))

    # Merge masks. This is more complicated than you might think since you
    # can't "or" nomask with itself.

    jointmask = np.ma.nomask

    for i in xrange (len (images)):
        if jointmask is np.ma.nomask:
            if images[i].mask is not np.ma.nomask:
                jointmask = images[i].mask
        else:
            np.logical_or (jointmask, images[i].mask, jointmask)

    for im in images:
        im.mask = jointmask

    ndshow.cycle (images, args, toworlds=toworlds, yflip=True)
cmd_blink.argspec = '<images...>'
cmd_blink.summary = 'Blink zero or more images.'


def cmd_fitsrc (args):
    from ..immodel import fit_one_source
    forcepoint = pop_option ('p', args)

    if len (args) != 3:
        raise UsageError ('expect exactly three arguments')

    im = astimage.open (args[0], 'r').simple ()
    x = int (args[1])
    y = int (args[2])

    fit_one_source (im, x, y, forcepoint=forcepoint)
cmd_fitsrc.argspec = '[-p] <image> <x(pixels)> <y(pixels)>'
cmd_fitsrc.summary = 'Fit a compact-source model to a location in an image.'
cmd_fitsrc.moredocs = """-p  - Force use of a point-source model."""


def cmd_hackdata (args):
    if len (args) != 2:
        raise UsageError ('expect exactly two arguments')

    inpath, outpath = args

    try:
        with astimage.open (inpath, 'r') as imin:
            indata = imin.read ()
    except Exception as e:
        die ('cannot open input "%s": %s', inpath, e)

    try:
        with astimage.open (outpath, 'rw') as imout:
            if imout.size != indata.size:
                die ('cannot import data: input has %d pixels; output has %d',
                     indata.size, imout.size)

            imout.write (indata)
    except Exception as e:
        die ('cannot write to output "%s": %s', outpath, e)
cmd_hackdata.argspec = '<inpath> <outpath>'
cmd_hackdata.summary = 'Blindly copy pixel data from one image to another.'


def _info_print (path):
    from ..astutil import fmtradec, R2A, R2D

    try:
        im = astimage.open (path, 'r')
    except Exception as e:
        die ('can\'t open "%s": %s', path, e)

    print ('kind     =', im.__class__.__name__)

    latcell = loncell = None

    if im.toworld is not None:
        latax, lonax = im._latax, im._lonax
        delta = 1e-6
        p = 0.5 * (np.asfarray (im.shape) - 1)
        w1 = im.toworld (p)
        p[latax] += delta
        w2 = im.toworld (p)
        latcell = (w2[latax] - w1[latax]) / delta
        p[latax] -= delta
        p[lonax] += delta
        w2 = im.toworld (p)
        loncell = (w2[lonax] - w1[lonax]) / delta * np.cos (w2[latax])

    if im.pclat is not None:
        print ('center   =', fmtradec (im.pclon, im.pclat), '# pointing')
    elif im.toworld is not None:
        w = im.toworld (0.5 * (np.asfarray (im.shape) - 1))
        print ('center   =', fmtradec (w[lonax], w[latax]), '# lattice')

    if im.shape is not None:
        print ('shape    =', ' '.join (str (x) for x in im.shape))
        npix = 1
        for x in im.shape:
            npix *= x
        print ('npix     =', npix)

    if im.axdescs is not None:
        print ('axdescs  =', ' '.join (x for x in im.axdescs))

    if im.charfreq is not None:
        print ('charfreq = %f GHz' % im.charfreq)

    if im.mjd is not None:
        from time import gmtime, strftime
        posix = 86400. * (im.mjd - 40587.)
        ts = strftime ('%Y-%m-%dT%H-%M-%SZ', gmtime (posix))
        print ('mjd      = %f # %s' % (im.mjd, ts))

    if latcell is not None:
        print ('ctrcell  = %fʺ × %fʺ # lat, lon' % (latcell * R2A,
                                                    loncell * R2A))

    if im.bmaj is not None:
        print ('beam     = %fʺ × %fʺ @ %f°' % (im.bmaj * R2A,
                                               im.bmin * R2A,
                                               im.bpa * R2D))

        if latcell is not None:
            bmrad2 = 2 * np.pi * im.bmaj * im.bmin / (8 * np.log (2))
            cellrad2 = latcell * loncell
            print ('ctrbmvol = %f px' % np.abs (bmrad2 / cellrad2))

    if im.units is not None:
        print ('units    =', im.units)


def cmd_info (args):
    if len (args) == 1:
        _info_print (args[0])
    else:
        for i, path in enumerate (args):
            if i > 0:
                print ()
            print ('path     =', path)
            _info_print (path)
cmd_info.argspec = '<images...>'
cmd_info.summary = 'Print properties of the image.'


def cmd_setrect (args):
    if len (args) != 5:
        raise UsageError ('expected exactly 5 arguments')

    path = args[0]

    try:
        x = int (args[1])
        y = int (args[2])
        halfwidth = int (args[3])
        value = float (args[4])
    except ValueError:
        raise UsageError ('could not parse one of the numeric arguments')

    try:
        img = astimage.open (path, 'rw')
    except Exception as e:
        die ('can\'t open path “%s”: %s', path, e)

    data = img.read ()
    data[...,y-halfwidth:y+halfwidth,x-halfwidth:x+halfwidth] = value
    img.write (data)
cmd_setrect.argspec = '<image> <x> <y> <halfwidth> <value>'
cmd_setrect.summary = 'Set a rectangle in an image to a constant.'


def cmd_show (args):
    anyfailures = False
    ndshow = load_ndshow ()

    for path in args:
        try:
            img = astimage.open (path, 'r')
        except Exception as e:
            print ('pwshow: can\'t open path “%s”: %s' % (path, e), file=sys.stderr)
            anyfailures = True
            continue

        try:
            img = img.simple ()
        except Exception as e:
            print ('pwshow: can\'t convert “%s” to simple 2D sky image; taking '
                   ' first plane' % path, file=sys.stderr)
            data = img.read (flip=True)[tuple (np.zeros (img.shape.size - 2))]
            toworld = None
        else:
            data = img.read (flip=True)
            toworld = img.toworld

        ndshow.view (data, title=path + ' — Array Viewer',
                     toworld=toworld, yflip=True)

    sys.exit (int (anyfailures))
cmd_show.argspec = '<image> [images...]'
cmd_show.summary = 'Show images interactively.'


def _stats_print (path):
    try:
        img = astimage.open (path, 'r')
    except Exception as e:
        die ('error: can\'t open "%s": %s', path, e)

    try:
        img = img.simple ()
    except Exception, e:
        print ('imstats: can\'t convert “%s” to simple 2D sky image; '
               'taking first plane' % path, file=sys.stderr)
        data = img.read ()[tuple (np.zeros (img.shape.size - 2))]
    else:
        data = img.read ()

    h, w = data.shape
    patchhalfsize = 32

    p = data[h//2 - patchhalfsize:h//2 + patchhalfsize,
             w//2 - patchhalfsize:w//2 + patchhalfsize]

    mx = p.max ()
    mn = p.min ()
    med = np.median (p)
    rms = np.sqrt ((p**2).mean ())

    sc = max (abs (mx), abs (mn))
    if sc <= 0:
        expt = 0
    else:
        expt = 3 * (int (np.floor (np.log10 (sc))) // 3)
    f = 10**-expt

    print ('min  = %.2f * 10^%d' % (f * mn, expt))
    print ('max  = %.2f * 10^%d' % (f * mx, expt))
    print ('med  = %.2f * 10^%d' % (f * med, expt))
    print ('rms  = %.2f * 10^%d' % (f * rms, expt))


def cmd_stats (args):
    if len (args) == 1:
        _stats_print (args[0])
    else:
        for i, path in enumerate (args):
            if i > 0:
                print ()
            print ('path =', path)
            _stats_print (path)
cmd_stats.argspec = '<images...>'
cmd_stats.summary = 'Compute and print statistics of a 64×64 patch at image center.'


# The driver.

def _fullusage ():
    usagestr = """imtool <command> [arguments...]

This is a tool for miscellaneous operations on astronomical images.

Subcommands are:

"""

    g = globals ()
    cnames = sorted (n for n in g.iterkeys () if n.startswith ('cmd_'))

    for cname in cnames:
        usagestr += '  imtool %-8s - %s\n' % (cname[4:], g[cname].summary)

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
        print ('usage: imtool', cmdname, func.argspec)
        print ()
        print (func.summary)
        if hasattr (func, 'moredocs'):
            print ()
            print (func.moredocs)
        return

    try:
        func (args)
    except UsageError as e:
        print ('error:', e, '\n\nusage: imtool', cmdname, func.argspec,
               file=sys.stderr)
