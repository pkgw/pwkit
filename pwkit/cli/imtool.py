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
from .. import ndshow_gtk2 as ndshow # one day: flexible backend


class UsageError (PKError):
    pass


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


def cmd_show (args):
    anyfailures = False

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
