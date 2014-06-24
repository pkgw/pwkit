# -*- mode: python; coding: utf-8 -*-
# Copyright 2014 Peter Williams <peter@newton.cx> and collaborators.
# Licensed under the MIT License.

"""pwkit.cli.imtool - the 'imtool' program.

FIXME: the structure of this is all very redundant with astrotool.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = [b'commandline']

import numpy as np, sys

from . import *
from .. import astimage
from .. import ndshow_gtk2 as ndshow # one day: flexible backend


# The commands.

def cmd_show (args):
    anyfailures = False

    for path in args:
        try:
            img = astimage.open (path, 'r')
        except Exception, e:
            print ('pwshow: can\'t open path “%s”: %s' % (path, e), file=sys.stderr)
            anyfailures = True
            continue

        try:
            img = img.simple ()
        except Exception, e:
            print ('pwshow: can\'t convert “%s” to simple 2D sky image; taking '
                   ' first plane' % path, file=sys.stderr)
            data = img.read (flip=True)[tuple (np.zeros (img.shape.size - 2))]
            toworld = None
        else:
            data = img.read (flip=True)
            toworld = img.toworld

        ndshow.view (data, title=path + ' — Array Viewer',
                     toworld=toworld, yflip=True)
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
