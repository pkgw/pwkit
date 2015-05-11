# -*- mode: python; coding: utf-8 -*-
# Copyright 2015 Peter Williams <peter@newton.cx> and collaborators
# Licensed under the MIT License

"""This file is a casapy script. Do not use it as a module."""

def in_casapy (helper, args):
    """This function is run inside the weirdo casapy IPython environment! A
    strange set of modules is available, and the
    `pwkit.environments.casa.scripting` system sets up a very particular
    environment to allow encapsulated scripting.

    """
    import numpy as np, os

    if len (args) != 3:
        helper.die ('usage: getopacities <MS> <spwwidths> <plotdest>')

    ms = args[0]
    spwwidths = [int (w) for w in args[1].split (',')]
    plotdest = args[2]

    opac = helper.casans.plotweather (vis=ms)

    averaged = []
    idx = 0

    for width in spwwidths:
        a = np.asarray (opac[idx:idx+width])
        averaged.append (a.mean ())
        idx += width

    helper.log ('opacity = [%s]', ', '.join ('%.5f' % q for q in averaged))
    os.rename (ms + '.plotweather.png', plotdest)
