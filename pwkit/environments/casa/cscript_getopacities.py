# -*- mode: python; coding: utf-8 -*-
# Copyright 2015 Peter Williams <peter@newton.cx> and collaborators
# Licensed under the MIT License

"""This file is a casapy script. Do not use it as a module.

It is also useless to run directly via pkcasascript. Use
`pwkit.environments.casa.tasks.getopacities`.

"""

def in_casapy(helper, ms=None, plotdest=None):
    """This function is run inside the weirdo casapy IPython environment! A
    strange set of modules is available, and the
    `pwkit.environments.casa.scripting` system sets up a very particular
    environment to allow encapsulated scripting.

    """
    import numpy as np, os

    if ms is None:
        raise ValueError('ms')
    if plotdest is None:
        raise ValueError('plotdest')

    opac = helper.casans.plotweather(vis=ms, plotName=plotdest)
    opac = np.asarray(opac)
    with open(helper.temppath('opac.npy'), 'wb') as f:
        np.save(f, opac)
