# -*- mode: python; coding: utf-8 -*-
# Copyright 2015 Peter Williams <peter@newton.cx> and collaborators
# Licensed under the MIT License

"""This file is a casapy script. Do not use it as a module.

It is also useless to run directly via pkcasascript. Use
`pwkit.environments.casa.tasks.getopacities`.

"""

def in_casapy (helper, args):
    """This function is run inside the weirdo casapy IPython environment! A
    strange set of modules is available, and the
    `pwkit.environments.casa.scripting` system sets up a very particular
    environment to allow encapsulated scripting.

    """
    import numpy as np, os, cPickle as pickle

    if len (args) != 2:
        helper.die ('usage: cscript_getopacities.py <MS> <plotdest>')

    ms = args[0]
    plotdest = args[1]

    opac = helper.casans.plotweather (vis=ms)

    opac = np.asarray (opac)
    with open (helper.temppath ('opac.npy'), 'wb') as f:
        pickle.dump (opac, f)

    os.rename (ms + '.plotweather.png', plotdest)
