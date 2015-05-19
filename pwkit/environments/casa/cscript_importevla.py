# -*- mode: python; coding: utf-8 -*-
# Copyright 2015 Peter Williams <peter@newton.cx> and collaborators
# Licensed under the MIT License

"""This file is a casapy script. Do not use it as a module.

It is also not intended to be invoked directly through pkcasascript. See
`pwkit.environments.casa.tasks.importevla`.

"""

def in_casapy (helper, asdm=None, ms=None, tbuff=None):
    """This function is run inside the weirdo casapy IPython environment! A
    strange set of modules is available, and the
    `pwkit.environments.casa.scripting` system sets up a very particular
    environment to allow encapsulated scripting.

    """
    if asdm is None:
        raise ValueError ('asdm')
    if ms is None:
        raise ValueError ('ms')
    if tbuff is None:
        raise ValueError ('tbuff')

    helper.casans.importevla (asdm=asdm, vis=ms, ocorr_mode='co', online=True,
                              tbuff=tbuff, flagpol=False, tolerance=1.3,
                              applyflags=True, flagbackup=False)
