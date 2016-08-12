# -*- mode: python; coding: utf-8 -*-
# Copyright 2016 Peter Williams <peter@newton.cx> and collaborators
# Licensed under the MIT License

"""This file is a casapy script. Do not use it as a module.

It is also not intended to be invoked directly through pkcasascript. See
`pwkit.environments.casa.tasks.importalma`.

"""

def in_casapy (helper, asdm=None, ms=None):
    """This function is run inside the weirdo casapy IPython environment! A
    strange set of modules is available, and the
    `pwkit.environments.casa.scripting` system sets up a very particular
    environment to allow encapsulated scripting.

    """
    if asdm is None:
        raise ValueError ('asdm')
    if ms is None:
        raise ValueError ('ms')

    helper.casans.importasdm (
        asdm = asdm,
        vis = ms,
        asis = 'Antenna Station Receiver Source CalAtmosphere CalWVR CorrelatorMode SBSummary',
        bdfflags = True,
        lazy = False,
        process_caldevice = False,
    )
