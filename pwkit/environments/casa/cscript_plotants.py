# -*- mode: python; coding: utf-8 -*-
# Copyright 2015 Peter Williams <peter@newton.cx> and collaborators
# Licensed under the MIT License

"""This file is a casapy script. Do not use it as a module.

It is also not intended to be invoked directly through pkcasascript. See
`pwkit.environments.casa.tasks.plotcal`.

"""

def in_casapy (helper, vis=None, figfile=None):
    """This function is run inside the weirdo casapy IPython environment! A
    strange set of modules is available, and the
    `pwkit.environments.casa.scripting` system sets up a very particular
    environment to allow encapsulated scripting.

    """
    if vis is None:
        raise ValueError ('vis')

    helper.casans.plotants (vis=vis, figfile=figfile)
