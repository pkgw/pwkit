# -*- mode: python; coding: utf-8 -*-
# Copyright 2015, 2017 Peter Williams <peter@newton.cx> and collaborators
# Licensed under the MIT License

"""This file is a casapy script. Do not use it as a module.

It is also not intended to be invoked directly through pkcasascript. See
`pwkit.environments.casa.tasks.gencal`.

"""

def in_casapy(helper, vis=None):
    """This function is run inside the weirdo casapy IPython environment! A
    strange set of modules is available, and the
    `pwkit.environments.casa.scripting` system sets up a very particular
    environment to allow encapsulated scripting.

    """
    import numpy as np, sys
    from correct_ant_posns import correct_ant_posns

    info = correct_ant_posns(vis, False)
    if len(info) != 3 or info[0] != 0 or not len(info[1]):
        helper.die('failed to fetch VLA antenna positions; got %r', info)

    antenna = info[1]
    parameter = info[2]

    with open(helper.temppath('info.npy'), 'wb') as f:
        np.save(f, antenna)
        np.save(f, parameter)
