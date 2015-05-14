# -*- mode: python; coding: utf-8 -*-
# Copyright 2015 Peter Williams <peter@newton.cx> and collaborators
# Licensed under the MIT License

"""This file is a casapy script. Do not use it as a module.

It is also not intended to be invoked directly through pkcasascript. See
`pwkit.environments.casa.tasks.plotcal`.

"""

def in_casapy (helper, caltable=None, selectcals={}, plotoptions={},
               xaxis=None, yaxis=None, figfile=None):
    """This function is run inside the weirdo casapy IPython environment! A
    strange set of modules is available, and the
    `pwkit.environments.casa.scripting` system sets up a very particular
    environment to allow encapsulated scripting.

    """
    if caltable is None:
        raise ValueError ('caltable')

    show_gui = (figfile is None)
    cp = helper.casans.cp

    helper.casans.tp.setgui (show_gui)
    cp.open (caltable)
    cp.selectcal (**selectcals)
    cp.plotoptions (**plotoptions)
    cp.plot (xaxis, yaxis)

    if show_gui:
        import pylab as pl
        pl.show ()
    else:
        cp.savefig (figfile)

