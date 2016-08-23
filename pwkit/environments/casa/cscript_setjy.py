# -*- mode: python; coding: utf-8 -*-
# Copyright 2016 Peter Williams <peter@newton.cx> and collaborators
# Licensed under the MIT License

"""This file is a casapy script. Do not use it as a module.

It is also not intended to be invoked directly through pkcasascript. See
`pwkit.environments.casa.tasks:setjy`.

We can invoke the C++ code directly for most setjy functionality, but the
current solar system model implementation is based on a big batch of Python
code in the CASA distribution. So when those are requested, we farm out to
that.

"""

def in_casapy (helper, **kwargs):
    """This function is run inside the weirdo casapy IPython environment! A
    strange set of modules is available, and the
    `pwkit.environments.casa.scripting` system sets up a very particular
    environment to allow encapsulated scripting.

    """
    helper.casans.setjy (**kwargs)
