# -*- mode: python; coding: utf-8 -*-
# Copyright 2017 Peter Williams <peter@newton.cx> and collaborators.
# Licensed under the MIT License.

"""This module contains helpers for modeling X-ray spectra with the `Sherpa
<http://cxc.harvard.edu/sherpa/>`_ package.

"""
from __future__ import absolute_import, division, print_function

__all__ = '''
FilterAdditionHack
get_source_qq_data
get_bkg_qq_data
make_qq_plot
'''.split()

from sherpa.astro import ui
from sherpa.models import CompositeModel, ArithmeticModel
import numpy as np


class FilterAdditionHack(CompositeModel, ArithmeticModel):
    """Create a new model that adds together two models, one filtered through
    instrumental response functions, one not.

    *dataobj*
      An object representing the input data; in standard usage, this is the
      result of calling :func:`sherpa.astro.ui.get_data`. This is needed
      to implement the hack.
    *lhs*
      A source model that *is* filtered through the telescope's response function.
    *rhs*
      A source model that is *not* filtered through the telescope's response function.

    As of version 4.9, Sherpa has some problems when combining the
    ``set_{bkg_,}_full_model`` functions with energy filtering. There is a
    relevant-looking bug notice `here
    <http://cxc.harvard.edu/sherpa/bugs/set_full_model.html>`_, although I
    think that I might be seeing a *slightly* different problem than the one
    that that page describes. Regardless, I can't get their suggested fix to
    work. Looking at how the background model is evaluated, I don't see how
    their suggested fix can be relevant, either. This class implements my
    workaround, which hopefully isn't totally crazy.

    """
    def __init__(self, dataobj, lhs, rhs):
        self.dataobj = dataobj
        self.lhs = lhs
        self.rhs = rhs
        self.op = np.add
        CompositeModel.__init__(self, '%s + %s' % (lhs.name, rhs.name), (lhs, rhs))

    def startup(self):
        self.lhs.startup()
        self.rhs.startup()
        CompositeModel.startup(self)

    def teardown(self):
        self.lhs.teardown()
        self.rhs.teardown()
        CompositeModel.teardown(self)

    def calc(self, p, *args, **kwargs):
        nlhs = len(self.lhs.pars)
        lhs = self.lhs.calc(p[:nlhs], *args, **kwargs)
        rhs = self.rhs.calc(p[nlhs:], *args, **kwargs)

        # the hack!
        lhs = self.dataobj.apply_filter(lhs)

        return self.op(lhs, rhs)


def get_source_qq_data():
    """Get data for a quantile-quantile plot of the source data and model.

    The inputs are implicit; the data are obtained from the current state of
    the Sherpa ``ui`` module.

    Returns an array of shape ``(3, npts)``. The first slice is the energy
    axis in keV; the second is the observed values in each bin (counts, or
    rate, or rate per keV, etc.); the third is the corresponding model value
    in each bin.

    """
    sdata = ui.get_data()
    kev = sdata.get_x()
    obs_data = sdata.counts
    model_data = ui.get_model()(kev)
    return np.vstack((kev, obs_data, model_data))


def get_bkg_qq_data():
    """Get data for a quantile-quantile plot of the background data and model.

    The inputs are implicit; the data are obtained from the current state of
    the Sherpa ``ui`` module.

    Returns an array of shape ``(3, npts)``. The first slice is the energy
    axis in keV; the second is the observed values in each bin (counts, or
    rate, or rate per keV, etc.); the third is the corresponding model value
    in each bin.

    """
    bdata = ui.get_bkg()
    bfp = ui.get_bkg_fit_plot()
    kev = bdata.get_x()
    obs_data = bdata.get_y()
    model_data = bfp.modelplot.y
    return np.vstack((kev, obs_data, model_data))


def make_qq_plot(kev, obs, mdl, unit, key_text):
    """Make a quantile-quantile plot comparing events and a model.

    *kev*
      A 1D, sorted array of event energy bins measured in keV.
    *obs*
      A 1D array giving the number or rate of events in each bin.
    *mdl*
      A 1D array giving the modeled number or rate of events in each bin.
    *unit*
      Text describing the unit in which *obs* and *mdl* are measured; will
      be shown on the plot axes.
    *key_text*
      Text describing the quantile-quantile comparison quantity; will be
      shown on the plot legend.

    Returns an :mod:`omega.RectPlot` instance.

    *TODO*: nothing about this is Sherpa-specific. Same goes for some of the
    plotting routines in :mod:`pkwit.environments.casa.data`; might be
    reasonable to add a submodule for generic X-ray-y plotting routines.

    """
    import omega as om

    kev = np.asarray(kev)
    obs = np.asarray(obs)
    mdl = np.asarray(mdl)

    c_obs = np.cumsum(obs)
    c_mdl = np.cumsum(mdl)
    mx = max(c_obs[-1], c_mdl[-1])

    p = om.RectPlot()
    p.addXY([0, mx], [0, mx], '1:1')
    p.addXY(c_mdl, c_obs, key_text)

    locs = np.linspace(0., kev.size - 2, 10)
    c0 = mx * 1.05
    c1 = mx * 1.1

    for loc in locs:
        i0 = int(np.floor(loc))
        frac = loc - i0
        kevval = (1 - frac) * kev[i0] + frac * kev[i0+1]
        mdlval = (1 - frac) * c_mdl[i0] + frac * c_mdl[i0+1]
        obsval = (1 - frac) * c_obs[i0] + frac * c_obs[i0+1]
        p.addXY([mdlval, mdlval], [c0, c1], '%.2f keV' % kevval, dsn=2)
        p.addXY([c0, c1], [obsval, obsval], None, dsn=2)

    p.setLabels('Cumulative model ' + unit, 'Cumulative data ' + unit)
    p.defaultKeyOverlay.vAlign = 0.3
    return p
