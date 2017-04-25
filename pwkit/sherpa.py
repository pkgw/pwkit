# -*- mode: python; coding: utf-8 -*-
# Copyright 2017 Peter Williams <peter@newton.cx> and collaborators.
# Licensed under the MIT License.

"""This module contains helpers for modeling X-ray spectra with the `Sherpa
<http://cxc.harvard.edu/sherpa/>`_ package.

"""
from __future__ import absolute_import, division, print_function

__all__ = '''
FilterAdditionHack
expand_rmf_matrix
get_source_qq_data
get_bkg_qq_data
make_qq_plot
make_spectrum_plot
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
        old_shape = lhs.shape
        lhs = self.dataobj.apply_filter(lhs)
        print('CC', old_shape, lhs.shape, rhs.shape)

        return self.op(lhs, rhs)


def expand_rmf_matrix(rmf):
    """Expand an RMF matrix stored in compressed form.

    *rmf*
      An RMF object as might be returned by ``sherpa.astro.ui.get_rmf()``.

    Returns: a non-sparse RMF matrix

    The Response Matrix Function (RMF) of an X-ray telescope like Chandra can
    be stored in a sparse format as defined in `OGIP Calibration Memo
    CAL/GEN/92-002
    <https://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/docs/memos/cal_gen_92_002/cal_gen_92_002.html>`_.
    For visualization and analysis purposes, it can be useful to de-sparsify
    the matrices stored in this way. This function does that, returning a
    two-dimensional Numpy array.

    """
    n_chan = (rmf.n_chan + rmf.f_chan).max()
    n_energy = rmf.n_grp.size

    expanded = np.zeros((n_energy, n_chan))
    mtx_ofs = 0
    grp_ofs = 0

    for i in range(n_energy):
        for j in range(rmf.n_grp[i]):
            f = rmf.f_chan[grp_ofs]
            n = rmf.n_chan[grp_ofs]
            expanded[i,f:f+n] = rmf.matrix[mtx_ofs:mtx_ofs+n]
            mtx_ofs += n
            grp_ofs += 1

    return expanded


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


def make_spectrum_plot(model_plot, data_plot, desc, xmin_clamp=0.01,
                       min_valid_x=None, max_valid_x=None):
    """Make a plot of a spectral model and data.

    *model_plot*
      A model plot object returned by Sherpa from a call like `ui.get_model_plot()`
      or `ui.get_bkg_model_plot()`.
    *data_plot*
      A data plot object returned by Sherpa from a call like `ui.get_source_plot()`
      or `ui.get_bkg_plot()`.
    *desc*
      Text describing the origin of the data; will be shown in the plot legend
      (with "Model" and "Data" appended).
    *xmin_clamp*
      The smallest "x" (energy axis) value that will be plotted; default is 0.01.
      This is needed to allow the plot to be shown on a logarithmic scale if
      the energy axes of the model go all the way to 0.
    *min_valid_x*
      Either None, or the smallest "x" (energy axis) value in which the model and
      data are valid; this could correspond to a range specified in the "notice"
      command during analysis. If specified, a gray band will be added to the plot
      showing the invalidated regions.
    *max_valid_x*
      Like *min_valid_x* but for the largest "x" (energy axis) value in which the
      model and data are valid.

    Returns ``(plot, xlow, xhigh)``, where *plot* an OmegaPlot RectPlot instance,
    *xlow* is the left edge of the plot bounds, and *xhigh* is the right edge of
    the plot bounds.

    The plot bounds are

    """
    import omega as om

    model_x = np.concatenate((model_plot.xlo, [model_plot.xhi[-1]]))
    model_x[0] = max(model_x[0], xmin_clamp)
    model_y = np.concatenate((model_plot.y, [0.]))

    data_left_edges = data_plot.x - 0.5 * data_plot.xerr
    data_left_edges[0] = max(data_left_edges[0], xmin_clamp)
    data_hist_x = np.concatenate((data_left_edges, [data_plot.x[-1] + 0.5 * data_plot.xerr[-1]]))
    data_hist_y = np.concatenate((data_plot.y, [0.]))

    log_bounds_pad_factor = 0.9
    xlow = model_x[0] * log_bounds_pad_factor
    xhigh = model_x[-1] / log_bounds_pad_factor

    p = om.RectPlot()

    if min_valid_x is not None:
        p.add(om.rect.XBand(1e-3 * xlow, min_valid_x, keyText=None), zheight=-1, dsn=1)
    if max_valid_x is not None:
        p.add(om.rect.XBand(max_valid_x, xhigh * 1e3, keyText=None), zheight=-1, dsn=1)

    csp = om.rect.ContinuousSteppedPainter(keyText=desc + ' Model')
    csp.setFloats(model_x, model_y)
    p.add(csp)

    csp = om.rect.ContinuousSteppedPainter(keyText=None)
    csp.setFloats(data_hist_x, data_hist_y)
    p.add(csp)
    p.addXYErr(data_plot.x, data_plot.y, data_plot.yerr, desc + ' Data', lines=0, dsn=1)

    p.setLabels(data_plot.xlabel, data_plot.ylabel)
    p.setLinLogAxes(True, False)
    p.setBounds (xlow, xhigh)
    return p, xlow, xhigh
