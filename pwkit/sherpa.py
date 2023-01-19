# -*- mode: python; coding: utf-8 -*-
# Copyright 2017 Peter Williams <peter@newton.cx> and collaborators.
# Licensed under the MIT License.

"""This module contains helpers for modeling X-ray spectra with the `Sherpa
<http://cxc.harvard.edu/sherpa/>`_ package.

"""
from __future__ import absolute_import, division, print_function

__all__ = '''
PowerLawApecDemModel
make_fixed_temp_multi_apec
expand_rmf_matrix
derive_identity_rmf
derive_identity_arf
get_source_qq_data
get_bkg_qq_data
make_qq_plot
make_multi_qq_plots
make_spectrum_plot
make_multi_spectrum_plots
'''.split()

import numpy as np
from sherpa.astro import ui
from sherpa.astro.xspec import XSAdditiveModel, _xspec
from sherpa.models import Parameter
from sherpa.models.parameter import hugeval


# Some helpful models

DEFAULT_KT_ARRAY = np.logspace(-1.5, 1, 20)

class PowerLawApecDemModel(XSAdditiveModel):
    """A model with contributions from APEC plasmas at a range of
    temperatures, scaling with temperature.

    Constructor arguments are:

    *name*
      The Sherpa name of the resulting model instance.
    *kt_array* = None
      An array of temperatures to use for the plasma models. If left at the
      default of None, a hard-coded default is used that spans temperatures of
      ~0.03 to 10 keV with logarithmic spacing.

    The contribution at each temperature scales with kT as a power law. The
    model parameters are:

    *gfac*
      The power-law normalization parameter. The contribution at temperature *kT*
      is ``norm * kT**gfac``.
    *Abundanc*
      The standard APEC abundance parameter.
    *redshift*
      The standard APEC redshift parameter.
    *norm*
      The standard overall normalization parameter.

    This model is only efficient to compute if *Abundanc* and *redshift* are
    frozen.

    """
    def __init__(self, name, kt_array=None):
        if kt_array is None:
            kt_array = DEFAULT_KT_ARRAY
        else:
            kt_array = np.atleast_1d(np.asfarray(kt_array))

        self.gfac = Parameter(name, 'gfac', 0.5, 1e-4, 1e4, 1e-6, 1e6)
        self.Abundanc = Parameter(name, 'Abundanc', 1., 0., 5., 0.0, hugeval, frozen=True)
        self.redshift = Parameter(name, 'redshift', 0., -0.999, 10., -0.999, hugeval, frozen=True)
        self.norm = Parameter(name, 'norm', 1.0, 0.0, 1e24, 0.0, hugeval)

        self._kt_array = kt_array
        self._cur_cache_key = None
        self._cached_vals = None
        XSAdditiveModel.__init__(self, name, (self.gfac, self.Abundanc, self.redshift, self.norm))

    def _calc(self, params, *args, **kwargs):
        gfac, abund, redshift, norm = params
        cache_key = (abund, redshift)

        if self._cur_cache_key != cache_key:
            self._cached_vals = [None] * self._kt_array.size

            for i in range(self._kt_array.size):
                apec_params = [self._kt_array[i], abund, redshift, 1.]
                self._cached_vals[i] = _xspec.xsaped(apec_params, *args, **kwargs)

            self._cur_cache_key = cache_key
            self._cached_vals = np.array(self._cached_vals).T

        scales = norm * self._kt_array**gfac
        return (self._cached_vals * scales).sum(axis=1)

ui.add_model(PowerLawApecDemModel)


def make_fixed_temp_multi_apec(kTs, name_template='apec%d', norm=None):
    """Create a model summing multiple APEC components at fixed temperatures.

    *kTs*
      An iterable of temperatures for the components, in keV.
    *name_template* = 'apec%d'
      A template to use for the names of each component; it is string-formatted
      with the 0-based component number as an argument.
    *norm* = None
      An initial normalization to be used for every component, or None to use
      the Sherpa default.
    Returns:
      A tuple ``(total_model, sub_models)``, where *total_model* is a Sherpa
      model representing the sum of the APEC components and *sub_models* is
      a list of the individual models.

    This function creates a vector of APEC model components and sums them.
    Their *kT* parameters are set and then frozen (using
    :func:`sherpa.astro.ui.freeze`), so that upon exit from this function, the
    amplitude of each component is the only free parameter.

    """
    total_model = None
    sub_models = []

    for i, kT in enumerate(kTs):
        component = ui.xsapec(name_template % i)
        component.kT = kT
        ui.freeze(component.kT)
        if norm is not None:
            component.norm = norm
        sub_models.append(component)

        if total_model is None:
            total_model = component
        else:
            total_model = total_model + component

    return total_model, sub_models


def expand_rmf_matrix(rmf):
    """Expand an RMF matrix stored in compressed form.

    *rmf*
      An RMF object as might be returned by ``sherpa.astro.ui.get_rmf()``.
    Returns:
      A non-sparse RMF matrix.

    The Response Matrix Function (RMF) of an X-ray telescope like Chandra can
    be stored in a sparse format as defined in `OGIP Calibration Memo
    CAL/GEN/92-002
    <https://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/docs/memos/cal_gen_92_002/cal_gen_92_002.html>`_.
    For visualization and analysis purposes, it can be useful to de-sparsify
    the matrices stored in this way. This function does that, returning a
    two-dimensional Numpy array.

    """
    n_chan = rmf.e_min.size
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


def derive_identity_rmf(name, rmf):
    """Create an "identity" RMF that does not mix energies.

    *name*
      The name of the RMF object to be created; passed to Sherpa.
    *rmf*
      An existing RMF object on which to base this one.
    Returns:
      A new RMF1D object that has a response matrix that is as close to
      diagonal as we can get in energy space, and that has a constant
      sensitivity as a function of detector channel.

    In many X-ray observations, the relevant background signal does not behave
    like an astrophysical source that is filtered through the telescope's
    response functions. However, I have been unable to get current Sherpa
    (version 4.9) to behave how I want when working with backround models that
    are *not* filtered through these response functions. This function
    constructs an "identity" RMF response matrix that provides the best
    possible approximation of a passthrough "instrumental response": it mixes
    energies as little as possible and has a uniform sensitivity as a function
    of detector channel.

    """
    from sherpa.astro.data import DataRMF
    from sherpa.astro.instrument import RMF1D

    # The "x" axis of the desired matrix -- the columnar direction; axis 1 --
    # is "channels". There are n_chan of them and each maps to a notional
    # energy range specified by "e_min" and "e_max".
    #
    # The "y" axis of the desired matrix -- the row direction; axis 1 -- is
    # honest-to-goodness energy. There are tot_n_energy energy bins, each
    # occupying a range specified by "energ_lo" and "energ_hi".
    #
    # We want every channel that maps to a valid output energy to have a
    # nonzero entry in the matrix. The relative sizes of n_energy and n_cell
    # can vary, as can the bounds of which regions of each axis can be validly
    # mapped to each other. So this problem is basically equivalent to that of
    # drawing an arbitrary pixelated line on bitmap, without anti-aliasing.
    #
    # The output matrix is represented in a row-based sparse format.
    #
    # - There is a integer vector "n_grp" of size "n_energy". It gives the
    #   number of "groups" needed to fill in each row of the matrix. Let
    #   "tot_groups = sum(n_grp)". For a given row, "n_grp[row_index]" may
    #   be zero, indicating that the row is all zeros.
    # - There are integer vectors "f_chan" and "n_chan", each of size
    #   "tot_groups", that define each group. "f_chan" gives the index of
    #   the first channel column populated by the group; "n_chan" gives the
    #   number of columns populated by the group. Note that there can
    #   be multiple groups for a single row, so successive group records
    #   may fill in different pieces of the same row.
    # - Let "tot_cells = sum(n_chan)".
    # - There is a vector "matrix" of size "tot_cells" that stores the actual
    #   matrix data. This is just a concatenation of all the data corresponding
    #   to each group.
    # - Unpopulated matrix entries are zero.
    #
    # See expand_rmf_matrix() for a sloppy implementation of how to unpack
    # this sparse format.

    n_chan = rmf.e_min.size
    n_energy = rmf.energ_lo.size

    c_lo_offset = rmf.e_min[0]
    c_lo_slope = (rmf.e_min[-1] - c_lo_offset) / (n_chan - 1)

    c_hi_offset = rmf.e_max[0]
    c_hi_slope = (rmf.e_max[-1] - c_hi_offset) / (n_chan - 1)

    e_lo_offset = rmf.energ_lo[0]
    e_lo_slope = (rmf.energ_lo[-1] - e_lo_offset) / (n_energy - 1)

    e_hi_offset = rmf.energ_hi[0]
    e_hi_slope = (rmf.energ_hi[-1] - e_hi_offset) / (n_energy - 1)

    all_e_indices = np.arange(n_energy)
    all_e_los = e_lo_slope * all_e_indices + e_lo_offset
    start_chans = np.floor((all_e_los - c_lo_offset) / c_lo_slope).astype(int)

    all_e_his = e_hi_slope * all_e_indices + e_hi_offset
    stop_chans = np.ceil((all_e_his - c_hi_offset) / c_hi_slope).astype(int)

    first_e_index_on_channel_grid = 0
    while stop_chans[first_e_index_on_channel_grid] < 0:
        first_e_index_on_channel_grid += 1

    last_e_index_on_channel_grid = n_energy - 1
    while start_chans[last_e_index_on_channel_grid] >= n_chan:
        last_e_index_on_channel_grid -= 1

    n_nonzero_rows = last_e_index_on_channel_grid + 1 - first_e_index_on_channel_grid
    e_slice = slice(first_e_index_on_channel_grid, last_e_index_on_channel_grid + 1)
    n_grp = np.zeros(n_energy, dtype=int)
    n_grp[e_slice] = 1

    start_chans = np.maximum(start_chans[e_slice], 0)
    stop_chans = np.minimum(stop_chans[e_slice], n_chan - 1)

    # We now have a first cut at a row-oriented expression of our "identity"
    # RMF. However, it's conservative. Trim down to eliminate overlaps between
    # sequences.

    for i in range(n_nonzero_rows - 1):
        my_end = stop_chans[i]
        next_start = start_chans[i+1]
        if next_start <= my_end:
            stop_chans[i] = max(start_chans[i], next_start - 1)

    # Results are funky unless the sums along the vertical axis are constant.
    # Ideally the sum along the *horizontal* axis would add up to 1 (since,
    # ideally, each row is a probability distribution), but it is not
    # generally possible to fulfill both of these constraints simultaneously.
    # The latter constraint does not seem to matter in practice so we ignore it.
    # Due to the funky encoding of the matrix, we need to build a helper table
    # to meet the vertical-sum constraint.

    counts = np.zeros(n_chan, dtype=int)

    for i in range(n_nonzero_rows):
        counts[start_chans[i]:stop_chans[i]+1] += 1

    counts[:start_chans.min()] = 1
    counts[stop_chans.max()+1:] = 1
    assert (counts > 0).all()

    # We can now build the matrix.

    f_chan = start_chans
    rmfnchan = stop_chans + 1 - f_chan
    assert (rmfnchan > 0).all()

    matrix = np.zeros(rmfnchan.sum())
    amounts = 1. / counts
    ofs = 0

    for i in range(n_nonzero_rows):
        f = f_chan[i]
        n = rmfnchan[i]
        matrix[ofs:ofs+n] = amounts[f:f+n]
        ofs += n

    # All that's left to do is create the Python objects.

    drmf = DataRMF(
        name,
        rmf.detchans,
        rmf.energ_lo,
        rmf.energ_hi,
        n_grp,
        f_chan,
        rmfnchan,
        matrix,
        offset = 0,
        e_min = rmf.e_min,
        e_max = rmf.e_max,
        header = None
    )

    return RMF1D(drmf, pha=rmf._pha)


def derive_identity_arf(name, arf):
    """Create an "identity" ARF that has uniform sensitivity.

    *name*
      The name of the ARF object to be created; passed to Sherpa.
    *arf*
      An existing ARF object on which to base this one.
    Returns:
      A new ARF1D object that has a uniform spectral response vector.

    In many X-ray observations, the relevant background signal does not behave
    like an astrophysical source that is filtered through the telescope's
    response functions. However, I have been unable to get current Sherpa
    (version 4.9) to behave how I want when working with backround models that
    are *not* filtered through these response functions. This function
    constructs an "identity" ARF response function that has uniform sensitivity
    as a function of detector channel.

    """
    from sherpa.astro.data import DataARF
    from sherpa.astro.instrument import ARF1D

    darf = DataARF(
        name,
        arf.energ_lo,
        arf.energ_hi,
        np.ones(arf.specresp.shape),
        arf.bin_lo,
        arf.bin_hi,
        arf.exposure,
        header = None,
    )
    return ARF1D(darf, pha=arf._pha)


def get_source_qq_data(id=None):
    """Get data for a quantile-quantile plot of the source data and model.

    *id*
      The dataset id for which to get the data; defaults if unspecified.
    Returns:
      An ndarray of shape ``(3, npts)``. The first slice is the energy axis in
      keV; the second is the observed values in each bin (counts, or rate, or
      rate per keV, etc.); the third is the corresponding model value in each
      bin.

    The inputs are implicit; the data are obtained from the current state of
    the Sherpa ``ui`` module.

    """
    sdata = ui.get_data(id=id)
    kev = sdata.get_x()
    obs_data = sdata.counts
    model_data = ui.get_model(id=id)(kev)
    return np.vstack((kev, obs_data, model_data))


def get_bkg_qq_data(id=None, bkg_id=None):
    """Get data for a quantile-quantile plot of the background data and model.

    *id*
      The dataset id for which to get the data; defaults if unspecified.
    *bkg_id*
      The identifier of the background; defaults if unspecified.
    Returns:
      An ndarray of shape ``(3, npts)``. The first slice is the energy axis in
      keV; the second is the observed values in each bin (counts, or rate, or
      rate per keV, etc.); the third is the corresponding model value in each
      bin.

    The inputs are implicit; the data are obtained from the current state of
    the Sherpa ``ui`` module.

    """
    bdata = ui.get_bkg(id=id, bkg_id=bkg_id)
    kev = bdata.get_x()
    obs_data = bdata.counts
    model_data = ui.get_bkg_model(id=id, bkg_id=bkg_id)(kev)
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
    Returns:
      An :class:`omega.RectPlot` instance.

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

    # HACK: this range of numbers is chosen to give reasonable sampling for my
    # sources, which are typically quite soft.
    locs = np.array([0, 0.05, 0.08, 0.11, 0.17, 0.3, 0.4, 0.7, 1]) * (kev.size - 2)
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


def make_multi_qq_plots(arrays, key_text):
    """Make a quantile-quantile plot comparing multiple sets of events and models.

    *arrays*
      X.
    *key_text*
      Text describing the quantile-quantile comparison quantity; will be
      shown on the plot legend.
    Returns:
      An :class:`omega.RectPlot` instance.

    *TODO*: nothing about this is Sherpa-specific. Same goes for some of the
    plotting routines in :mod:`pkwit.environments.casa.data`; might be
    reasonable to add a submodule for generic X-ray-y plotting routines.

    *TODO*: Some gross code duplication here.

    """
    import omega as om

    p = om.RectPlot()
    p.addXY([0, 1.], [0, 1.], '1:1')

    for index, array in enumerate(arrays):
        kev, obs, mdl = array
        c_obs = np.cumsum(obs)
        c_mdl = np.cumsum(mdl)

        mx = 0.5 * (c_obs[-1] + c_mdl[-1])
        c_obs /= mx
        c_mdl /= mx

        p.addXY(c_mdl, c_obs, '%s #%d' % (key_text, index))

    # HACK: this range of numbers is chosen to give reasonable sampling for my
    # sources, which are typically quite soft.
    #
    # Note: this reuses the variables from the last loop iteration.
    locs = np.array([0, 0.05, 0.08, 0.11, 0.17, 0.3, 0.4, 0.7, 1]) * (kev.size - 2)
    c0 = 1.05
    c1 = 1.1

    for loc in locs:
        i0 = int(np.floor(loc))
        frac = loc - i0
        kevval = (1 - frac) * kev[i0] + frac * kev[i0+1]
        mdlval = (1 - frac) * c_mdl[i0] + frac * c_mdl[i0+1]
        obsval = (1 - frac) * c_obs[i0] + frac * c_obs[i0+1]
        p.addXY([mdlval, mdlval], [c0, c1], '%.2f keV' % kevval, dsn=2)
        p.addXY([c0, c1], [obsval, obsval], None, dsn=2)

    p.setLabels('Cumulative rescaled model', 'Cumulative rescaled data')
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
    Returns:
      A tuple ``(plot, xlow, xhigh)``, where *plot* an OmegaPlot RectPlot
      instance, *xlow* is the left edge of the plot bounds, and *xhigh* is the
      right edge of the plot bounds.

    """
    import omega as om

    model_x = np.concatenate((model_plot.xlo, [model_plot.xhi[-1]]))
    model_x[0] = max(model_x[0], xmin_clamp)
    model_y = np.concatenate((model_plot.y, [0.]))

    # Sigh, sometimes Sherpa gives us bad values.
    is_bad = ~np.isfinite(model_y)
    if is_bad.sum():
        from .cli import warn
        warn('bad Sherpa model Y value(s) at: %r', np.where(is_bad)[0])
        model_y[is_bad] = 0

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


def make_multi_spectrum_plots(model_plot, plotids, data_getter, desc, xmin_clamp=0.01,
                              min_valid_x=None, max_valid_x=None):
    """Make a plot of multiple spectral models and data.

    *model_plot*
      A model plot object returned by Sherpa from a call like
      ``ui.get_model_plot()`` or ``ui.get_bkg_model_plot()``.
    *data_plots*
      An iterable of data plot objects returned by Sherpa from calls like
      ``ui.get_source_plot(id)`` or ``ui.get_bkg_plot(id)``.
    *desc*
      Text describing the origin of the data; will be shown in the plot legend
      (with "Model" and "Data #<number>" appended).
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
    Returns:
      A tuple ``(plot, xlow, xhigh)``, where *plot* an OmegaPlot RectPlot
      instance, *xlow* is the left edge of the plot bounds, and *xhigh* is the
      right edge of the plot bounds.

    TODO: not happy about the code duplication with :func:`make_spectrum_plot`
    but here we are.

    """
    import omega as om
    from omega.stamps import DataThemedStamp, WithYErrorBars

    model_x = np.concatenate((model_plot.xlo, [model_plot.xhi[-1]]))
    model_x[0] = max(model_x[0], xmin_clamp)
    model_y = np.concatenate((model_plot.y, [0.]))

    # Sigh, sometimes Sherpa gives us bad values.
    is_bad = ~np.isfinite(model_y)
    if is_bad.sum():
        from .cli import warn
        warn('bad Sherpa model Y value(s) at: %r', np.where(is_bad)[0])
        model_y[is_bad] = 0

    p = om.RectPlot()
    data_csps = []
    data_lines = []
    xlow = xhigh = None

    for index, plotid in enumerate(plotids):
        data_plot = data_getter(plotid)
        data_left_edges = data_plot.x - 0.5 * data_plot.xerr
        data_left_edges[0] = max(data_left_edges[0], xmin_clamp)
        data_hist_x = np.concatenate((data_left_edges, [data_plot.x[-1] + 0.5 * data_plot.xerr[-1]]))
        data_hist_y = np.concatenate((data_plot.y, [0.]))

        if xlow is None:
            xlow = model_x[0]
            xhigh = model_x[-1]
        else:
            xlow = min(xlow, model_x[0])
            xhigh = max(xhigh, model_x[-1])

        csp = om.rect.ContinuousSteppedPainter(keyText=None)
        csp.setFloats(data_hist_x, data_hist_y)
        data_csps.append(csp)

        inner_stamp = DataThemedStamp(None)
        stamp = WithYErrorBars(inner_stamp)

        lines = om.rect.XYDataPainter(
            lines = False,
            pointStamp = stamp,
            keyText = '%s Data #%d' % (desc, index)
        )
        lines.setFloats(data_plot.x, data_plot.y,
                        data_plot.y + data_plot.yerr,
                        data_plot.y - data_plot.yerr)
        inner_stamp.setHolder(lines)
        data_lines.append(lines)

    log_bounds_pad_factor = 0.9
    xlow *= log_bounds_pad_factor
    xhigh /= log_bounds_pad_factor

    if min_valid_x is not None:
        p.add(om.rect.XBand(1e-3 * xlow, min_valid_x, keyText=None), zheight=-1, dsn=1)
    if max_valid_x is not None:
        p.add(om.rect.XBand(max_valid_x, xhigh * 1e3, keyText=None), zheight=-1, dsn=1)

    model_csp = om.rect.ContinuousSteppedPainter(keyText=desc + ' Model')
    model_csp.setFloats(model_x, model_y)
    p.add(model_csp)

    for index, (data_csp, lines) in enumerate(zip(data_csps, data_lines)):
        p.add(data_csp, dsn=index + 1)
        p.add(lines, dsn=index + 1)

    p.setLabels(data_plot.xlabel, data_plot.ylabel) # data_plot = last one from the for loop
    p.setLinLogAxes(True, False)
    p.setBounds (xlow, xhigh)
    return p, xlow, xhigh
