# -*- mode: python; coding: utf-8 -*-
# Copyright 2016 Peter Williams <peter@newton.cx> and collaborators.
# Licensed under the MIT License.

"""Compute phase closure diagnostics from a Measurement Set

For most results to make sense, the data should be observations of a bright
point source at phase center.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = str("Config ClosureCalculator closures_cli").split()

import collections
import numpy as np

from ...cli import check_usage, die
from ...kwargv import ParseKeywords, Custom
from ...environments.casa import util
from ...environments.casa.util import sanitize_unicode as b

closures_doc = """
casatask closures vis=MS [keywords...]

THIS TASK IS CURRENTLY BROKEN AND WILL REFUSE TO RUN.

Plot diagnostics related to phase closure triple.

vis=
  Path of the MeasurementSet dataset to read. Required.

array=, baseline=, field=, observation=, polarization=, scan=,
scanintent=, spw=, taql=, time=, uvdist=
  MeasurementSet selectors used to filter the input data.
  Default polarization is 'RR,LL'. All polarizations are averaged
  together, so mixing parallel- and cross-hand pols is almost
  never what you want to do.

datacol=
  Name of the column to use for visibility data. Defaults to 'data'.
  You might want it to be 'corrected_data', although the point of
  phase closure analysis is that antenna-based calibrations don't
  affect its results ...
"""


class Config(ParseKeywords):
    vis = Custom(str, required=True)
    datacol = "data"

    # MeasurementSet filters
    array = str
    baseline = str
    field = str
    observation = str
    polarization = "RR,LL"
    scan = str
    scanintent = str
    spw = str
    taql = str
    time = str
    uvdist = str

    loglevel = "warn"


# ########################################################################
# Begin copying/emulating mirtask.util

FPOL_X = 0
FPOL_Y = 1
FPOL_R = 2
FPOL_L = 3
FPOL_I = 4
FPOL_Q = 5
FPOL_U = 6
FPOL_V = 7

fpol_names = "XYRLIQUV"

# This table helps split a CASA pol code into a pair of fpol values. The pair
# is packed into 8 bits, the upper 3 being for the left pol and the lower 4
# being for the right.

_pol_to_fpol = np.array(
    [
        0xFFFF,  # ?, illegal
        0x44,
        0x55,
        0x66,
        0x77,  # I Q U V
        0x22,
        0x23,
        0x32,
        0x33,  # RR RL LR LL
        0x00,
        0x01,
        0x10,
        0x11,  # XX XY YX YY
        0x20,
        0x21,
        0x30,
        0x31,  # RX RY LX LY
        0x02,
        0x03,
        0x12,
        0x13,  # XR XL YR YL
        # not bothering with the rest because seriously
    ]
)

pol_is_intensity = np.array(
    [
        False,  # ?, illegal
        True,
        True,
        True,
        True,  # I Q U V
        True,
        False,
        False,
        True,  # RR RL LR LL
        True,
        False,
        False,
        True,  # XX XY YX YY
        False,
        False,
        False,
        False,  # RX RY LX LY
        False,
        False,
        False,
        False,  # XR XL YR YL
        # not bothering with the rest because seriously
    ]
)

# This table performs the reverse mapping, with index being the two f-pol
# values packed into four bits each. A value of 0xFF indicates an illegal
# pairing. The values come from pwkit.environments.casa.util:pol_names

_fpol_to_pol = np.ndarray(128, dtype=np.int8)
_fpol_to_pol.fill(0xFF)
_fpol_to_pol[0x00] = 9
_fpol_to_pol[0x01] = 10
_fpol_to_pol[0x10] = 11
_fpol_to_pol[0x11] = 12
_fpol_to_pol[0x22] = 5
_fpol_to_pol[0x23] = 5
_fpol_to_pol[0x32] = 7
_fpol_to_pol[0x33] = 8
_fpol_to_pol[0x44] = 1
_fpol_to_pol[0x55] = 2
_fpol_to_pol[0x66] = 3
_fpol_to_pol[0x77] = 4

# A "antpol" (AP) is an integer identifying an antenna/fpol pair. It
# can be decoded without any external information. We used zero-based
# integer antenna numbers so that
#
#   AP = M << 3 + FP


def ap_format(ap, getname=str):
    return "%s%c" % (getname(ap >> 3), fpol_names[ap & 0x7])


ap_ant = lambda ap: ap >> 3
ap_fpol = lambda ap: ap & 0x7
antpol_to_ap = lambda antnum, fpol: (antnum << 3) + fpol

# A "basepol" is 2-tuple of antpols.


def bp_format(bp, getname=str):
    ap1, ap2 = bp
    if ap1 < 0:
        raise ValueError("first antpol %d is negative" % ap1)
    if ap2 < 0:
        raise ValueError("second antpol %d is negative" % ap2)

    return "%s%c-%s%c" % (
        getname(ap1 >> 3),
        fpol_names[ap1 & 0x7],
        getname(ap2 >> 3),
        fpol_names[ap2 & 0x7],
    )


def bp_to_aap(bp):
    """Converts a basepol into a tuple of (ant1, ant2, pol)."""

    ap1, ap2 = bp
    if ap1 < 0:
        raise ValueError("first antpol %d is negative" % ap1)
    if ap2 < 0:
        raise ValueError("second antpol %d is negative" % ap2)

    pol = _fpol_to_pol[((ap1 & 0x7) << 4) + (ap2 & 0x7)]
    if pol == 0xFF:
        raise ValueError(
            "no CASA polarization code for pairing "
            "%c-%c" % (fpol_names[ap1 & 0x7], fpol_names[ap2 & 0x7])
        )

    return ap1 >> 3, ap2 >> 3, pol


def aap_to_bp(ant1, ant2, pol):
    """Create a basepol from antenna numbers and a CASA polarization code."""

    if ant1 < 0:
        raise ValueError("first antenna is below 0: %s" % ant1)
    if ant2 < ant1:
        raise ValueError("second antenna is below first: %s" % ant2)
    if pol < 1 or pol > 12:
        raise ValueError("illegal polarization code %s" % pol)

    fps = _pol_to_fpol[pol]
    ap1 = (ant1 << 3) + ((fps >> 4) & 0x07)
    ap2 = (ant2 << 3) + (fps & 0x07)
    return ap1, ap2


# End copying/emulating mirtask.util
# ########################################################################


class StatsCollector(object):
    def __init__(self, chunk0size=64):
        self.chunk0size = chunk0size
        self._keymap = collections.defaultdict(lambda: len(self._keymap))
        self._m0 = None  # 0th moment
        self._m1 = None  #
        self._m2 = None  # 2nd moment

    def accum(self, key, value, weight=1):
        index = self._keymap[key]

        if self._m0 is None:
            self._m0 = np.zeros((self.chunk0size,), dtype=np.result_type(weight))
            self._m1 = np.zeros((self.chunk0size,), dtype=np.result_type(value, weight))
            self._m2 = np.zeros_like(self._m1)
        elif index >= self._m0.size:
            self._m0 = np.concatenate((self._m0, np.zeros_like(self._m0)))
            self._m1 = np.concatenate((self._m1, np.zeros_like(self._m1)))
            self._m2 = np.concatenate((self._m2, np.zeros_like(self._m2)))

        self._m0[index] += weight
        q = weight * value
        self._m1[index] += q
        q *= value
        self._m2[index] += q
        return self

    def finish(self, keyset, mask=True):
        """Returns (weights, means, variances), where:

        weights
          ndarray of number of samples per key
        means
          computed mean value for each key
        variances
          computed variance for each key

        """
        n_us = len(self._keymap)
        # By definition (for now), wt >= 1 everywhere, so we don't need to
        # worry about div-by-zero.
        wt_us = self._m0[:n_us]
        mean_us = self._m1[:n_us] / wt_us
        var_us = self._m2[:n_us] / wt_us - mean_us**2

        n_them = len(keyset)
        wt = np.zeros(n_them, dtype=self._m0.dtype)
        mean = np.empty(n_them, dtype=self._m1.dtype)
        mean.fill(np.nan)
        var = np.empty_like(mean)
        var.fill(np.nan)

        us_idx = []
        them_idx = []

        for them_i, key in enumerate(keyset):
            us_i = self._keymap[key]
            if us_i < n_us:
                them_idx.append(them_i)
                us_idx.append(us_i)
            # otherwise, we must not have seen that key

        wt[them_idx] = wt_us[us_idx]
        mean[them_idx] = mean_us[us_idx]
        var[them_idx] = var_us[us_idx]

        if mask:
            m = ~np.isfinite(mean)
            mean = np.ma.MaskedArray(mean, m)
            var = np.ma.MaskedArray(var, m)

        self._m0 = self._m1 = self._m2 = None
        self._keymap.clear()

        return wt, mean, var


class StatsCollector2D(object):
    def __init__(self, chunk0size=64):
        self.chunk0size = chunk0size
        self._key1map = collections.defaultdict(lambda: len(self._key1map))
        self._key2map = collections.defaultdict(lambda: len(self._key2map))
        self._m0 = None
        self._m1 = None
        self._m2 = None

    def accum(self, key1, key2, value, weight=1):
        index1 = self._key1map[key1]
        index2 = self._key2map[key2]

        if self._m0 is None:
            self._m0 = np.zeros(
                (self.chunk0size, self.chunk0size), dtype=np.result_type(weight)
            )
            self._m1 = np.zeros(
                (self.chunk0size, self.chunk0size), dtype=np.result_type(value, weight)
            )
            self._m2 = np.zeros_like(self._m1)

        if index1 >= self._m0.shape[0]:
            self._m0 = np.concatenate((self._m0, np.zeros_like(self._m0)), axis=0)
            self._m1 = np.concatenate((self._m1, np.zeros_like(self._m1)), axis=0)
            self._m2 = np.concatenate((self._m2, np.zeros_like(self._m2)), axis=0)

        if index2 >= self._m0.shape[1]:
            self._m0 = np.concatenate((self._m0, np.zeros_like(self._m0)), axis=1)
            self._m1 = np.concatenate((self._m1, np.zeros_like(self._m1)), axis=1)
            self._m2 = np.concatenate((self._m2, np.zeros_like(self._m2)), axis=1)

        self._m0[index1, index2] += weight
        q = weight * value
        self._m1[index1, index2] += q
        q *= value
        self._m2[index1, index2] += q
        return self

    def finish(self, key1set, key2set, mask=True):
        """Returns (weights, means, variances), where:

        weights
          ndarray of number of samples per key; shape (n1, n2), where n1 is the
          size of key1set and n2 is the size of key2set.
        means
          computed mean value for each key
        variances
          computed variance for each key

        """
        n1_us = len(self._key1map)
        n2_us = len(self._key2map)
        wt_us = self._m0[:n1_us, :n2_us]
        badwt = (wt_us == 0) | ~np.isfinite(wt_us)
        wt_us[badwt] = 1
        mean_us = self._m1[:n1_us, :n2_us] / wt_us
        var_us = self._m2[:n1_us, :n2_us] / wt_us - mean_us**2
        wt_us[badwt] = 0
        mean_us[badwt] = np.nan
        var_us[badwt] = np.nan

        n1_them = len(key1set)
        n2_them = len(key2set)
        wt = np.zeros((n1_them, n2_them), dtype=self._m0.dtype)
        mean = np.empty((n1_them, n2_them), dtype=self._m1.dtype)
        mean.fill(np.nan)
        var = np.empty_like(mean)
        var.fill(np.nan)

        # You can't fancy-index on two axes simultaneously, so we do a manual
        # loop on the first axis.

        us_idx2 = []
        them_idx2 = []

        for them_i2, key2 in enumerate(key2set):
            us_i2 = self._key2map[key2]
            if us_i2 < n2_us:
                them_idx2.append(them_i2)
                us_idx2.append(us_i2)
            # otherwise, we must not have seen that key

        for them_i1, key1 in enumerate(key1set):
            us_i1 = self._key1map[key1]
            if us_i1 >= n1_us:
                continue  # don't have this key

            wt[them_i1, them_idx2] = wt_us[us_i1, us_idx2]
            mean[them_i1, them_idx2] = mean_us[us_i1, us_idx2]
            var[them_i1, them_idx2] = var_us[us_i1, us_idx2]

        if mask:
            m = ~np.isfinite(mean)
            mean = np.ma.MaskedArray(mean, m)
            var = np.ma.MaskedArray(var, m)

        self._m0 = self._m1 = self._m2 = None
        self._key1map.clear()
        self._key2map.clear()

        return wt, mean, var


class StatsCollectorND(object):
    """This is vaguely like StatsCollector2D, but rather than having two discrete
    keyed axes, we are passed one key and an N-dimensional vector of values.

    """

    def __init__(self, chunk0size=64):
        self.chunk0size = chunk0size
        self._keymap = collections.defaultdict(lambda: len(self._keymap))
        self._m0 = None
        self._m1 = None
        self._m2 = None

    def accum(self, key, values, weights=1):
        index = self._keymap[key]
        values = np.asarray(values)
        weights = np.broadcast_to(weights, values.shape)

        if self._m0 is None:
            self._m0 = np.zeros((self.chunk0size,) + values.shape, dtype=weights.dtype)
            self._m1 = np.zeros(
                (self.chunk0size,) + values.shape, dtype=np.result_type(weights, values)
            )
            self._m2 = np.zeros_like(self._m1)
        elif index >= self._m0.size:
            self._m0 = np.concatenate((self._m0, np.zeros_like(self._m0)))
            self._m1 = np.concatenate((self._m1, np.zeros_like(self._m1)))
            self._m2 = np.concatenate((self._m2, np.zeros_like(self._m2)))

        if values.shape != self._m0.shape[1:]:
            raise ValueError(
                "inconsistent `values` shapes: had %r, got %r"
                % (self._m0.shape[1:], values.shape)
            )

        self._m0[index] += weights
        q = weights * values
        self._m1[index] += q
        q *= values
        self._m2[index] += q
        return self

    def finish(self, keyset, mask=True):
        """Returns (weights, means, variances), where:

        weights
          total weights per key
        means
          computed mean value for each key
        variances
          computed variance for each key

        The arrays all have a shape of `(nkeys,)+shape(values)`.

        """
        n_us = len(self._keymap)
        wt_us = self._m0[:n_us]
        badwt = (wt_us == 0) | ~np.isfinite(wt_us)
        wt_us[badwt] = 1.0
        mean_us = self._m1[:n_us] / wt_us
        var_us = self._m2[:n_us] / wt_us - mean_us**2
        wt_us[badwt] = 0
        mean_us[badwt] = np.nan
        var_us[badwt] = np.nan

        n_them = len(keyset)
        wt = np.zeros((n_them,) + mean_us.shape[1:], dtype=self._m0.dtype)
        mean = np.empty((n_them,) + mean_us.shape[1:], dtype=self._m1.dtype)
        mean.fill(np.nan)
        var = np.empty_like(mean)
        var.fill(np.nan)

        us_idx = []
        them_idx = []

        for them_i, key in enumerate(keyset):
            us_i = self._keymap[key]
            if us_i < n_us:
                them_idx.append(them_i)
                us_idx.append(us_i)
            # otherwise, we must not have seen that key

        wt[them_idx] = wt_us[us_idx]
        mean[them_idx] = mean_us[us_idx]
        var[them_idx] = var_us[us_idx]

        if mask:
            m = ~np.isfinite(mean)
            mean = np.ma.MaskedArray(mean, m)
            var = np.ma.MaskedArray(var, m)

        self._m0 = self._m1 = self._m2 = None
        self._keymap.clear()

        return wt, mean, var


def postproc(stats_result):
    """Simple helper to postprocess angular outputs from StatsCollectors in the
    way we want.

    """
    n, mean, scat = stats_result
    mean *= 180 / np.pi  # rad => deg
    scat /= n  # variance-of-samples => variance-of-mean
    scat **= 0.5  # variance => stddev
    scat *= 180 / np.pi  # rad => deg
    return mean, scat


def postproc_mask(stats_result):
    """Simple helper to postprocess angular outputs from StatsCollectors in the
    way we want.

    """
    n, mean, scat = stats_result

    ok = np.isfinite(mean)
    n = n[ok]
    mean = mean[ok]
    scat = scat[ok]

    mean *= 180 / np.pi  # rad => deg
    scat /= n  # variance-of-samples => variance-of-mean
    scat **= 0.5  # variance => stddev
    scat *= 180 / np.pi  # rad => deg
    return ok, mean, scat


def grid_bp_data(bps, items, mask=True):
    """Given a bunch of scalars associated with intensity-type basepols, place
    them onto a grid. There should be only two polarizations represented (e.g.
    RR, LL); baselines for one of them will be put into the upper triangle of
    the grid, while baselines for the other will be put into the lower triangle.

    `bps` is an iterable that yields a superset of all bps to be gridded.

    `items` is an iterable that yields tuples of (bp, value). (`bps` is
    therefore a bit redundant with it, but this structure makes it so that we
    don't need to iterate of `items` twice, which can be convenient.)

    Returns: (pol1, pol2, ants, data), where

    pol1
      The polarization gridded into the upper triangle
    pol2
      The polarization gridded into the lower triangle
    ants
      An array of the antenna numbers seen in `bps`
    data
      An n-by-n grid of the values, where n is the size of `ants`. Unsampled
      basepols are filled with NaNs, or masked if `mask` is True.

    The basepol (ant1, ant2, pol1) is gridded into `data[i1,i2]`, where `i1`
    is the index of `ant1` in `ants`, etc. The basepol (ant1, ant2, pol2) is
    gridded into `data[i2,i1]`.

    """
    seen_ants = set()
    seen_pols = set()

    for ant1, ant2, pol in (bp_to_aap(bp) for bp in bps):
        seen_ants.add(ant1)
        seen_ants.add(ant2)
        seen_pols.add(pol)

    if len(seen_pols) != 2:
        raise Exception("can only work with 2 polarizations")
    pol1, pol2 = sorted(seen_pols)

    seen_ants = np.array(sorted(seen_ants))
    ant_map = dict((a, i) for (i, a) in enumerate(seen_ants))

    data = None
    n = len(seen_ants)

    for bp, value in items:
        if data is None:
            data = np.empty((n, n), dtype=np.result_type(value))
            data.fill(np.nan)

        ant1, ant2, pol = bp_to_aap(bp)
        i1 = ant_map[ant1]
        i2 = ant_map[ant2]

        if pol == pol1:
            data[i1, i2] = value
        else:
            data[i2, i1] = value

    if mask:
        data = np.ma.MaskedArray(data, ~np.isfinite(data))

    return pol1, pol2, seen_ants, data


class ClosureCalculator(object):
    def process(self, cfg):
        # Initialize whole-run buffers.
        raise Exception("BROKEN: NEEDS UPDATE TO SELECT ON DATA_DESC_ID")

        self.all_aps = set()
        self.all_bps = set()
        self.all_times = set()
        self.global_stats_by_time = StatsCollector()
        self.ap_stats_by_ddid = collections.defaultdict(StatsCollector)
        self.bp_stats_by_ddid = collections.defaultdict(StatsCollector)
        self.ap_spec_stats_by_ddid = collections.defaultdict(StatsCollectorND)
        self.ap_time_stats_by_ddid = collections.defaultdict(StatsCollector2D)

        self._process_ms(cfg)
        return self

    def _process_ms(self, cfg):
        tb = util.tools.table()
        ms = util.tools.ms()
        me = util.tools.measures()

        # Prep.

        ms.open(b(cfg.vis))
        ms.msselect(
            b(
                dict(
                    (n, cfg.get(n))
                    for n in util.msselect_keys
                    if cfg.get(n) is not None
                )
            )
        )

        rangeinfo = ms.range(b"data_desc_id field_id".split())
        ddids = rangeinfo["data_desc_id"]
        fields = rangeinfo["field_id"]

        tb.open(b(cfg.vis + "/DATA_DESCRIPTION"), nomodify=True)
        ddid_to_polid = tb.getcol(b"POLARIZATION_ID")
        ddid_to_spwid = tb.getcol(b"SPECTRAL_WINDOW_ID")
        tb.close()

        tb.open(b(cfg.vis + "/POLARIZATION"), nomodify=True)
        polid_to_polns = {}
        for i in range(tb.nrows()):
            polid_to_polns[i] = tb.getcell(b"CORR_TYPE", i)
        tb.close()

        tb.open(b(cfg.vis + "/ANTENNA"), nomodify=True)
        self.ant_names = tb.getcol(b"NAME")
        self.ant_stations = tb.getcol(b"STATION")
        tb.close()

        # Read stuff in. We can't expect that weight values have their
        # absolute scale set correctly, but we can still use them to set the
        # relative weighting of the data points.
        #
        #   datacol is (ncorr, nchan, nchunk)
        #   flag is (ncorr, nchan, nchunk); zero means OK data
        #   weight is (ncorr, nchunk) [XXX: WEIGHT_SPECTRUM?]
        #   uvw is (3, nchunk)
        #   time is (nchunk)
        #
        # Iteration order, from slowest varying to fastest varying:
        #
        #   spw, time, baseline, poln, frequency
        #
        # We are encouraged to use the new iteration interface
        # (ms.niterinit(), etc), but as of 4.6.0 it is fundamentally broken
        # (asking for ANTENNA2 gets you ANTENNA1) so never mind.

        colnames = [cfg.datacol] + "flag weight time antenna1 antenna2".split()
        colnames = b(colnames)

        for ddid in ddids:
            raise Exception("UPDATE TO SELECT ON DATA_DESC_ID BECAUSE CASA IS BROKEN")

            ms.selectinit(ddid)
            ms.iterinit(maxrows=4096)
            ms.iterorigin()

            self.cur_ddid = ddid
            self.cur_time = -1.0
            self._start_timeslot()
            all_polns = polid_to_polns[ddid_to_polid[ddid]]
            polinfo = [(i, p) for i, p in enumerate(all_polns) if pol_is_intensity[p]]

            while True:
                cols = ms.getdata(items=colnames)

                for i in range(cols["time"].size):  # all records
                    time = cols["time"][i]
                    if time != self.cur_time:
                        self._finish_timeslot()
                        self.cur_time = time
                        self._start_timeslot()

                    antenna1 = cols["antenna1"][i]
                    antenna2 = cols["antenna2"][i]
                    if antenna1 == antenna2:
                        continue  # no autocorrelations

                    for j, poln in polinfo:  # all polns
                        flags = cols["flag"][j, :, i]
                        if flags.all():
                            continue  # all flagged

                        data = cols[cfg.datacol][j, :, i]
                        # data and flags are now both shape (nchan,)
                        np.logical_not(flags, flags)
                        # flags=1 now indicates good data

                        self.all_times.add(time)
                        bp = aap_to_bp(antenna1, antenna2, poln)
                        self.all_bps.add(bp)
                        self.data_by_bp[bp] = (data, flags)  # + weight spectrum?

                        # Here we exploit the fact that we're only considering
                        # intensity-type polarizations, so bp[0] and bp[1]
                        # always have the same fpol. Also that ap_by_fpol is a
                        # defaultdict.
                        for ap in bp:
                            self.all_aps.add(ap)
                            self.ap_by_fpol[ap & 0x7].add(ap)

                if not ms.iternext():
                    self._finish_timeslot()
                    break

        ms.close()
        return self

    def _getname(self, antidx):
        return "%s@%s:" % (self.ant_names[antidx], self.ant_stations[antidx])

    def _start_timeslot(self):
        self.data_by_bp = {}
        self.ap_by_fpol = collections.defaultdict(set)

    def _finish_timeslot(self):
        """We have loaded in all of the visibilities in one timeslot. We can now
        compute the phase closure triples.

        XXX: we should only process independent triples. Are we???

        """
        for fpol, aps in self.ap_by_fpol.items():
            aps = sorted(aps)
            nap = len(aps)

            for i1, ap1 in enumerate(aps):
                for i2 in range(i1, nap):
                    ap2 = aps[i2]
                    bp1 = (ap1, ap2)
                    info = self.data_by_bp.get(bp1)
                    if info is None:
                        continue

                    data1, flags1 = info

                    for i3 in range(i2, nap):
                        ap3 = aps[i3]
                        bp2 = (ap2, ap3)
                        info = self.data_by_bp.get(bp2)
                        if info is None:
                            continue

                        data2, flags2 = info
                        bp3 = (ap1, aps[i3])
                        info = self.data_by_bp.get(bp3)
                        if info is None:
                            continue

                        data3, flags3 = info

                        # try to minimize allocations:
                        tflags = flags1 & flags2
                        np.logical_and(tflags, flags3, tflags)
                        if not tflags.any():
                            continue

                        triple = data3.conj()
                        np.multiply(triple, data1, triple)
                        np.multiply(triple, data2, triple)
                        self._process_sample(ap1, ap2, ap3, triple, tflags)

        # Reset for next timeslot

        self.cur_time = -1.0
        self.bp_by_ap = None
        self.ap_by_fpol = None

    def _process_sample(self, ap1, ap2, ap3, triple, tflags):
        """We have computed one independent phase closure triple in one timeslot."""
        # Frequency-resolved:
        np.divide(triple, np.abs(triple), triple)
        phase = np.angle(triple)

        self.ap_spec_stats_by_ddid[self.cur_ddid].accum(ap1, phase, tflags + 0.0)
        self.ap_spec_stats_by_ddid[self.cur_ddid].accum(ap2, phase, tflags + 0.0)
        self.ap_spec_stats_by_ddid[self.cur_ddid].accum(ap3, phase, tflags + 0.0)

        # Frequency-averaged:
        triple = np.dot(triple, tflags) / tflags.sum()
        phase = np.angle(triple)

        self.global_stats_by_time.accum(self.cur_time, phase)

        self.ap_stats_by_ddid[self.cur_ddid].accum(ap1, phase)
        self.ap_stats_by_ddid[self.cur_ddid].accum(ap2, phase)
        self.ap_stats_by_ddid[self.cur_ddid].accum(ap3, phase)
        self.bp_stats_by_ddid[self.cur_ddid].accum((ap1, ap2), phase)
        self.bp_stats_by_ddid[self.cur_ddid].accum((ap1, ap3), phase)
        self.bp_stats_by_ddid[self.cur_ddid].accum((ap2, ap3), phase)

        self.ap_time_stats_by_ddid[self.cur_ddid].accum(self.cur_time, ap1, phase)
        self.ap_time_stats_by_ddid[self.cur_ddid].accum(self.cur_time, ap2, phase)
        self.ap_time_stats_by_ddid[self.cur_ddid].accum(self.cur_time, ap3, phase)

    def report(self, cfg):
        import omega as om
        import omega.gtk3
        from pwkit import ndshow_gtk3

        self.all_aps = np.sort(list(self.all_aps))
        self.all_bps = sorted(self.all_bps)
        self.all_times = np.sort(list(self.all_times))

        # Antpols by DDID, time:
        data = []
        descs = []

        for ddid, stats in self.ap_time_stats_by_ddid.items():
            mean, scat = postproc(stats.finish(self.all_times, self.all_aps))
            data.append(mean / scat)
            descs.append("DDID %d" % ddid)

        print("Viewing X axis: antpol; Y axis: time; iteration: DDID (~= spwid) ...")
        ndshow_gtk3.cycle(data, descs, run_main=True)

        # Antpols by DDID, freq:
        data = []
        descs = []

        for ddid, stats in self.ap_spec_stats_by_ddid.items():
            mean, scat = postproc(stats.finish(self.all_aps))
            data.append(mean / scat)
            descs.append("DDID %d" % ddid)

        print("Viewing X axis: frequency; Y axis: antpol; iteration: DDID ...")
        ndshow_gtk3.cycle(data, descs, run_main=True)

        # Antpols by DDID
        p = om.RectPlot()

        for ddid, stats in self.ap_stats_by_ddid.items():
            ok, mean, scat = postproc_mask(stats.finish(self.all_aps))
            p.addXYErr(np.arange(len(self.all_aps))[ok], mean, scat, "DDID %d" % ddid)

        p.setBounds(-0.5, len(self.all_aps) - 0.5)
        p.setLabels("Antpol number", "Mean closure phase (rad)")
        p.addHLine(0, keyText=None, zheight=-1)
        print("Viewing everything grouped by antpol ...")
        p.show()

        # Basepols by DDID
        data = []
        descs = []
        tostatuses = []

        def bpgrid_status(pol1, pol2, ants, yx):
            i, j = [int(_) for _ in np.floor(yx + 0.5)]
            if i < 0 or j < 0 or i >= ants.size or j >= ants.size:
                return ""

            ni = self._getname(ants[i])
            nj = self._getname(ants[j])

            if i <= j:
                return "%s-%s %s" % (ni, nj, util.pol_names[pol1])
            return "%s-%s %s" % (nj, ni, util.pol_names[pol2])

        for ddid, stats in self.bp_stats_by_ddid.items():
            mean, scat = postproc(stats.finish(self.all_bps))
            nmean = mean / scat
            pol1, pol2, ants, grid = grid_bp_data(
                self.all_bps, zip(self.all_bps, nmean)
            )
            data.append(grid)
            descs.append("DDID %d" % ddid)
            tostatuses.append(lambda yx: bpgrid_status(pol1, pol2, ants, yx))

        print("Viewing X axis: antpol1; Y axis: antpol2; iteration: DDID ...")
        ndshow_gtk3.cycle(data, descs, tostatuses=tostatuses, run_main=True)

        # Everything by time
        ok, mean, scat = postproc_mask(self.global_stats_by_time.finish(self.all_times))
        stimes = self.all_times[ok] / 86400
        st0 = int(np.floor(stimes.min()))
        stimes -= st0
        p = om.quickXYErr(stimes, mean, scat)
        p.addHLine(0, keyText=None, zheight=-1)
        p.setLabels("MJD - %d" % st0, "Mean closure phase (rad)")
        print("Viewing everything grouped by time ...")
        p.show()


def closures_cli(argv):
    check_usage(closures_doc, argv, usageifnoargs=True)
    cfg = Config().parse(argv[1:])
    util.logger(cfg.loglevel)
    ClosureCalculator().process(cfg).report(cfg)
