# -*- mode: python; coding: utf-8 -*-
# Copyright 2013, 2016-2018 Peter Williams <peter@newton.cx> and collaborators
# Licensed under the MIT License.

# NB. This is super-redundant with both dftphotom and dftspect; things are
# getting a little silly here. But I think it's faster to copy/paste/hack than
# it is to merge everything into one uberprogram.

"""This module provides code to extract dynamic spectra from CASA Measurement
Sets. CASA doesn't have a task that does this.

"""
from __future__ import absolute_import, division, print_function

__all__ = "Config Loader dftdynspec dftdynspec_cli".split()

import io, os.path
import numpy as np

# Note: zany spacing so that Sphinx can parse the file correctly.
from ...astutil import *
from ...cli import check_usage, die
from ...io import get_stdout_bytes
from ...kwargv import ParseKeywords, Custom
from . import util
from .util import sanitize_unicode as b

dftdynspec_doc = """
casatask dftdynspec vis=<MS> [keywords...]

Extract a dynamic spectrum from the visibilities in a measurement set. See
below the keyword docs for some important caveats.

vis=
  Path of the MeasurementSet dataset to read. Required.

out=
  Path to which data will be written. If unspecified, data are written to
  standard output.

rephase=RA,DEC
  Phase the data to extract fluxes in the specified direction. If unspecified,
  the data are not rephased, i.e. the flux at the phase center is extracted.
  RA and DEC should be in sexigesimal with fields separated by colons.

array=, baseline=, field=, observation=, polarization=, scan=,
scanintent=, spw=, taql=, time=, uvdist=
  MeasurementSet selectors used to filter the input data. Default polarization
  is 'RR,LL'. All polarizations are averaged together, so mixing parallel- and
  cross-hand pols is almost never what you want to do.

  NOTE: only whole spectral windows can be selected. If you attempt to select
  channels within a window, a warning will be printed and that component of
  the selection will be ignored.

datacol=
  Name of the column to use for visibility data. Defaults to 'data'. You might
  want it to be 'corrected_data'.

believeweights=[t|f]
  Defaults to false, which means that we assume that the 'weight' column in
  the dataset is NOT scaled such that the variance in the visibility samples
  is equal to 1/weight. In this case uncertainties are assessed from the
  scatter of all the visibilities in each timeslot. If true, we trust that
  variance=1/weight and propagate this in the standard way.

IMPORTANT: the fundamental assumption of this task is that the only
signal in the visibilities is from a point source at the phasing
center. We also assume that all sampled polarizations get equal
contributions from the source (though you can resample the Stokes
parameters on the fly, so this is not quite the same thing as
requiring the source be unpolarized).

The output data are stored as a file containing several serialized Numpy
arrays. The Python class `pwkit.environments.casa.dftdynspec.Loader` can be
used to load the data into a Python program.

"""


class Config(ParseKeywords):
    vis = Custom(str, required=True)
    datacol = "data"
    believeweights = False

    @Custom(str, uiname="out")
    def outstream(val):
        if val is None:
            return get_stdout_bytes()
        try:
            return open(val, "wb")
        except Exception as e:
            die('cannot open path "%s" for writing', val)

    @Custom([str, str], default=None)
    def rephase(val):
        if val is None:
            return None

        try:
            ra = parsehours(val[0])
            dec = parsedeglat(val[1])
        except Exception as e:
            die('cannot parse "rephase" values as RA/dec: %s', e)
        return ra, dec

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


def dftdynspec(cfg):
    tb = util.tools.table()
    ms = util.tools.ms()
    me = util.tools.measures()

    # Read stuff in. Even if the weight values don't have their
    # absolute scale set correctly, we can still use them to set the
    # relative weighting of the data points.
    #
    # datacol is (ncorr, nchan, nchunk)
    # flag is (ncorr, nchan, nchunk)
    # weight is (ncorr, nchunk)
    # uvw is (3, nchunk)
    # time is (nchunk)
    # axis_info.corr_axis is (ncorr)
    # axis_info.freq_axis.chan_freq is (nchan, 1) [for now?]
    #
    # Note that we apply msselect() again when reading the data because
    # selectinit() is broken, but the invocation here is good because it
    # affects the results from ms.range() and friends.

    if ":" in (cfg.spw or ""):
        warn(
            "it looks like you are attempting to select channels within one or more spws"
        )
        warn("this is NOT IMPLEMENTED; I will process the whole spw instead")

    ms.open(b(cfg.vis))
    totrows = ms.nrow()
    ms_sels = dict(
        (n, cfg.get(n)) for n in util.msselect_keys if cfg.get(n) is not None
    )
    ms.msselect(b(ms_sels))

    rangeinfo = ms.range(b"data_desc_id field_id".split())
    ddids = rangeinfo["data_desc_id"]
    fields = rangeinfo["field_id"]
    colnames = [cfg.datacol] + "flag weight time axis_info".split()
    rephase = cfg.rephase is not None

    if fields.size != 1:
        # I feel comfortable making this a fatal error, even if we're
        # not rephasing.
        die("selected data should contain precisely one field; got %d", fields.size)

    tb.open(b(os.path.join(cfg.vis, "DATA_DESCRIPTION")))
    ddspws = tb.getcol(b"SPECTRAL_WINDOW_ID")
    tb.close()

    # Get frequencies and precompute merged, sorted frequency array
    # FIXME: below we get 'freqs' on the fly; should honor that.
    # But then mapping and data storage get super inefficient.

    tb.open(b(os.path.join(cfg.vis, "SPECTRAL_WINDOW")))
    nspw = tb.nrows()
    spwfreqs = []
    for i in range(nspw):
        spwfreqs.append(tb.getcell(b"CHAN_FREQ", i) * 1e-9)  # -> GHz
    tb.close()

    allfreqs = set()
    for freqs in spwfreqs:
        allfreqs.update(freqs)
    allfreqs = np.asarray(sorted(allfreqs))
    nfreq = allfreqs.size

    freqmaps = []
    for i in range(nspw):
        freqmaps.append(np.searchsorted(allfreqs, spwfreqs[i]))

    if rephase:
        fieldid = fields[0]
        tb.open(b(os.path.join(cfg.vis, "FIELD")))
        phdirinfo = tb.getcell(b"PHASE_DIR", fieldid)
        tb.close()

        if phdirinfo.shape[1] != 1:
            die(
                "trying to rephase but target field (#%d) has a "
                "time-variable phase center, which I can't handle",
                fieldid,
            )
        ra0, dec0 = phdirinfo[:, 0]  # in radians.

        # based on intflib/pwflux.py, which was copied from
        # hex/hex-lib-calcgainerr:

        dra = cfg.rephase[0] - ra0
        dec = cfg.rephase[1]
        l = np.sin(dra) * np.cos(dec)
        m = np.sin(dec) * np.cos(dec0) - np.cos(dra) * np.cos(dec) * np.sin(dec0)
        n = np.sin(dec) * np.sin(dec0) + np.cos(dra) * np.cos(dec) * np.cos(dec0)
        n -= 1  # makes the work below easier
        lmn = np.asarray([l, m, n])
        colnames.append("uvw")

    tbins = {}
    colnames = b(colnames)

    for ddindex, ddid in enumerate(ddids):
        # Starting in CASA 4.6, selectinit(ddid) stopped actually filtering
        # your data to match the specified DDID! What garbage. Work around
        # with our own filtering.
        ms_sels["taql"] = "DATA_DESC_ID == %d" % ddid
        ms.msselect(b(ms_sels))

        ms.selectinit(ddid)
        if cfg.polarization is not None:
            ms.selectpolarization(b(cfg.polarization.split(",")))
        ms.iterinit(maxrows=4096)
        ms.iterorigin()

        spwid = ddspws[ddid]

        while True:
            cols = ms.getdata(items=colnames)

            if rephase:
                # With appropriate spw/DDID selection, `freqs` has shape
                # (nchan, 1). Convert to m^-1 so we can multiply against UVW
                # directly.
                freqs = cols["axis_info"]["freq_axis"]["chan_freq"]
                assert freqs.shape[1] == 1, "internal inconsistency, chan_freq??"
                freqs = freqs[:, 0] * util.INVERSE_C_MS

            for i in range(cols["time"].size):  # all records
                time = cols["time"][i]
                # get out of UTC as fast as we can! For some reason
                # giving 'unit=s' below doesn't do what one might hope it would.
                # CASA can convert to a variety of timescales; TAI is probably
                # the safest conversion in terms of being helpful while remaining
                # close to the fundamental data, but TT is possible and should
                # be perfectly precise for standard applications.
                mq = me.epoch(b"utc", b({"value": time / 86400.0, "unit": "d"}))
                mjdtt = me.measure(b(mq), b"tt")["m0"]["value"]

                tdata = tbins.get(mjdtt)
                if tdata is None:
                    tdata = tbins[mjdtt] = np.zeros((nfreq, 7))

                if rephase:
                    uvw = cols["uvw"][:, i]
                    ph = np.exp((0 - 2j) * np.pi * np.dot(lmn, uvw) * freqs)

                for j in range(cols["flag"].shape[0]):  # all polns
                    # We just average together all polarizations right now!
                    # (Not actively, but passively by just iterating over them.)
                    data = cols[cfg.datacol][j, :, i]
                    flags = cols["flag"][j, :, i]

                    # XXXXX casacore is currently broken and returns the raw
                    # weights from the dataset rather than applying the
                    # polarization selection. Fortunately all of our weights
                    # are the same, and you can never fetch more pol types
                    # than the dataset has, so this bit works despite the bug.

                    w = np.where(~flags)[0]
                    if not w.size:
                        continue  # all flagged

                    if rephase:
                        data *= ph

                    m = freqmaps[spwid][w]
                    d = data[w]
                    wt = cols["weight"][j, i]

                    tdata[m, 0] += wt * d.real
                    tdata[m, 1] += wt * d.imag
                    tdata[m, 2] += wt * d.real**2
                    tdata[m, 3] += wt * d.imag**2
                    tdata[m, 4] += wt
                    tdata[m, 5] += wt**2
                    tdata[m, 6] += 1

            if not ms.iternext():
                break

        ms.reset()  # reset selection filter so we can get next DDID

    ms.close()

    # Could gain some efficiency by using a better data structure than a dict().

    smjd = np.asarray(sorted(tbins.keys()))
    data = np.zeros((5, smjd.size, nfreq))

    for tid in range(smjd.size):
        mjd = smjd[tid]

        wr, wi, wr2, wi2, wt, wt2, n = tbins[mjd].T
        w = np.where(n > 0)[0]
        if w.size < 3:  # not enough data for meaningful statistics
            continue

        r = wr[w] / wt[w]
        i = wi[w] / wt[w]

        if cfg.believeweights:
            ru = wt[w] ** -0.5
            iu = wt[w] ** -0.5
        else:
            r2 = wr2[w] / wt[w]
            i2 = wi2[w] / wt[w]
            rv = r2 - r**2  # variance among real/imag msmts
            iv = i2 - i**2
            ru = np.sqrt(rv * wt2[w]) / wt[w]  # uncert in mean real/img values
            iu = np.sqrt(iv * wt2[w]) / wt[w]

        data[0, tid, w] = r
        data[1, tid, w] = ru
        data[2, tid, w] = i
        data[3, tid, w] = iu
        data[4, tid, w] = n[w]

    np.save(cfg.outstream, smjd)
    np.save(cfg.outstream, allfreqs)
    np.save(cfg.outstream, data)


def dftdynspec_cli(argv):
    check_usage(dftdynspec_doc, argv, usageifnoargs=True)
    cfg = Config().parse(argv[1:])
    util.logger(cfg.loglevel)
    dftdynspec(cfg)


class Loader(object):
    """Read in a dynamic-spectrum file produced by the `dftdynspec` task.

    **Constructor arguments**

    *path*
      The path of the file to read.

    """

    mjds = None
    "A 1D sorted array of the MJDs of the data samples."

    freqs = None
    "A 1D sorted array of the frequencies of the data samples, measured in GHz."

    reals = None
    """A 2D array of the real parts of the averaged visibilities. Shape is
    `(mjds.size, freqs.size)`.

    """
    u_reals = None
    """A 2D array of the estimated uncertainties on the real parts of the averaged
    visibilities. Shape is `(mjds.size, freqs.size)`.

    """
    imags = None
    """A 2D array of the imaginary parts of the averaged visibilities. Shape is
    `(mjds.size, freqs.size)`.

    """
    u_imags = None
    """A 2D array of the estimated uncertainties on the imaginary parts of the
    averaged visibilities. Shape is `(mjds.size, freqs.size)`.

    """
    counts = None
    """A 2D array recording the number of visibilities that went into each
    average. Shape is `(mjds.size, freqs.size)`.

    """

    def __init__(self, path):
        with io.open(path, "rb") as f:
            self.mjds = np.load(f)
            self.freqs = np.load(f)
            cube = np.load(f)
            self.reals, self.u_reals, self.imags, self.u_imags, self.counts = cube

    @property
    def n_mjds(self):
        "The size of the MJD axis of the data arrays; an integer."
        return self.mjds.size

    @property
    def n_freqs(self):
        "The size of the frequency axis of the data arrays; an integer."
        return self.freqs.size
