# -*- mode: python; coding: utf-8 -*-
# Copyright 2012-2015 Peter Williams <peter@newton.cx> and collaborators.
# Licensed under the MIT License.

"""pwkit.environments.casa.dftphotom - point-source photometry from visibilities

CASA doesn't yet have a task to do this.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = str ('Config dftphotom dftphotom_cli').split ()

import six, sys, os.path, numpy as np
from six.moves import range

from ... import binary_type, text_type
from ...astutil import *
from ...cli import check_usage, die
from ...kwargv import ParseKeywords, Custom
from . import util
from .util import sanitize_unicode as b

dftphotom_doc = \
"""
casatask dftphotom vis=MS [keywords...]

Extract photometry from the visibilities in a measurement set. See the full
documentation for some important caveats.

vis=
  Path of the MeasurementSet dataset to read. Required.

out=
  Path to which data will be written. If unspecified, data are written
  to standard output.

rephase=RA,DEC
  Phase the data to extract fluxes in the specified direction. If
  unspecified, the data are not rephased, i.e. the flux at the
  phase center is extracted. RA and DEC should be in sexigesimal
  with fields separated by colons.

array=, baseline=, field=, observation=, polarization=, scan=,
scanintent=, spw=, taql=, time=, uvdist=
  MeasurementSet selectors used to filter the input data.
  Default polarization is 'RR,LL'. All polarizations are averaged
  together, so mixing parallel- and cross-hand pols is almost
  never what you want to do.

datacol=
  Name of the column to use for visibility data. Defaults to 'data'.
  You might want it to be 'corrected_data'.

datascale=
  Multiply fluxes by this number before reporting them. Defaults to
  1e6, so that the output is in terms of microjanskys if the data are
  correctly flux-scaled. The textual output has two decimal places so
  adjusting this value can give better results if your characteristic
  fluxes are significantly different than this.

believeweights=[t|f]
  Defaults to false, which means that we assume that the 'weight'
  column in the dataset is NOT scaled such that the variance in the
  visibility samples is equal to 1/weight. In this case
  uncertainties are assessed from the scatter of all the visibilities
  in each timeslot. If true, we trust that variance=1/weight and
  propagate this in the standard way.

format=[humane(default)|pandas]
  The format of the output. The default is "humane" which is described
  below. The "pandas" format is slightly less human-friendly but can
  be read directly into a Pandas DataFrame with pandas.read_table ().

IMPORTANT: the fundamental assumption of this task is that the only
signal in the visibilities is from a point source at the phasing
center. We also assume that all sampled polarizations get equal
contributions from the source (though you can resample the Stokes
parameters on the fly, so this is not quite the same thing as
requiring the source be unpolarized).

In the "humane" format, the output columns are:

  MJD(TT) dt[min] re reErr im imErr mag magErr npts

with sorted MJDs, one record for each timestamp present in the dataset,
with all fluxes in ujy. dt is simply (MJD - MJD[0]) * 1440. The units
of re, im, mag, and their uncertainties are microjansky by default,
but see the datascale keyword, and there's no way to know if the
data have actually been flux-calibrated or not.
"""

class HumaneOutputFormat (object):
    def header (self, cfg):
        pass

    def row (self, cfg, mjd, dtmin, r_sc, ru_sc, i_sc, iu_sc, mag, umag, n):
        print ('%12.5f %6.2f %10.2f %10.2f %10.2f %10.2f %10.2f %10.2f %d' %
               (mjd, dtmin, r_sc, ru_sc, i_sc, iu_sc, mag, umag, n),
               file=cfg.outstream)


class PandasOutputFormat (object):
    def header (self, cfg):
        print ('mjd dtmin re ure im uim abs uabs nsamp'.replace (' ', '\t'),
               file=cfg.outstream)

    def row (self, cfg, *args):
        print ('\t'.join (str (x) for x in args), file=cfg.outstream)


class Config (ParseKeywords):
    vis = Custom (str, required=True)
    datacol = 'data'
    believeweights = False

    @Custom (str, uiname='out')
    def outstream (val):
        if val is None:
            return sys.stdout
        try:
            return open (val, 'w')
        except Exception as e:
            die ('cannot open path "%s" for writing', val)

    datascale = 1e6

    @Custom (str, default='humane')
    def format (val):
        if val is None or val == 'humane':
            return HumaneOutputFormat ()

        if val == 'pandas':
            return PandasOutputFormat ()

        die ('unrecognized output format %r', val)

    @Custom ([str, str], default=None)
    def rephase (val):
        if val is None:
            return None

        try:
            ra = parsehours (val[0])
            dec = parsedeglat (val[1])
        except Exception as e:
            die ('cannot parse "rephase" values as RA/dec: %s', e)
        return ra, dec

    # MeasurementSet filters
    array = str
    baseline = str
    field = str
    observation = str
    polarization = 'RR,LL'
    scan = str
    scanintent = str
    spw = str
    taql = str
    time = str
    uvdist = str

    loglevel = 'warn'


def dftphotom (cfg):
    tb = util.tools.table ()
    ms = util.tools.ms ()
    me = util.tools.measures ()

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

    ms.open (b(cfg.vis))
    totrows = ms.nrow ()
    sels = dict ((n, cfg.get (n)) for n in util.msselect_keys
                 if cfg.get (n) is not None)
    ms.msselect (b(sels))

    rangeinfo = ms.range (b'data_desc_id field_id'.split ())
    ddids = rangeinfo['data_desc_id']
    fields = rangeinfo['field_id']
    colnames = [cfg.datacol] + 'flag weight time axis_info'.split ()
    rephase = (cfg.rephase is not None)

    if fields.size != 1:
        # I feel comfortable making this a fatal error, even if we're
        # not rephasing.
        die ('selected data should contain precisely one field; got %d', fields.size)

    if rephase:
        fieldid = fields[0]
        tb.open (b(os.path.join (cfg.vis, 'FIELD')))
        phdirinfo = tb.getcell (b'PHASE_DIR', fieldid)
        tb.close ()

        if phdirinfo.shape[1] != 1:
            die ('trying to rephase but target field (#%d) has a '
                 'time-variable phase center, which I can\'t handle', fieldid)
        ra0, dec0 = phdirinfo[:,0] # in radians.

        # based on intflib/pwflux.py, which was copied from
        # hex/hex-lib-calcgainerr:

        dra = cfg.rephase[0] - ra0
        dec = cfg.rephase[1]
        l = np.sin (dra) * np.cos (dec)
        m = np.sin (dec) * np.cos (dec0) - np.cos (dra) * np.cos (dec) * np.sin (dec0)
        n = np.sin (dec) * np.sin (dec0) + np.cos (dra) * np.cos (dec) * np.cos (dec0)
        n -= 1 # makes the work below easier
        lmn = np.asarray ([l, m, n])
        colnames.append ('uvw')

        # Also need this although 99% of the time `ddid` and `spwid` are the same
        tb.open (b(os.path.join (cfg.vis, 'DATA_DESCRIPTION')))
        ddspws = np.asarray (tb.getcol (b'SPECTRAL_WINDOW_ID'))
        tb.close ()

    tbins = {}
    colnames = b(colnames)

    for ddid in ddids:
        ms.selectinit (ddid)
        if cfg.polarization is not None:
            ms.selectpolarization (b(cfg.polarization.split (',')))
        ms.iterinit (maxrows=4096)
        ms.iterorigin ()

        while True:
            cols = ms.getdata (items=colnames)

            if rephase:
                freqs = cols['axis_info']['freq_axis']['chan_freq']
                # In CASA <= 4.5, `freqs` has shape (nchan, 1) if you've done
                # a selectinit() on ddid. In 4.6, it has shape (nchan, nspw).
                # So:
                if freqs.shape[1] == 1:
                    freqs = freqs[:,0]
                else:
                    freqs = freqs[:,ddspws[ddid]]

                # convert to m^-1 so we can multiply against UVW directly:
                freqs *= util.INVERSE_C_MS

            for i in range (cols['time'].size): # all records
                time = cols['time'][i]
                # get out of UTC as fast as we can! For some reason
                # giving 'unit=s' below doesn't do what one might hope it would.
                # CASA can convert to a variety of timescales; TAI is probably
                # the safest conversion in terms of being helpful while remaining
                # close to the fundamental data, but TT is possible and should
                # be perfectly precise for standard applications.
                mq = me.epoch (b'utc', b({'value': time / 86400., 'unit': 'd'}))
                mjdtt = me.measure (mq, b'tt')['m0']['value']

                tdata = tbins.get (mjdtt, None)
                if tdata is None:
                    tdata = tbins[mjdtt] = [0., 0., 0., 0., 0]

                if rephase:
                    uvw = cols['uvw'][:,i]
                    ph = np.exp ((0-2j) * np.pi * np.dot (lmn, uvw) * freqs)

                for j in range (cols['flag'].shape[0]): # all polns
                    # We just average together all polarizations right now!
                    # (Not actively, but passively by just iterating over them.)
                    data = cols[cfg.datacol][j,:,i]
                    flags = cols['flag'][j,:,i]

                    # XXXXX casacore is currently (ca. 2012) broken and
                    # returns the raw weights from the dataset rather than
                    # applying the polarization selection. Fortunately all of
                    # our weights are the same, and you can never fetch more
                    # pol types than the dataset has, so this bit works
                    # despite the bug.

                    w = np.where (~flags)[0]
                    if not w.size:
                        continue # all flagged

                    if rephase:
                        data *= ph

                    d = data[w].mean ()
                    # account for flagged parts. 90% sure this is the
                    # right thing to do:
                    wt = cols['weight'][j,i] * w.size / data.size
                    wd = wt * d
                    # note a little bit of a hack here to encode real^2 and
                    # imag^2 separately:
                    wd2 = wt * (d.real**2 + (1j) * d.imag**2)

                    tdata[0] += wd
                    tdata[1] += wd2
                    tdata[2] += wt
                    tdata[3] += wt**2
                    tdata[4] += 1

            if not ms.iternext ():
                break

    ms.close ()

    # Could gain some efficiency by using a better data structure than a dict().
    smjd = sorted (six.iterkeys (tbins))
    cfg.format.header (cfg)

    for mjd in smjd:
        wd, wd2, wt, wt2, n = tbins[mjd]
        if n == 0:
            continue # could be all flagged

        dtmin = 1440 * (mjd - smjd[0])
        r_sc = wd.real / wt * cfg.datascale
        i_sc = wd.imag / wt * cfg.datascale
        r2_sc = wd2.real / wt * cfg.datascale**2
        i2_sc = wd2.imag / wt * cfg.datascale**2

        if cfg.believeweights:
            ru_sc = wt**-0.5 * cfg.datascale
            iu_sc = wt**-0.5 * cfg.datascale
        else:
            rv_sc = r2_sc - r_sc**2 # variance among real/imag msmts
            iv_sc = i2_sc - i_sc**2
            ru_sc = np.sqrt (rv_sc * wt2) / wt # uncert in mean real/img values
            iu_sc = np.sqrt (iv_sc * wt2) / wt

        mag = np.sqrt (r_sc**2 + i_sc**2)
        umag = np.sqrt (r_sc**2 * ru_sc**2 + i_sc**2 * iu_sc**2) / mag
        cfg.format.row (cfg, mjd, dtmin, r_sc, ru_sc, i_sc, iu_sc, mag, umag, n)


def dftphotom_cli (argv):
    check_usage (dftphotom_doc, argv, usageifnoargs=True)
    cfg = Config ().parse (argv[1:])
    util.logger (cfg.loglevel)
    dftphotom (cfg)
