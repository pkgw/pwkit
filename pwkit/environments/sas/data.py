# -*- mode: python; coding: utf-8 -*-
# Copyright 2015 Peter Williams <peter@newton.cx> and collaborators.
# Licensed under the MIT License.

"""pwkit.environments.sas.data - loading up SAS data sets

"""
from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = str ('BaseSASData Events GTIData Lightcurve RegionData').split ()

import numpy as np, pandas as pd
from six.moves import range
from astropy.time import Time
from ... import astutil, cli
from ...io import Path
from ...numutil import fits_recarray_to_data_frame


class BaseSASData (object):
    telescope = None
    "Telescope name: likely 'XMM'"

    instrument = None
    "Instrument used: likely 'EMOS1', 'EMOS2', 'EPN'"

    obsid = None
    "Observation ID as a string; resembles '0748391401'."

    expid = None
    "Exposure ID as a string; resembles '0748391401003'."

    revnum = None
    "Revolution (orbit) number as an integer."

    timesys = None
    "Time reference system used; e.g. 'TT'."

    mjdref = None
    "MJD reference value; mjd = MET / 86400 + mjdref."

    t0 = None
    """Offset Mission Elapsed Time (seconds) value ; default is
    the first data timestamp in the data set.

    """
    mjd0 = None
    "Offset MJD; default mjd0 = floor (t0 / 86400 + mjdref)."

    obs_start = None
    "Start time of the observation as an astropy.time.Time() instance."

    obs_end = None
    "End time of the observation as an astropy.time.Time() instance."

    targ_name = None
    "Name of the observing target as a string."

    targ_ra = None
    "RA of the observing target, in radians."

    targ_dec = None
    "Dec of the observing target, in radians."

    proj_csys = None
    "Coordinate system of the Y/X projection as a string; e.g. 'FK5'."

    proj_equinox = None
    "Equinox of the Y/X projection as a float; e.g. 2000.0."

    proj_types = None
    """Axis types of the Y/X projection as a pair of strings; e.g. ['DEC--TAN',
    'RA---TAN'].

    """
    proj_crpix = None
    """Reference point pixel values of the Y/X projection as a 2-element ndarray.
    Order is [y,x].

    """
    proj_crval = None
    """Reference point coordinate values of the Y/X projection as a 2-element
    ndarray, in radians. Order is [dec,ra] (= [lat,lon] = [y,x]).

    """
    proj_cdelt = None
    """Pixel-to-world scale factors of the Y/X projection as a 2-element ndarray,
    in radians per pixel. Order is [y,x].

    """
    calindex = None
    "DataFrame of calibration table info. Schema to be investigated."

    def __init__ (self, path, mjd0=None, t0=None):
        self.mjd0 = mjd0
        self.t0 = t0

        with Path (path).read_fits () as hdulist:
            self._process_main (hdulist, hdulist[0].header)
            assert self.mjdref is not None
            assert self.t0 is not None
            assert self.mjd0 is not None

            for hdu in hdulist[1:]:
                self._process_hdu (hdu)


    def _process_main (self, hdulist, header):
        self.telescope = header.get ('TELESCOP')
        self.instrument = header.get ('INSTRUME')
        self.obsid = header.get ('OBS_ID')
        self.expid = header.get ('EXP_ID')
        self.revnum = header.get ('REVOLUT')
        self.filter = header.get ('FILTER')
        self.targ_name = header.get ('OBJECT')
        if 'DATE-OBS' in header:
            self.obs_start = Time (header['DATE-OBS'], format='isot')
        if 'DATE-END' in header:
            self.obs_stop = Time (header['DATE-END'], format='isot')
        if 'RA_OBJ' in header:
            self.targ_ra = header['RA_OBJ'] * astutil.D2R
            self.targ_dec = header['DEC_OBJ'] * astutil.D2R

        if 'RADECSYS' in header:
            if header['REFYCUNI'] != 'deg' or header['REFXCUNI'] != 'deg':
                raise ValueError ('expect projection to be in degree units')

            self.proj_csys = header['RADECSYS']
            self.proj_equinox = header['EQUINOX']
            self.proj_types = [header['REFYCTYP'], header['REFXCTYP']]
            self.proj_crpix = np.asarray ([header['REFYCRPX'], header['REFXCRPX']])
            self.proj_crval = np.asarray ([header['REFYCRVL'], header['REFXCRVL']])
            self.proj_crval *= astutil.D2R
            self.proj_cdelt = np.asarray ([header['REFYCDLT'], header['REFXCDLT']])
            self.proj_cdelt *= astutil.D2R


    def _process_hdu (self, hdu):
        if hdu.name == 'CALINDEX':
            self.calindex = fits_recarray_to_data_frame (hdu.data)
        else:
            cli.warn ('ignoring HDU named %s', hdu.name)


class GTIData (BaseSASData):
    gti = None
    """Dict mapping CCD number to DataFrames of GTI info. Index: integers.
    Columns:

        start_met  - GTI start time in MET seconds
        stop_met   - GTI stop time in MET seconds
        start_mjd  - GTI start time as MJD
        stop_mjd   - GTI stop time as MJD
        start_dks  - GTI start time as delta-kiloseconds
        stop_dks   - GTI stop time as delta-kiloseconds
        start_dmjd - GTI start time as `mjd - mjd0`
        stop_dmjd  - GTI stop time as `mjd - mjd0`

    """
    def _process_main (self, hdulist, header):
        super (GTIData, self)._process_main (hdulist, header)
        self.gti = {}


    def _process_hdu (self, hdu):
        if hdu.name.startswith ('STDGTI'):
            ccd = int (hdu.name[6:])
            gti = self.gti[ccd] = fits_recarray_to_data_frame (hdu.data)
            gti.rename (columns={'start': 'start_met', 'stop': 'stop_met'},
                        inplace=True)
            gti['start_mjd'] = gti.start_met / 86400 + self.mjdref
            gti['start_dks'] = 1e-3 * (gti.start_met - self.t0)
            gti['start_dmjd'] = gti.start_mjd - self.mjd0
            gti['stop_mjd'] = gti.stop_met / 86400 + self.mjdref
            gti['stop_dks'] = 1e-3 * (gti.stop_met - self.t0)
            gti['stop_dmjd'] = gti.stop_mjd - self.mjd0
        else:
            super (GTIData, self)._process_hdu (hdu)


    def _plot_add_gtis (self, p, ccdnum, tunit='dmjd'):
        import omega as om

        gti = self.gti[ccdnum]
        ngti = gti.shape[0]
        gti0 = gti.at[0,'start_'+tunit]
        gti1 = gti.at[ngti-1,'stop_'+tunit]
        smallofs = (gti1 - gti0) * 0.03

        start = gti0 - smallofs

        for i in range (ngti):
            p.add (om.rect.XBand (start, gti.at[i,'start_'+tunit], keyText=None), zheight=-1, dsn=1)
            start = gti.at[i,'stop_'+tunit]

        p.add (om.rect.XBand (start, start + smallofs, keyText=None), zheight=-1, dsn=1)
        p.setBounds (gti0 - smallofs, gti1 + smallofs)


class RegionData (BaseSASData):
    regions = None
    """Dict mapping identifier to DataFrames of selection region info. Identifiers
    are like "00106" in SAS but I don't understand their significance.
    DataFrame index: integers. Columns:
        shape     - region shape as string: 'CIRCLE', ...
        x         - region center in X
        y         - region center in Y
        r         - circle radius (meaning for other shapes?)
        component - ?

    """
    def _process_main (self, hdulist, header):
        super (RegionData, self)._process_main (hdulist, header)
        self.regions = {}


    def _process_hdu (self, hdu):
        if hdu.name.startswith ('REG'):
            ident = hdu.name[3:]
            self.regions[ident] = fits_recarray_to_data_frame (hdu.data)
        else:
            super (RegionData, self)._process_hdu (hdu)


class Events (GTIData, RegionData):
    filter = None
    "Filter used as a string; e.g. 'Medium'."

    elapsed = None
    "Time elapsed over all events, in seconds."

    ccd_info = None
    """DataFrame of scalar CCD info. Index: CCD numbers. Columns:
         ontime   - CCD on time (sum of GTIs) in seconds
         livetime - CCD live time (always less than ontime) in seconds
    """

    events = None
    """DataFrame of events. Index: integers. Columns:
          ccdnr    - CCD number of event
          detx     - detector X; not very useful
          dety     - detector Y
          flag     -
          pat_id   -
          pat_seq  -
          pattern  -
          pha      - pulse height amplitude
          pi       - pulse-independent energy
          rawx     - raw CCD X
          rawy     - raw CCD Y
          time     - event time in MET seconds
          time_raw - ?
          x        - corrected X
          y        - corrected Y
          mjd      - event time as MJD
          dks      - event time as delta-kiloseconds
          dmjd     - event time as `mjd - mjd0`
    """
    offsets = None
    """DataFrame of "offsets". Some kind of CCD meta-info."""

    #exposure = None
    # Commented out; exposure tables are quite large and I don't know if they're
    # useful for anything.

    badpix = None
    """Dict mapping CCD number to DataFrames of bad-pixel info. Index: integers.
    Columns: badflag, rawx, rawy, type, yextent.

    """
    def _process_main (self, hdulist, header):
        super (Events, self)._process_main (hdulist, header)

        ccd_nums = set ()
        for hdu in hdulist[1:]:
            if hdu.name.startswith ('EXPOSU'):
                ccd_nums.add (int (hdu.name[6:]))

        ccd_nums = sorted (ccd_nums)

        #self.exposure = {}
        self.badpix = {}
        self.ccd_info = pd.DataFrame ({}, index=ccd_nums)

        hdu = hdulist['EVENTS']
        self.events = fits_recarray_to_data_frame (hdu.data)
        self.mjdref = hdu.header['MJDREF']

        if self.t0 is None:
            self.t0 = self.events.time.min ()

        if self.mjd0 is None:
            self.mjd0 = np.floor (self.t0 / 86400 + self.mjdref)

        self.events['mjd'] = self.events.time / 86400 + self.mjdref
        self.events['dks'] = 1e-3 * (self.events.time - self.t0)
        self.events['dmjd'] = self.events.mjd - self.mjd0
        self.timesys = hdu.header['TIMESYS']
        self.elapsed = hdu.header['TELAPSE']
        self.ccd_info['ontime'] = np.nan
        self.ccd_info['livetime'] = np.nan

        for ccd in ccd_nums:
            self.ccd_info.at[ccd,'ontime'] = hdu.header['ONTIME%02d' % ccd]
            self.ccd_info.at[ccd,'livetime'] = hdu.header['LIVETI%02d' % ccd]


    def _process_hdu (self, hdu):
        if hdu.name == 'EVENTS':
            pass
        elif hdu.name == 'OFFSETS':
            self.offsets = fits_recarray_to_data_frame (hdu.data)
        elif hdu.name.startswith ('EXPOSU'):
            # These data are very large, and their purpose is unclear to me.
            pass
            #ccd = int (hdu.name[6:])
            #exp = self.exposure[ccd] = fits_recarray_to_data_frame (hdu.data)
            #exp['mjd'] = exp.time / 86400 + self.mjdref
            #exp['dks'] = 1e-3 * (exp.time - self.t0)
        elif hdu.name.startswith ('BADPIX'):
            ccd = int (hdu.name[6:])
            self.badpix[ccd] = fits_recarray_to_data_frame (hdu.data)
        else:
            super (Events, self)._process_hdu (hdu)


    def plot_pi_time (self, ccdnum):
        import omega as om

        p = om.quickDF (self.events[self.events.ccdnr == ccdnum][['dmjd', 'pi']],
                        self.targ_name,
                        lines=False)
        self._plot_add_gtis (p, ccdnum)
        p.setLabels ('MJD - %.0f' % self.mjd0, 'PI')
        return p


class Lightcurve (GTIData, RegionData):
    filter = None
    "Filter used as a string; e.g. 'Medium'."

    binsize = None
    "The bin size used when creating the light curve, in seconds."

    start_met = start_mjd = start_dks = start_dmjd = None
    "The binning start time, in various time units."

    stop_met = stop_mjd = stop_dks = stop_dmjd = None
    "The binning stop time, in various time units."

    energy_type = None
    "The name of the energy column used; likely 'PHA' or 'PI'."

    energy_min = None
    "The lower limit on the energy column; units depend on `energy_type`."

    energy_max = None
    "The upper limit on the energy column; units depend on `energy_type`."

    exposure = None
    "Weighted live time of all the CCDs in the extraction region."

    lc = None
    """DataFrame of light curve bins. Index: integers. Columns:

        time       - timestamp of bin center in MET seconds
        mjd        - timestamp of bin center as MJD
        dks        - timestamp of bin center in delta-kiloseconds
        dmjd       - timestamp of bin center as `mjd - mjd0`
        left_dmjd  - timestamp of bin left edge as `mjd - mjd0`
        right_dmjd - timestamp of bin right edge as `mjd - mjd0`
        counts     - number of events in this bin
        rate       - event rate in counts per second
        u_rate     - uncertainty on `rate`.

    I'm pretty sure that `u_rate` may not be just the straight Poisson noise
    since there may be GTI gaps.

    """
    def _process_main (self, hdulist, header):
        super (Lightcurve, self)._process_main (hdulist, header)

        # Early loading of bulk data

        hdu = hdulist['RATE']
        self.lc = fits_recarray_to_data_frame (hdu.data)
        self.lc.rename (columns={'error': 'u_rate'}, inplace=True)

        # Straighten out timekeeping

        self.mjdref = float (hdu.header['MJDREF'])
        self.timesys = hdu.header.get ('TIMESYS')

        if self.t0 is None:
            self.t0 = self.lc.time.min ()

        if self.mjd0 is None:
            self.mjd0 = np.floor (self.t0 / 86400 + self.mjdref)

        # Fill in the rest

        self.filter = hdu.header.get ('FILTER')
        self.binsize = hdu.header.get ('TIMEDEL')
        self.start_met = hdu.header['TSTART']
        self.start_mjd = self.start_met / 86400 + self.mjdref
        self.start_dks = 1e-3 * (self.start_met - self.t0)
        self.start_dmjd = self.start_mjd - self.mjd0
        self.stop_met = hdu.header['TSTOP']
        self.stop_mjd = self.stop_met / 86400 + self.mjdref
        self.stop_dks = 1e-3 * (self.stop_met - self.t0)
        self.stop_dmjd = self.stop_mjd - self.mjd0
        self.energy_min = hdu.header.get ('CHANMIN')
        self.energy_max = hdu.header.get ('CHANMAX')
        self.energy_type = hdu.header.get ('CHANTYPE')
        self.exposure = hdu.header.get ('EXPOSURE')

        self.lc['counts'] = self.lc.rate * self.binsize
        self.lc['mjd'] = self.lc.time / 86400 + self.mjdref
        self.lc['dks'] = 1e-3 * (self.lc.time - self.t0)
        self.lc['dmjd'] = self.lc.mjd - self.mjd0
        self.lc['left_dmjd'] = self.lc.dmjd - 0.5 * self.binsize / 86400
        self.lc['right_dmjd'] = self.lc.dmjd + 0.5 * self.binsize / 86400


    def _process_hdu (self, hdu):
        if hdu.name == 'RATE':
            pass
        else:
            super (Lightcurve, self)._process_hdu (hdu)


    def plot_curve (self, ccdnum=None):
        import omega as om

        p = om.quickDF (self.lc[['dmjd', 'rate', 'u_rate']].dropna (),
                        self.targ_name,
                        lines=False)
        if ccdnum is not None:
            self._plot_add_gtis (p, ccdnum)
        p.setLabels ('MJD - %.0f' % self.mjd0, 'Count rate')
        return p
