# -*- mode: python; coding: utf-8 -*-
# Copyright 2015 Peter Williams <peter@newton.cx> and collaborators.
# Licensed under the MIT License.

"""pwkit.environments.sas.data - loading up SAS data sets

"""
from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = (b'').split ()

import numpy as np, pandas as pd
from astropy.time import Time
from ... import astutil, cli
from ...io import Path
from ...numutil import fits_recarray_to_data_frame


class Events (object):
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

    filter = None
    "Filter used as a string; e.g. 'Medium'."

    t0 = None
    """Minimum time in EVENTS table, in Mission Elapsed time units (MET; seconds).
    dks = 1e-3 * (MET - t0).

    """
    mjdref = None
    "MJD reference value; mjd = MET / 86400 + mjdref."

    mjd0 = None
    "Offset MJD; mjd0 = floor (t0 / 86400 + mjdref)."

    timesys = None
    "Time reference system used; e.g. 'TT'."

    elapsed = None
    "Time elapsed over all events, in seconds."

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

    calindex = None
    "DataFrame of calibration table info."

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
    #exposure = None
    # Commented out; exposure tables are quite large and I don't know if they're
    # useful for anything.

    badpix = None
    """Dict mapping CCD number to DataFrames of bad-pixel info. Index: integers.
    Columns: badflag, rawx, rawy, type, yextent.

    """
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
    def __init__ (self, path, mjd0=None):
        with Path (path).read_fits () as hdulist:
            hmain = hdulist[0].header

            self.telescope = hmain['TELESCOP']
            self.instrument = hmain['INSTRUME']
            self.obsid = hmain['OBS_ID']
            self.expid = hmain['EXP_ID']
            self.revnum = hmain['REVOLUT']
            self.filter = hmain['FILTER']
            self.obs_start = Time (hmain['DATE-OBS'], format='isot')
            self.obs_stop = Time (hmain['DATE-END'], format='isot')
            self.targ_name = hmain['OBJECT']
            self.targ_ra = hmain['RA_OBJ'] * astutil.D2R
            self.targ_dec = hmain['DEC_OBJ'] * astutil.D2R

            if hmain['REFYCUNI'] != 'deg' or hmain['REFXCUNI'] != 'deg':
                raise ValueError ('expect projection to be in degree units')

            self.proj_csys = hmain['RADECSYS']
            self.proj_equinox = hmain['EQUINOX']
            self.proj_types = [hmain['REFYCTYP'], hmain['REFXCTYP']]
            self.proj_crpix = np.asarray ([hmain['REFYCRPX'], hmain['REFXCRPX']])
            self.proj_crval = np.asarray ([hmain['REFYCRVL'], hmain['REFXCRVL']])
            self.proj_crval *= astutil.D2R
            self.proj_cdelt = np.asarray ([hmain['REFYCDLT'], hmain['REFXCDLT']])
            self.proj_cdelt *= astutil.D2R

            ccd_nums = set ()
            for hdu in hdulist[1:]:
                if hdu.name.startswith ('EXPOSU'):
                    ccd_nums.add (int (hdu.name[6:]))

            ccd_nums = sorted (ccd_nums)
            n_ccds = len (ccd_nums)
            ccd_map = dict ((n, i) for i, n in enumerate (ccd_nums))

            self.gti = {}
            #self.exposure = {}
            self.badpix = {}
            self.regions = {}
            self.ccd_info = pd.DataFrame ({}, index=ccd_nums)

            hdu = hdulist['EVENTS']
            self.events = fits_recarray_to_data_frame (hdu.data)
            self.t0 = self.events.time.min ()
            self.mjdref = hdu.header['MJDREF']

            if mjd0 is not None:
                self.mjd0 = mjd0
            else:
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

            for hdu in hdulist[1:]:
                if hdu.name == 'EVENTS':
                    pass
                elif hdu.name == 'OFFSETS':
                    self.offsets = fits_recarray_to_data_frame (hdu.data)
                elif hdu.name == 'CALINDEX':
                    self.calindex = fits_recarray_to_data_frame (hdu.data)
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
                elif hdu.name.startswith ('STDGTI'):
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
                elif hdu.name.startswith ('REG'):
                    ident = hdu.name[3:]
                    self.regions[ident] = fits_recarray_to_data_frame (hdu.data)
                else:
                    cli.warn ('ignoring event HDU %s', hdu.name)


    def plot_pi_time (self, ccdnum):
        import omega as om

        p = om.quickDF (self.events[self.events.ccdnr == ccdnum][['dmjd', 'pi']],
                        self.targ_name,
                        lines=False)

        gti = self.gti[ccdnum]
        ngti = gti.shape[0]
        gti0 = gti.at[0,'start_dmjd']
        gti1 = gti.at[ngti-1,'stop_dmjd']
        smallofs = (gti1 - gti0) * 0.03

        start = gti0 - smallofs

        for i in xrange (ngti):
            p.add (om.rect.XBand (start, gti.at[i,'start_dmjd'], keyText=None), zheight=-1, dsn=1)
            start = gti.at[i,'stop_dmjd']

        p.add (om.rect.XBand (start, start + smallofs, keyText=None), zheight=-1, dsn=1)
        p.setLabels ('MJD - %.0f' % self.mjd0, 'PI')
        p.setBounds (gti0 - smallofs, gti1 + smallofs)

        return p
