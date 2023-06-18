# -*- mode: python; coding: utf-8 -*-
# Copyright 2016 Peter Williams <peter@newton.cx> and collaborators.
# Licensed under the MIT License.

"""pwkit.environments.ciao.data - loading up X-ray data sets

TODO: SAS, CIAO, HEASoft, etc all use very similar data formats; more code
should be shared.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = str(
    """
BaseCIAOData
GTIData
Events
OIFData
ChandraDataSet
"""
).split()

import numpy as np
from astropy.time import Time
from ...cli import warn
from ...io import Path
from ...numutil import fits_recarray_to_data_frame


def tight_bounds(minval, maxval):
    if minval > maxval:
        maxval, minval = minval, maxval

    span = maxval - minval
    if span <= 0:
        span = abs(maxval)
        if span == 0:
            span = 1

    span *= 0.05
    return minval - span, maxval + span


class BaseCIAOData(object):
    telescope = None
    "Telescope name: likely 'CHANDRA'"

    instrument = None
    "Instrument used: likely 'ACIS'"

    obsid = None
    "Observation ID as a string; resembles '17754'."

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

    def __init__(self, path, mjd0=None, t0=None):
        self.mjd0 = mjd0
        self.t0 = t0

        with Path(path).read_fits() as hdulist:
            self._process_main(hdulist, hdulist[0].header)
            assert self.mjdref is not None
            assert np.isfinite(self.mjdref)
            assert self.t0 is not None
            assert np.isfinite(self.t0)
            assert self.mjd0 is not None
            assert np.isfinite(self.mjd0)

            for hdu in hdulist[1:]:
                self._process_hdu(hdu)

    def _process_main(self, hdulist, header):
        self.telescope = header.get("TELESCOP")
        self.instrument = header.get("INSTRUME")
        self.obsid = header.get("OBS_ID")
        if "DATE-OBS" in header:
            self.obs_start = Time(header["DATE-OBS"], format="isot")
        if "DATE-END" in header:
            self.obs_stop = Time(header["DATE-END"], format="isot")

        if "MJDREF" in header:
            self.mjdref = header["MJDREF"]
            # TODO: use TIMEZERO correctly?
            self.timesys = header["TIMESYS"]
        elif "MJDREFI" in header:
            self.mjdref = float(header["MJDREFI"])
            if "MJDREFF" in header:
                self.mjdref += header["MJDREFF"]
            self.timesys = header["TIMESYS"]

    def _process_hdu(self, hdu):
        warn("ignoring HDU named %s", hdu.name)


class GTIData(BaseCIAOData):
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

    def _process_main(self, hdulist, header):
        super(GTIData, self)._process_main(hdulist, header)
        self.gti = {}

    def _process_hdu(self, hdu):
        if hdu.name in ("GTI", "STDGTI"):
            ccd = hdu.header.get("CCD_ID", 0)
            gti = self.gti[ccd] = fits_recarray_to_data_frame(hdu.data)
            gti.rename(columns={"start": "start_met", "stop": "stop_met"}, inplace=True)
            gti["start_mjd"] = gti.start_met / 86400 + self.mjdref
            gti["start_dks"] = 1e-3 * (gti.start_met - self.t0)
            gti["start_dmjd"] = gti.start_mjd - self.mjd0
            gti["stop_mjd"] = gti.stop_met / 86400 + self.mjdref
            gti["stop_dks"] = 1e-3 * (gti.stop_met - self.t0)
            gti["stop_dmjd"] = gti.stop_mjd - self.mjd0
        else:
            super(GTIData, self)._process_hdu(hdu)

    def _plot_add_gtis(self, p, ccdnum, tunit="dmjd"):
        import omega as om

        gti = self.gti[ccdnum]
        ngti = gti.shape[0]
        if ngti == 0:
            return

        gti0 = gti.at[0, "start_" + tunit]
        gti1 = gti.at[ngti - 1, "stop_" + tunit]
        smallofs = (gti1 - gti0) * 0.03

        start = gti0 - smallofs

        for i in range(ngti):
            p.add(
                om.rect.XBand(start, gti.at[i, "start_" + tunit], keyText=None),
                zheight=-1,
                dsn=1,
            )
            start = gti.at[i, "stop_" + tunit]

        p.add(om.rect.XBand(start, start + smallofs, keyText=None), zheight=-1, dsn=1)
        p.setBounds(gti0 - smallofs, gti1 + smallofs)


class Events(GTIData):
    events = None
    """DataFrame of event data. Columns are:

    ccd_id
      CCD on which the event happened.
    chipx, chipy
      Row/column on each individual chip.
    detx, dety
      Detector or "mirror off-axis" coordinates: photon positions projected onto the tangent
      plane of the unit sphere, with the tangent point being the Chandra telescope's optical
      axis. Center of ACIS is (4096, 4096).
    energy
      Best-fit energy in eV.
    expno
      Exposure number?
    fltgrade
      Event grade assigned in-flight.
    grade
      Standardized event grade?
    node_id
      Node identifier?
    pha
      PHA: "pulse height amplitude".
    pha_ro
      PHA RO?
    pi
      PI: "pulse-independent" energy. `pi = energy // 14.6 + 1`.
    tdetx, tdety
      "Tiled detector": chip coordinates on a single logical plane. Mapping to physical
      coordinates is arbitrary though: use for visualization only.
    time
      Event time in MET.
    x, y
      Tangent plane pixels like `detx` and `dety`, but locked to ICRS equatorial coordinates
      with the provided WCS mapping named `EQPOS`. Center of ACIS is (4096.5, 4096.5).
    mjd
      Event time as an MJD (added in software).
    dks
      Event time as delta-time in kiloseconds (added in software).
    dmjd
      Event time as MJD-MJD0 (added in software).

    """
    exposure = None
    """The total effective exposure time, in seconds."""

    def _process_main(self, hdulist, header):
        super(Events, self)._process_main(hdulist, header)

        hdu = hdulist["EVENTS"]
        self.exposure = hdu.header.get("exposure")
        self.events = fits_recarray_to_data_frame(hdu.data)

        if self.t0 is None:
            self.t0 = self.events.time.min()
            if not np.isfinite(self.t0):
                # Can happen if there are zero events!
                self.t0 = 51544.5  # J2000 epoch as MJD

        if self.mjd0 is None:
            self.mjd0 = np.floor(self.t0 / 86400 + self.mjdref)

        self.events["mjd"] = self.events.time / 86400 + self.mjdref
        self.events["dks"] = 1e-3 * (self.events.time - self.t0)
        self.events["dmjd"] = self.events.mjd - self.mjd0

    def _process_hdu(self, hdu):
        "We've hacked the load order a bit to get t0 and mjd0 in _process_main()."

        if hdu.name == "EVENTS":
            pass
        else:
            super(Events, self)._process_hdu(hdu)

    def _plot_binned_event_energies(
        self,
        bbinfo,
        energy_scale=1.0,
        time_key="dmjd",
        target_max_per_bin=100,
        **kwargs
    ):
        import omega as om
        from scipy.stats.mstats_extras import mjci

        p = om.RectPlot()
        time = self.events[time_key]
        energy = self.events["energy"] * energy_scale

        for ledge, redge in zip(bbinfo.ledges, bbinfo.redges):
            subset = self.events[(time >= ledge) & (time < redge)]
            n = subset.shape[0]

            if n == 0:
                continue

            nbin = max(int(np.floor(n / target_max_per_bin)), 1)
            subbin_width = (redge - ledge) / nbin

            for i in range(nbin):
                t0 = ledge + i * subbin_width
                t1 = t0 + subbin_width
                tmid = 0.5 * (t0 + t1)
                matched = (time >= t0) & (time < t1)
                subsubset = self.events[matched]
                if subsubset.shape[0] == 0:
                    continue

                subsubenergy = energy[matched]
                med_energy = subsubenergy.median()
                u_med_energy = mjci(subsubenergy.data, prob=0.5).item()
                p.addXY([t0, t1], [med_energy, med_energy], None, **kwargs)
                p.addXY(
                    [tmid, tmid],
                    [med_energy - u_med_energy, med_energy + u_med_energy],
                    None,
                    **kwargs
                )

        return p

    def plot_lightcurve(self, ccd_id=None, bin_energies=False):
        import omega as om
        from ...bblocks import tt_bblock

        if ccd_id is None:
            if len(self.gti) != 1:
                raise Exception("must specify ccd_id")
            ccd_id = list(self.gti.keys())[0]

        kev = self.events["energy"] * 1e-3
        vb = om.layout.VBox(2)

        if kev.size == 0:
            vb[0] = om.RectPlot()
            vb[1] = om.RectPlot()
            tmin = self.gti[ccd_id]["start_dmjd"].min()
            tmax = self.gti[ccd_id]["stop_dmjd"].max()
            if np.isnan(tmin):
                tmin, tmax = -1.0, 1.0
            emin, emax = -1.0, 1.0
            rmin, rmax = -1.0, 1.0
        else:
            bbinfo = tt_bblock(
                self.gti[ccd_id]["start_dmjd"],
                self.gti[ccd_id]["stop_dmjd"],
                self.events["dmjd"],
                intersect_with_bins=True,
            )
            cps = bbinfo.rates / 86400

            tmin, tmax = tight_bounds(bbinfo.ledges[0], bbinfo.redges[-1])
            emin, emax = tight_bounds(kev.min(), kev.max())
            rmin, rmax = tight_bounds(cps.min(), cps.max())

            vb[0] = om.RectPlot()
            csp = om.rect.ContinuousSteppedPainter(
                keyText="%d events" % (self.events.shape[0])
            )
            csp.setFloats(
                np.concatenate((bbinfo.ledges, bbinfo.redges[-1:])),
                np.concatenate((cps, [0])),
            )
            vb[0].add(csp)

            if bin_energies:
                vb[1] = self._plot_binned_event_energies(
                    bbinfo, energy_scale=1e-3, dsn=0
                )
            else:
                vb[1] = om.quickXY(self.events["dmjd"], kev, None, lines=0)

        vb[0].setBounds(tmin, tmax, rmin, rmax)
        vb[0].setYLabel("Count rate (ct/s)")
        vb[0].bpainter.paintLabels = False
        self._plot_add_gtis(vb[0], ccd_id)

        vb[1].setBounds(tmin, tmax, emin, emax)
        vb[1].setLabels("MJD - %d" % self.mjd0, "Energy (keV)")
        self._plot_add_gtis(vb[1], ccd_id)
        return vb

    def print_lightcurve(self, ccd_id=None, header=True):
        from ...bblocks import tt_bblock

        if ccd_id is None:
            if len(self.gti) != 1:
                raise Exception("must specify ccd_id")
            ccd_id = list(self.gti.keys())[0]

        kev = self.events["energy"] * 1e-3

        bbinfo = tt_bblock(
            self.gti[ccd_id]["start_dmjd"],
            self.gti[ccd_id]["stop_dmjd"],
            self.events["dmjd"],
            intersect_with_bins=True,
        )
        cps = bbinfo.rates / 86400
        tprev = None

        if header:
            print(
                "%7s  %8s  %8s  %5s  %7s  %4s"
                % ("Gap", "Tstart", "Tstop", "Nevt", "Ct/s", "keV")
            )

        for i in range(bbinfo.rates.size):
            tstart = bbinfo.ledges[i]
            tstop = bbinfo.redges[i]
            rate = cps[i]

            if tprev is None:
                gapstr = ""
            elif tstart == tprev:
                gapstr = "--"
            else:
                gapstr = "%7.5f" % (tstart - tprev)

            subset = self.events[
                (self.events["dmjd"] >= tstart) & (self.events["dmjd"] < tstop)
            ]
            n = subset.shape[0]
            kev = subset["energy"].median() * 1e-3

            print(
                "%7s  %8.5f  %8.5f  %5d  %7.4f  %4.1f"
                % (gapstr, tstart, tstop, n, rate, kev)
            )
            tprev = tstop


class OIFData(BaseCIAOData):
    contents = None
    """DataFrame giving information about the various files in a Chandra dataset.
    Columns are:

    member_content
      E.g. "OIF", "BADPIX", "BIAS0", "mtl"; there are duplicates.
    member_date-end
      DATE-END header of the relevant file; ISOT-format date string.
    member_date-obs
      DATE-OBS header of the relevant file; ISOT-format date string.
    member_location
      Path of the relevant file, excluding .gz extension; mine begin with "./".
    member_name
      E.g. "GROUPING", "BADPIX", "BIAS", "MTL"; there are duplicates.
    member_revision
      Revision number? All 1 in my example.
    member_size
      Size? Units are totally unclear! Range between 16 and 22000.
    member_tstart
      TSTART header of relevant file; MET seconds.
    member_tstop
      TSTOP header of relevant file; MET seconds.
    member_uri_type
      All "URL" in my example.
    member_version
      Processing version number? Mostly 1 in my example.
    member_xtension
      Extension type? "PRIMARY" or "BINTABLE" in my example.
    """

    def _process_hdu(self, hdu):
        if hdu.name == "GROUPING":
            self.contents = fits_recarray_to_data_frame(hdu.data)
        else:
            super(OIFData, self)._process_hdu(hdu)


class ChandraDataSet(object):
    path = None
    "A `pwkit.io.Path` pointing to the data set top level."

    timesys = None
    "Time reference system used; e.g. 'TT'."

    mjdref = None
    "MJD reference value; mjd = MET / 86400 + mjdref."

    t0 = None
    """Offset Mission Elapsed Time (seconds) value ; default is
    the TSTART recorded in the oif.fits file.

    """
    mjd0 = None
    "Offset MJD; default mjd0 = floor (t0 / 86400 + mjdref)."

    _toc = None
    "Table of Contents of data files within this dataset."

    _content_to_factory = {
        "oif": OIFData,
        "badpix": BaseCIAOData,
        "fov": BaseCIAOData,
        "hiresimg": BaseCIAOData,
        "evt2": Events,
        "loresimg": BaseCIAOData,
        "orbitephem1": BaseCIAOData,
        "aspsol": BaseCIAOData,
        "evt1": Events,
        "gti": BaseCIAOData,
        "msk": BaseCIAOData,
        "mtl": BaseCIAOData,
        "expstats": BaseCIAOData,
        "pbk": BaseCIAOData,
        "aspqual": BaseCIAOData,
        "angleephem": BaseCIAOData,
        "lunarephem1": BaseCIAOData,
        "solarephem1": BaseCIAOData,
    }

    def __init__(self, path):
        self.path = Path(path)
        self._toc = {"oif": (self.path / "oif.fits", OIFData)}

        if not self._toc["oif"][0].exists():
            # This is nominally racy against the load(), but should be convenient.
            raise ValueError(
                "%r does not appear to point to a Chandra data set (no oif.fits inside)"
                % (path,)
            )

        with self.lookup("oif").read_fits() as hdul:
            self.timesys = hdul["GROUPING"].header["TIMESYS"]
            self.mjdref = hdul["GROUPING"].header["MJDREF"]
            self.t0 = hdul["GROUPING"].header["TSTART"]
            self.mjd0 = np.floor(self.t0 / 86400 + self.mjdref)
            contents = fits_recarray_to_data_frame(hdul["GROUPING"].data)

        for idx, row in contents.iterrows():
            content = row.member_content.strip().lower()

            if content == "oif":
                continue  # already dealt with

            if content in ("bias0", "obcsol"):
                # XXX: multiple files; not sure what to do about them
                continue

            factory = self._content_to_factory.get(content)
            if factory is None:
                warn("unexpected member_content %r in %s:oif.fits", content, self.path)
                continue

            p = row.member_location.strip()
            if p.startswith("./"):
                p = p[2:]
            p = self.path / p

            if not p.exists():
                p = p.with_name(p.name + ".gz")
                if not p.exists():
                    warn(
                        "missing member %r (from %s:oif.fits)",
                        row.member_location,
                        self.path,
                    )
                    continue

            self._toc[content] = (p, factory)

    def lookup(self, itemname):
        return self._toc[itemname][0]

    def load(self, itemname, **kwargs):
        path, factory = self._toc[itemname]
        return factory(path, t0=self.t0, mjd0=self.mjd0, **kwargs)
