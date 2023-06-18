#! /usr/bin/env python
# Copyright 2012-2015 Peter Williams <peter@newton.cx> and collaborators.
# Licensed under the MIT License.

"""Compute diagnostics regarding the quality of gain/phase calibration.

NB. The GainCal class should be generically useful.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = str(
    """
GainCal
EmulatedMultiStamp
ManualStampKeyPainter
Config
DiagnosticsTool
gpdiagnostics_cli
"""
).split()

import numpy as np
from scipy.stats import scoreatpercentile
import omega as om, omega.render
from pwkit.kwargv import ParseKeywords, Custom
from pwkit.io import Path
from pwkit import cli
from pwkit.environments.casa import util


gpdiagnostics_doc = """
casatask gpdiagnostics vis=MS [keywords...]

Solve for antenna calibration gains assuming a point-source model, and
generate diagnostic plots useful for identifying bad baselines and/or
antennas.

vis=
  Path of the MeasurementSet dataset to read. Required.

out=
  Path where output plots will be saved.

datacol=
  Name of the column to use for visibility data. Defaults to 'data'.
  You might want it to be 'corrected_data'.

dims=WIDTH,HEIGHT
  Dimensions of the output plot; units are pixels or points depending on the
  format.

margins=TOP,RIGHT,BOTTOM,LEFT
  Size of the blank margins around the output plot; units are pixels or points
  depending on the format.

array=, baseline=, field=, observation=, polarization=, scan=,
scanintent=, spw=, taql=, time=, uvdist=
  MeasurementSet selectors used to filter the input data.
  Default polarization is 'RR,LL'. All polarizations are averaged
  together, so mixing parallel- and cross-hand pols is almost
  never what you want to do.

loglevel=
  Level of detail from CASA logging system. Default: "warn"; allowed: "severe
    warn info info1 info2 info3 info4 info5 debug1 debug2 debugging"

This task reads in the UV data in `vis` and, for each unique combination of
timestamp, spectral window, and polarization, solves for per-antenna gain
amplitudes and phases, assuming a point-source model of unit flux density at
the phase center. It emits a plot showing the residuals to the gain solution
in terms of calibrated visibilities, allowing identification of bad baselines
and so on.

You probably want to time-average your data before running this program (see
`casatask split`) because it is not clever enough to average the input data on
the fly.

"""


class GainCal(object):
    nants = None
    "Number of unique antennas in this solution."

    ants = None
    """Array mapping antenna solution-local antenna indices to global antenna
    numbers.

    """
    ant_to_antidx = None
    """Dictionary mapping global antenna number to solution-local antenna index."""

    nsamps = None
    "Number of visibility samples/measurements in this solution."

    vis = None
    "The visibility samples; shape (nsamps,)."

    blidxs = None
    "Solution local antenna indices for each baseline; shape (nsamps,2)."

    ncontrib = None
    "Number of sub-samples contributing to each visibility sample; shape (nsamps,)."

    nperant = None
    "Number of samples for each antenna; shape (nants,)."

    gains = None
    "The solved-for antenna gains; shape (nants,)."

    normvis = None
    """The calibrated visibilities, i.e. self.vis multiplied by self.gains
    appropriately; shape (nsamps,).

    """

    def fill_from_dict(self, bybl):
        self.nsamps = len(bybl)

        seenants = set()
        for a1, a2 in bybl.keys():
            seenants.add(a1)
            seenants.add(a2)
        self.ants = np.array(sorted(seenants))
        self.nants = self.ants.size
        self.ant_to_antidx = dict((num, idx) for idx, num in enumerate(self.ants))

        self.ncontrib = np.empty((self.nsamps,), dtype=int)
        self.vis = np.empty((self.nsamps,), dtype=complex)
        self.blidxs = np.empty((self.nsamps, 2), dtype=int)
        self.nperant = np.zeros((self.nants,), dtype=int)

        for i, (bl, (data, flags)) in enumerate(bybl.items()):
            ok = ~flags
            self.ncontrib[i] = ok.sum()
            self.vis[i] = data[ok].mean()
            i1 = self.ant_to_antidx[bl[0]]
            i2 = self.ant_to_antidx[bl[1]]
            self.blidxs[i] = i1, i2
            self.nperant[i1] += 1
            self.nperant[i2] += 1

    def solve(self):
        if self.nants > self.nsamps:
            cli.warn(
                "not enough measurements to solve: %d ants, %d samples"
                % (self.nants, self.nsamps)
            )
            return

        # First solve for (log) amplitudes, which we can do as a classic
        # linear least squares problem (in log space). We're implicitly
        # modeling the source as 1+0j on all baselines, i.e., we're assuming a
        # point source and solving for amplitudes in units of the source flux
        # density.

        lna_A = np.zeros((self.nsamps, self.nants))

        for i in range(self.nsamps):
            i1, i2 = self.blidxs[i]
            lna_A[i, i1] = 1
            lna_A[i, i2] = 1

        lna_b = np.log(np.abs(self.vis))
        lna_x, lna_chisq, lna_rank, lna_sing = np.linalg.lstsq(lna_A, lna_b)
        lna_chisq = lna_chisq[0]

        # We just solved for log values to model visibilities; to bring
        # visibilities into model domain, we need the inverses of these
        # values. We can then normalize the amplitudes of all of the observed
        # visibilities.

        amps = np.exp(-lna_x)
        normvis = self.vis.copy()

        for i in range(self.nsamps):
            i1, i2 = self.blidxs[i]
            normvis[i] *= amps[i1] * amps[i2]

        # Now, solve for the phases with a bespoke (but simple) iterative
        # algorithm. For each antenna we just compute the phase of the summed
        # differences between it and the "model" and alter the phase by that.
        # Loosely modeled on MIRIAD gpcal PhaseSol().

        curphasors = np.ones(self.nants, dtype=complex)
        newphasors = np.empty(self.nants, dtype=complex)
        tol = 1e-5
        damping = 0.9

        for iter_num in range(100):
            newphasors.fill(0)

            for i, vis in enumerate(normvis):
                i1, i2 = self.blidxs[i]
                newphasors[i1] += curphasors[i2] * vis
                newphasors[i2] += curphasors[i1] * np.conj(vis)

            newphasors /= np.abs(newphasors)
            temp = curphasors + damping * (newphasors - curphasors)
            temp /= np.abs(temp)
            delta = (np.abs(temp - curphasors) ** 2).mean()
            # print ('ZZ', iter_num, delta, np.angle (temp, deg=True))
            curphasors = temp

            if delta < tol:
                break

        # Calibrate out phases too

        np.conj(curphasors, curphasors)
        gains = amps * curphasors

        for i in range(self.nsamps):
            i1, i2 = self.blidxs[i]
            normvis[i] *= curphasors[i1] * np.conj(curphasors[i2])

        self.gains = gains
        self.normvis = normvis

    def stats(self, antnames):
        """XXX may be out of date."""
        nbyant = np.zeros(self.nants, dtype=int)
        sum = np.zeros(self.nants, dtype=complex)
        sumsq = np.zeros(self.nants)
        q = np.abs(self.normvis - 1)

        for i in range(self.nsamps):
            i1, i2 = self.blidxs[i]
            nbyant[i1] += 1
            nbyant[i2] += 1
            sum[i1] += q[i]
            sum[i2] += q[i]
            sumsq[i1] += q[i] ** 2
            sumsq[i2] += q[i] ** 2

        avg = sum / nbyant
        std = np.sqrt(sumsq / nbyant - avg**2)
        navg = 1.0 / np.median(avg)
        nstd = 1.0 / np.median(std)

        for i in range(self.nants):
            print(
                "  %2d %10s %3d %f %f %f %f"
                % (
                    i,
                    antnames[i],
                    nbyant[i],
                    avg[i],
                    std[i],
                    avg[i] * navg,
                    std[i] * nstd,
                )
            )

    def get_normalized(self, antidx):
        buff = np.empty(self.nperant[antidx], dtype=complex)
        otherant = np.empty(self.nperant[antidx], dtype=int)
        ofs = 0

        for i in range(self.nsamps):
            i1, i2 = self.blidxs[i]

            if i1 == antidx:
                buff[ofs] = self.normvis[i]
                otherant[ofs] = i2
                ofs += 1
            elif i2 == antidx:
                buff[ofs] = self.normvis[i]
                otherant[ofs] = i1
                ofs += 1

        return buff, self.ants[otherant]


class Config(ParseKeywords):
    vis = Custom(str, required=True)
    out = str
    dims = [800, 900]
    margins = [4, 4, 4, 4]
    loglevel = "warn"
    datacol = "data"

    array = str
    baseline = str
    field = str
    observation = str
    polarization = str
    scan = str
    scanintent = str
    spw = str
    taql = str
    time = str
    uvdist = str


ms_selectors = frozenset(
    """
array baseline field observation polarization scan scanintent
spw taql time uvdist
""".split()
)

iter_maxrows = 1024

b = util.sanitize_unicode


class EmulatedMultiStamp(om.stamps.RStamp):
    def __init__(self, shape, cnum, size, fill=True):
        self.shape = shape
        self.cnum = cnum
        self.size = size
        self.fill = fill

    def paintAt(self, ctxt, style, x, y):
        symfunc = style.data.getStrictSymbolFunc(self.shape)
        c = style.colors.getDataColor(self.cnum)

        ctxt.save()
        om.styles.apply_color(ctxt, c)
        ctxt.translate(x, y)
        symfunc(ctxt, style, self.size, self.fill)
        ctxt.restore()


class ManualStampKeyPainter(om.rect.GenericKeyPainter):
    def __init__(self, keytext, stamp, stampstyle=None):
        self.keytext = keytext
        self.stamp = stamp
        self.stampStyle = stampstyle
        stamp.setData("no data allowed for key-only stamp")

    def _getText(self):
        return self.keytext

    def _drawLine(self):
        return False

    def _drawStamp(self):
        return True

    def _drawRegion(self):
        return False

    def _applyStampStyle(self, style, ctxt):
        style.apply(ctxt, self.stampStyle)

    def _getStamp(self):
        return self.stamp


class DiagnosticsTool(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def start_chunk(self):
        self.buffer = {}

    def finish_chunk(self, *key):
        if not len(self.buffer):
            return

        spw, time = key

        for polname, bybl in self.buffer.items():
            self.finish_polarization(spw, time, polname, bybl)

    def finish_polarization(self, spw, time, polname, bybl):
        gc = GainCal()
        gc.fill_from_dict(bybl)
        gc.solve()
        self.results[spw, polname, time] = gc

    def load(self):
        vis = Path(self.cfg.vis)
        tb = util.tools.table()
        ms = util.tools.ms()

        # Polarization info

        tb.open(b(vis / "DATA_DESCRIPTION"))
        ddid_to_pid = tb.getcol(b"POLARIZATION_ID")
        ddid_to_spwid = tb.getcol(b"SPECTRAL_WINDOW_ID")
        tb.close()

        tb.open(b(vis / "POLARIZATION"))
        numcorrs = tb.getcol(b"NUM_CORR")
        npids = numcorrs.size
        prodinfo = [None] * npids

        for i in range(npids):
            corrtypes = tb.getcell(b"CORR_TYPE", i)
            prodinfo[i] = [
                (j, util.pol_names[c])
                for j, c in enumerate(corrtypes)
                if util.pol_is_intensity[c]
            ]

        tb.close()

        ddprods = [prodinfo[p] for p in ddid_to_pid]

        # Antenna info

        tb.open(b(vis / "ANTENNA"))
        nants = tb.getcol(b"DISH_DIAMETER").size
        names = tb.getcol(b"NAME")
        stations = tb.getcol(b"STATION")
        self.antnames = ["%s@%s" % (names[i], stations[i]) for i in range(nants)]

        # Open and set up filtering. msselect() says it supports 'polarization' as
        # a field, but it doesn't seem to do anything?

        ms.open(b(vis))

        mssel = {}
        for sel in ms_selectors:
            val = getattr(self.cfg, sel, None)
            if val is not None:
                mssel[sel] = val
        ms.msselect(b(mssel))

        totrows = 0
        curkey = ()
        self.results = {}
        self.start_chunk()

        colnames = b(
            [self.cfg.datacol] + "time antenna1 antenna2 data_desc_id flag uvw".split()
        )
        ms.iterinit(maxrows=iter_maxrows)
        ms.iterorigin()

        while True:
            cols = ms.getdata(items=colnames)
            if self.cfg.datacol not in cols:
                raise Exception("no such data column %s" % self.cfg.datacol)

            # flag and data are (npol, nchan, nrows)
            #   [`data` is complex128!!! converting is super slow and sad :-(]
            # uvw is (3, nrows)
            # rest are scalars, shape (nrows,)

            nrows = cols["time"].size
            data = cols[self.cfg.datacol]
            flags = cols["flag"]

            for i in range(nrows):
                bl = (cols["antenna1"][i], cols["antenna2"][i])
                if bl[0] == bl[1]:
                    continue

                t = cols["time"][i] / 86400.0  # CASA to MJD
                ddid = cols["data_desc_id"][i]
                spw = ddid_to_spwid[ddid]
                pi = ddprods[ddid]
                npol = len(pi)
                totrows += 1
                key = (spw, t)  # XXX being cavalier about ddid vs spwid

                if key != curkey:
                    self.finish_chunk(*curkey)
                    curkey = key
                    self.start_chunk()

                for j, polname in pi:
                    # uvw = cols['uvw'][:,i] * util.INVERSE_C_MNS
                    d = data[j, :, i]
                    f = flags[j, :, i]
                    if not np.all(f):
                        self.buffer.setdefault(polname, {})[bl] = (d, f)

            if not ms.iternext():
                break

        ms.close()
        self.finish_chunk(*curkey)

    def plot(self):
        if isinstance(self.cfg.out, omega.render.Pager):
            # This is for non-CLI invocation.
            pager = self.cfg.out
        elif self.cfg.out is None:
            from omega import gtk3

            pager = om.makeDisplayPager()
        else:
            pager = om.makePager(
                self.cfg.out,
                dims=self.cfg.dims,
                margins=self.cfg.margins,
                style=om.styles.ColorOnWhiteVector(),
            )

        # Collect the data as loaded

        skeys = sorted(self.results.keys())
        normalized = {}
        spws = set()
        polns = set()

        for antnum, antname in enumerate(self.antnames):
            for key in skeys:
                spw, poln, time = key
                gc = self.results[key]
                idx = gc.ant_to_antidx.get(antnum)
                if idx is None:
                    continue

                samps, otherant = gc.get_normalized(idx)
                bysp = normalized.setdefault(antname, {})
                bytime = bysp.setdefault((spw, poln), {})
                bytime[time] = (samps, otherant)
                polns.add(poln)
                spws.add(spw)

        spws = sorted(spws)
        polns = sorted(polns)
        spwseq = dict((s, i) for (i, s) in enumerate(spws))
        polnseq = dict((p, i) for (i, p) in enumerate(polns))

        # Group by time and get dims

        rmin = rmax = imin = imax = amin = amax = None

        for antname, bysp in normalized.items():
            for spwpol in list(bysp.keys()):
                bytime = bysp[spwpol]
                times = sorted(bytime.keys())
                samps = np.concatenate(tuple(bytime[t][0] for t in times))
                otherants = np.concatenate(tuple(bytime[t][1] for t in times))
                logamps = np.log10(np.abs(samps))
                bysp[spwpol] = samps, otherants, logamps

                if rmin is None:
                    rmin = samps.real.min()
                    rmax = samps.real.max()
                    imin = samps.imag.min()
                    imax = samps.imag.max()
                    amin = logamps.min()
                    amax = logamps.max()
                else:
                    rmin = min(rmin, samps.real.min())
                    rmax = max(rmax, samps.real.max())
                    imin = min(imin, samps.imag.min())
                    imax = max(imax, samps.imag.max())
                    amin = min(amin, logamps.min())
                    amax = max(amax, logamps.max())

        # Square things up in the real/imag plot and add little margins

        rrange = rmax - rmin
        irange = imax - imin
        arange = amax - amin

        if rrange < irange:
            delta = 0.5 * (irange - rrange)
            rmax += delta
            rmin -= delta
            rrange = irange
        else:
            delta = 0.5 * (rrange - irange)
            imax += delta
            imin -= delta
            irange = rrange

        rmax += 0.05 * rrange
        rmin -= 0.05 * rrange
        imax += 0.05 * irange
        imin -= 0.05 * irange
        amax += 0.05 * arange
        amin -= 0.05 * arange

        # Info for overplotted cumulative histogram of log-ampls

        def getlogamps():
            for bysp in normalized.values():
                for samps, otherant, logamp in bysp.values():
                    yield logamp

        all_log_amps = np.concatenate(tuple(getlogamps()))
        all_log_amps.sort()
        all_log_amps_x = np.linspace(0.0, len(self.antnames), all_log_amps.size)
        all_log_amps_bounds = scoreatpercentile(all_log_amps, [2.5, 97.5])

        # Actually plot

        for antname in self.antnames:
            bysp = normalized.get(antname)
            if bysp is None:
                continue  # no data

            reim = om.RectPlot()
            reim.addKeyItem(antname)
            reim.addHLine(
                0,
                keyText=None,
                zheight=-2,
                dsn=0,
                lineStyle={"color": (0, 0, 0), "dashing": (2, 2)},
            )
            reim.addVLine(
                1,
                keyText=None,
                zheight=-2,
                dsn=0,
                lineStyle={"color": (0, 0, 0), "dashing": (2, 2)},
            )

            loga = om.RectPlot()
            loga.addHLine(
                0,
                keyText=None,
                zheight=-2,
                dsn=0,
                lineStyle={"color": (0, 0, 0), "dashing": (2, 2)},
            )
            loga.addXY(
                all_log_amps_x, all_log_amps, None, lineStyle={"color": (0, 0, 0)}
            )
            for a in all_log_amps_bounds:
                loga.addHLine(
                    a,
                    keyText=None,
                    zheight=-2,
                    dsn=0,
                    lineStyle={"color": (0, 0, 0), "dashing": (1, 3)},
                )

            for (spw, poln), (samps, otherant, logamp) in bysp.items():
                # Real/imag plot
                ms = om.stamps.MultiStamp("cnum", "shape", "tlines")
                ms.fixedsize = 4
                ms.fixedlinestyle = {"color": "muted"}

                cnum = np.zeros(samps.size, dtype=int)
                cnum.fill(spwseq[spw])

                shape = np.zeros_like(cnum)
                shape.fill(polnseq[poln])

                dp = om.rect.XYDataPainter(lines=False, pointStamp=ms, keyText=None)
                dp.setInts(cnum, shape, otherant + 1)
                dp.setFloats(samps.real, samps.imag)
                reim.add(dp)

                # Log-amplitudes plot
                ms = om.stamps.MultiStamp("cnum", "shape", "tlines")
                ms.fixedsize = 4
                ms.fixedlinestyle = {"color": "muted"}

                cnum = np.zeros(samps.size, dtype=int)
                cnum.fill(spwseq[spw])

                shape = np.zeros_like(cnum)
                shape.fill(polnseq[poln])

                dp = om.rect.XYDataPainter(lines=False, pointStamp=ms, keyText=None)
                dp.setInts(cnum, shape, otherant + 1)
                dp.setFloats(otherant, np.log10(np.abs(samps)))
                loga.add(dp)

            for spw in spws:
                for poln in polns:
                    s = EmulatedMultiStamp(polnseq[poln], spwseq[spw], 4)
                    reim.addKeyItem(ManualStampKeyPainter("spw#%d %s" % (spw, poln), s))

            reim.setLabels("Normalized real part", "Normalized imaginary part")
            reim.setBounds(rmin, rmax, imin, imax)
            # Unfortunately this is not compatible with VBox layout right now.
            # reim.fieldAspect = 1.

            loga.setLabels("Paired antenna number", "Log10 amplitude ratio")
            loga.setBounds(-0.5, len(self.antnames) + 0.5, amin, amax)

            vb = om.layout.VBox(2)
            vb[0] = reim
            vb[1] = loga
            vb.setWeight(0, 3)
            pager.send(vb)

        pager.done()


def gpdiagnostics_cli(argv):
    cli.check_usage(gpdiagnostics_doc, argv, usageifnoargs=True)
    cfg = Config().parse(argv[1:])
    util.logger(cfg.loglevel)

    tool = DiagnosticsTool(cfg)
    tool.load()
    tool.plot()
