# -*- mode: python; coding: utf-8 -*-
# Copyright 2014 Peter Williams <peter@newton.cx> and collaborators.
# Licensed under the MIT License.

"""pwkit.immodel - Analytical modeling of astronomical images.

This is derived from copl/pylib/bgfit.py and copl/bin/imsrcdebug. I keep on
wanting this code so I should put it somewhere more generic. Such as here.
Also, given the history, there are a lot more bells and whistles in the code
than the currently exposed UI really needs.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = str(
    """FitComponent Fitter GaussianComponent beam_volume
                  fit_one_source"""
).split()


import numpy as np

from . import ellipses, lmmin
from .astutil import *


NX = 4
X_DX, X_DY, X_LAT, X_LON = range(NX)


class FitComponent(object):
    npar = 0
    issource = False

    pofs = None
    setvalue = None
    setlimit = None

    def model(self, pars, x, y):
        pass

    def deriv(self, pars, x, jac):
        pass

    def lat_lon_bounds(self, pars, smallval):
        """return is (minlat, minlon, latwidth, lonwidth)"""
        return (None, None, None, None)

    def prep_problem(self):
        pass

    def postprocess(self, pars, perr, cov):
        pass


class GaussianComponent(FitComponent):
    npar = 6
    issource = True

    def model(self, pars, x, y):
        amp, slat, slon, p, q, th = pars
        lat = x[X_LAT]
        lon = x[X_LON]

        sn = np.sin(th)
        cs = np.cos(th)
        a = -0.5 * (cs * cs * p + sn * sn * q)
        b = sn * cs * (q - p)
        c = -0.5 * (sn * sn * p + cs * cs * q)

        dlat = lat - slat
        dlon = (lon - slon) * np.cos(lat)
        f = a * dlat**2 + b * dlat * dlon + c * dlon**2
        y += amp * np.exp(f)

    def deriv(self, pars, x, jac):
        # Need to compute the whole source function to get the
        # amplitude derivative.

        amp, slat, slon, p, q, th = pars
        lat = x[X_LAT]
        lon = x[X_LON]

        sn = np.sin(th)
        cs = np.cos(th)
        a = -0.5 * (cs * cs * p + sn * sn * q)
        b = sn * cs * (q - p)
        c = -0.5 * (sn * sn * p + cs * cs * q)
        dlat = lat - slat
        dlon = (lon - slon) * np.cos(lat)
        f = a * dlat**2 + b * dlat * dlon + c * dlon**2

        # Now, back to the Jacobian ...  I got this right on the first
        # try!

        jac[0] = np.exp(f)
        v = amp * jac[0]
        jac[1] = v * (-2 * a * dlat - b * dlon)
        jac[2] = v * (-2 * c * dlon - b * dlat) * np.cos(lat)
        jac[3] = v * (
            -0.5 * ((cs * dlat) ** 2 + (sn * dlon) ** 2) + cs * sn * dlat * dlon
        )
        jac[4] = v * (
            -0.5 * ((sn * dlat) ** 2 + (cs * dlon) ** 2) - cs * sn * dlat * dlon
        )
        jac[5] = (
            v
            * (p - q)
            * (cs * sn * (dlat**2 - dlon**2) - dlat * dlon * (cs**2 - sn**2))
        )

    def lat_lon_bounds(self, pars, smallval):
        amp, slat, slon, p, q, th = pars
        smaj = p**-0.5
        smin = q**-0.5
        s = np.sin(th)
        c = np.cos(th)

        siglat = np.sqrt((c * smaj) ** 2 + (s * smin) ** 2)
        siglon = np.sqrt((s * smaj) ** 2 + (c * smin) ** 2)

        if abs(amp) <= 90 * smallval:
            # 90 is about the cutoff for a 3 sigma attenuation
            n = 3
        else:
            n = np.sqrt(2 * np.log(abs(amp) / smallval))

        dlat = n * siglat
        dlon = n * siglon / np.cos(slat)
        return [slat - dlat, slon - dlon, 2 * dlat, 2 * dlon]

    def prep_problem(self):
        # this corresponds to a FWHM of ~135 degrees ...
        self.setlimit(3, lower=1)
        self.setlimit(4, lower=1)

    def postprocess(self, pars, perr, cov):
        self.f_pkflux = pars[0]
        self.e_pkflux = perr[0]
        self.f_dec = pars[1]
        self.e_dec = perr[1]
        self.f_ra = pars[2]
        self.e_ra = perr[2]
        self.f_rmajor = pars[3] ** -0.5 * S2F
        self.e_rmajor = 0.5 * self.f_rmajor**3 * perr[3] * S2F
        self.f_rminor = pars[4] ** -0.5 * S2F
        self.e_rminor = 0.5 * self.f_rminor**3 * perr[4] * S2F
        self.f_pa = orientcen(pars[5])
        self.e_pa = perr[5]

        # Uncertainty in the position as a 1-sigma ellipse
        if perr[1] == 0 and perr[2] == 0:
            self.eell_pos = (0, 0, 0)  # presume a fixed position
        else:
            s = ellipses.sigmascale(1)
            pmaj, pmin, ppa = ellipses.bivell(perr[1], perr[2], cov[1, 2])
            self.eell_pos = (pmaj * s, pmin * s, ppa)

        if self.f_rminor > self.f_rmajor:
            self.f_rminor, self.f_rmajor = self.f_rmajor, self.f_rminor
            self.e_rminor, self.e_rmajor = self.e_rmajor, self.e_rminor

            if self.f_pa > 0:
                self.f_pa -= halfpi
            else:
                self.f_pa += halfpi

    def postproc_total_flux(self, im):
        self.f_totflux = (
            self.f_pkflux * self.f_rmajor * self.f_rminor / (im.bmaj * im.bmin)
        )
        x = (
            (self.e_pkflux / self.f_pkflux) ** 2
            + (self.e_rmajor / self.f_rmajor) ** 2
            + (self.e_rminor / self.f_rminor) ** 2
        )
        self.e_totflux = abs(self.f_totflux) * np.sqrt(x)

    def deconvolve(self, im):
        return gaussian_deconvolve(
            self.f_rmajor, self.f_rminor, self.f_pa, im.bmaj, im.bmin, im.bpa
        )

    def setup(self, flux, lat, lon, maj, min, pa, fixpos=False, fixshape=False):
        self.setvalue(0, flux)
        self.setvalue(1, lat, fixed=fixpos)
        self.setvalue(2, lon, fixed=fixpos)
        self.setvalue(3, (maj * F2S) ** -2, fixed=fixshape)
        self.setvalue(4, (min * F2S) ** -2, fixed=fixshape)
        self.setvalue(5, pa, fixed=fixshape)

    def setup_point(self, flux, lat, lon, im, fixpos=False, fixshape=False):
        self.setup(
            flux, lat, lon, im.bmaj, im.bmin, im.bpa, fixpos=fixpos, fixshape=fixshape
        )


def beam_volume(im):
    delta = 1e-6
    latax, lonax = im._latax, im._lonax

    p = 0.5 * (np.asfarray(im.shape) - 1)
    w1 = im.toworld(p)
    p[latax] += delta
    w2 = im.toworld(p)
    latcell = (w2[latax] - w1[latax]) / delta
    p[latax] -= delta
    p[lonax] += delta
    w2 = im.toworld(p)
    loncell = (w2[lonax] - w1[lonax]) / delta * np.cos(w2[latax])
    bmrad2 = 2 * np.pi * im.bmaj * im.bmin / (8 * np.log(2))
    cellrad2 = latcell * loncell
    return np.abs(bmrad2 / cellrad2)


class Fitter(object):
    def __init__(self):
        self.npar = 0
        self.guess = np.zeros((0,))
        self.components = []
        self.prob = lmmin.Problem()

    def add(self, component):
        component.pofs = self.npar

        def setvalue(cidx, val, fixed=False):
            if cidx < 0 or cidx >= component.npar:
                raise ValueError("cidx")
            self.prob.p_value(cidx + component.pofs, val, fixed=fixed)
            self.guess[cidx + component.pofs] = val

        def setlimit(cidx, lower=-np.inf, upper=np.inf):
            if cidx < 0 or cidx >= component.npar:
                raise ValueError("cidx")
            self.prob.p_limit(cidx + component.pofs, lower, upper)

        component.setvalue = setvalue
        component.setlimit = setlimit

        newguess = np.zeros((self.npar + component.npar,))
        newguess[: self.npar] = self.guess
        self.guess = newguess

        self.components.append(component)
        self.npar += component.npar
        self.prob.set_npar(self.npar)
        return component

    def setup_problem(self, im, fullimdata, noise, smallvalfactor, stamphalfsize=None):
        if not len(self.components):
            raise RuntimeError("no components added to fitter")

        self.im = im
        self.fullimdata = fullimdata
        self.noise = noise

        # We're going to have to scale up our error estimates because noise is
        # heavily correlated between pixels. I don't pretend to know the
        # details but this is apparently the factor we need:

        self.imerrscale = np.sqrt(beam_volume(im))

        # Determine rough lat/lon bounds of emission. We have to be careful
        # because we're on a sphere and angles may wrap.

        wbounds = [None] * 4  # latmin, lonmin, latmax, lonmax

        for comp in self.components:
            p = self.guess[comp.pofs : comp.pofs + comp.npar]
            cbounds = comp.lat_lon_bounds(p, noise * smallvalfactor)

            for i in range(2):
                cb = cbounds[i]

                if cb is None:
                    continue

                if wbounds[i] is None:
                    wbounds[i] = cb
                    wbounds[i + 2] = cb + cbounds[i + 2]
                    continue

                if i == 0:  # latitude -- easier
                    wbounds[i] = min(wbounds[i], cb)
                    wbounds[i + 2] = max(wbounds[i + 2], cb + cbounds[i + 2])
                else:  # longitude -- worry about wraps
                    delta = angcen(cb - wbounds[i])
                    if delta < 0:
                        wbounds[i] += delta

                    delta = angcen(cb + cbounds[i + 2] - wbounds[i + 2])
                    if delta > 0:
                        wbounds[i + 2] += delta

        # could handle this in various ways instead of bailing
        if wbounds[0] is None:
            raise RuntimeError("no lat bounds")
        if wbounds[1] is None:
            raise RuntimeError("no lon bounds")

        # Now map these into a rectangle of pixels, with clipping and all of
        # that good stuff. TODO: if we don't have enough pixels to
        # successfully solve the problem, increase the bounds until we can.

        pbounds = [np.inf, -np.inf, np.inf, -np.inf]

        def adjust(lat, lon):
            py, px = im.topixel([lat, lon])
            px = int(np.floor(px))
            py = int(np.floor(py))
            pbounds[0] = min(pbounds[0], px)
            pbounds[1] = max(pbounds[1], px + 1)
            pbounds[2] = min(pbounds[2], py)
            pbounds[3] = max(pbounds[3], py + 1)

        adjust(wbounds[0], wbounds[1])
        adjust(wbounds[0], wbounds[3])
        adjust(wbounds[2], wbounds[1])
        adjust(wbounds[2], wbounds[3])

        if stamphalfsize is not None:
            # Hackishness for the 3-pixel parabolic fits a la
            # Bannister+ 2011.
            xmid = (pbounds[0] + pbounds[1]) // 2
            ymid = (pbounds[2] + pbounds[3]) // 2
            pbounds[:2] = xmid - stamphalfsize, xmid + stamphalfsize
            pbounds[2:] = ymid - stamphalfsize, ymid + stamphalfsize

        x0 = self.x0 = max(pbounds[0], 0)
        x1 = min(pbounds[1] + 1, im.shape[1])  # to Python range style
        y0 = self.y0 = max(pbounds[2], 0)
        y1 = min(pbounds[3] + 1, im.shape[0])

        if x0 >= im.shape[1] or x1 < 0 or y0 >= im.shape[0] or y1 < 0:
            raise RuntimeError("trying to fit a component outside the image")

        patchw = x1 - x0
        patchh = y1 - y0

        assert patchw > 0
        assert patchh > 0

        # Extract the postage stamp and set up our X values.

        data = self.data = self.fullimdata[y0:y1, x0:x1]
        x = np.empty((NX, patchh, patchw))
        x[X_DY], x[X_DX] = np.indices(data.shape)
        x[X_DX] -= 0.5 * (patchw - 1)
        x[X_DY] -= 0.5 * (patchh - 1)

        for i in range(patchh):
            for j in range(patchw):
                x[X_LAT, i, j], x[X_LON, i, j] = self.im.toworld([i + y0, j + x0])

        x = self.x = x.reshape((NX, patchh * patchw))

        # The functions.

        for c in self.components:
            c.prep_problem()

        def model(pars, outputs, sourcesonly=False):
            outputs.fill(0)

            for c in self.components:
                if sourcesonly and not c.issource:
                    continue

                p = pars[c.pofs : c.pofs + c.npar]
                c.model(p, x, outputs)

        def deriv(pars, jac):
            for c in self.components:
                p = pars[c.pofs : c.pofs + c.npar]
                subjac = jac[c.pofs : c.pofs + c.npar]
                c.deriv(p, x, subjac)

        self.model = model
        self.deriv = deriv
        self.prob.set_residual_func(
            data.flatten(), np.ones(data.size) / self.noise, model, deriv
        )
        return self

    def solve(self):
        self.soln = self.prob.solve(self.guess)
        return self

    def postprocess(self):
        pars = self.soln.params
        perr = self.soln.perror
        cov = self.soln.covar

        # Apply that correction for correlated pixels. (It seems like a good
        # idea to do it as far upstream as possible, which seems to be here.)

        perr *= self.imerrscale

        # Goodness-of-fit metrics

        self.rchisq = self.soln.fnorm / self.soln.ndof
        n = self.data.size
        k = n - self.soln.ndof
        self.aicc = self.soln.fnorm + 2.0 * k * n / (n - k - 1)
        self.residrms = np.sqrt(self.soln.fnorm / self.data.size) * self.noise

        for c in self.components:
            spar = pars[c.pofs : c.pofs + c.npar]
            serr = perr[c.pofs : c.pofs + c.npar]
            scov = cov[c.pofs : c.pofs + c.npar, c.pofs : c.pofs + c.npar]
            c.postprocess(spar, serr, scov)

        return self

    def subsources(self, imdata):
        # XXX not currently used.
        postage = np.empty(self.data.size)
        self.model(self.soln.params, postage, sourcesonly=True)
        h, w = self.data.shape
        x0, y0 = self.x0, self.y0
        imdata[y0 : y0 + h, x0 : x0 + w] -= postage.reshape((h, w))

    def display(self, run_main=True):
        from .ndshow_gtk3 import cycle

        arrays = [self.data[::-1,]]
        descs = ["Data"]

        model = np.empty(self.data.size)
        self.model(self.soln.params, model, sourcesonly=False)
        arrays.append(model.reshape(self.data.shape)[::-1,])
        descs.append("Model")

        arrays.append(arrays[0] - arrays[1])
        descs.append("Residual")

        cycle(arrays, descs, run_main=run_main, yflip=True)
        return self


def _guess_background_point_flux(im, imdata, xmid, ymid, patchhalfsize=16):
    imh, imw = imdata.shape

    x0 = max(xmid - patchhalfsize, 0)
    x1 = min(xmid + patchhalfsize + 1, imw)
    y0 = max(ymid - patchhalfsize, 0)
    y1 = min(ymid + patchhalfsize + 1, imh)

    d = imdata[y0:y1, x0:x1]
    dmed = np.median(d)
    dmax = np.max(d)
    return dmed, dmax - dmed


def fit_one_source(
    im,
    xmid,
    ymid,
    forcepoint=False,
    patchhalfsize=16,
    noise=1e-3,
    smallvalfactor=0.5,
    report_func=None,
    display=False,
):
    imdata = im.read()
    bgguess, ptguess = _guess_background_point_flux(
        im, imdata, xmid, ymid, patchhalfsize
    )
    lat, lon = im.toworld([ymid, xmid])

    fg = Fitter()
    fg.add(GaussianComponent()).setup_point(ptguess, lat, lon, im, fixshape=False)
    fg.setup_problem(im, imdata, noise, smallvalfactor)
    fg.solve().postprocess()

    dmaj, dmin, dpa, status = fg.components[-1].deconvolve(im)

    fp = Fitter()
    fp.add(GaussianComponent()).setup_point(ptguess, lat, lon, im, fixshape=True)
    fp.setup_problem(im, imdata, noise, smallvalfactor)
    fp.solve().postprocess()

    if forcepoint:
        f = fp
        kind = "point"
        reason = "forced"
    elif status != "ok":
        f = fp
        kind = "point"
        if fg.rchisq < 0.95 * fp.rchisq:
            reason = "couldnt_deconvolve"
        else:
            reason = "better_fit"
    elif fg.rchisq < 0.95 * fp.rchisq:
        f = fg
        kind = "gaussian"
        reason = "better_fit"
    else:
        f = fp
        kind = "point"
        reason = "better_fit"

    s = f.residrms / noise
    c = f.components[-1]
    c.postproc_total_flux(im)
    fy, fx = im.topixel([c.f_dec, c.f_ra])
    if im.pclat is not None:
        d = sphdist(c.f_dec, c.f_ra, im.pclat, im.pclon)

    if report_func is not None:
        report = report_func
    else:

        def report(key, fmt, value):
            print(key, "=", fmt % value, sep="")

    report("preferred_shape", "%s", kind)
    report("preferred_shape_reason", "%s", reason)
    report("rel_rchisq_point", "%.2f", fp.rchisq / f.rchisq)
    report("rel_rchisq_gauss", "%.2f", fg.rchisq / f.rchisq)
    report("resid_rms_mjy", "%.4f", f.residrms * 1e3)
    report("ndof", "%d", f.soln.ndof)
    report("ndata", "%d", f.data.size)
    report("nparam", "%d", f.data.size - f.soln.ndof)
    report("ra_sexg", "%s", fmthours(c.f_ra))
    report("dec_sexg", "%s", fmtdeglat(c.f_dec))
    report("x_pix", "%.1f", fx)
    report("y_pix", "%.1f", fy)

    if im.pclat is not None:
        report("pnt_ctr_distance", "%.3f", d * R2A)

    report("pos_err_major_arcsec", "%.3f", s * c.eell_pos[0] * R2A)
    report("pos_err_minor_arcsec", "%.3f", s * c.eell_pos[1] * R2A)
    report("pos_err_pa_deg", "%.1f", c.eell_pos[2] * R2D)
    report("tot_flux_mjy", "%.4f", c.f_totflux * 1e3)
    report("tot_flux_err_mjy", "%.4f", s * c.e_totflux * 1e3)
    report("tot_flux_err_frac", "%.3f", s * c.e_totflux / c.f_totflux)

    if kind == "gaussian":
        report("shape_pa_err_deg", "%.1f", s * c.e_pa * R2D)
        report("dshape_major_arcsec", "%.3f", dmaj * R2A)
        report("dshape_minor_arcsec", "%.3f", dmin * R2A)
        report("dshape_pa_deg", "%.2f", dpa * R2D)
        report(
            "dshape_area_ratio", "%.3f", c.f_rmajor * c.f_rminor / (im.bmaj * im.bmin)
        )

    if display:
        f.display()
