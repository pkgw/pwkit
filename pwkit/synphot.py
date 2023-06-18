# -*- mode: python; coding: utf-8 -*-
# Copyright 2014-2022 Peter Williams <peter@newton.cx> and collaborators.
# Licensed under the MIT License.

"""Synthetic photometry and database of instrumental bandpasses.

The basic structure is that we have a registry of bandpass info. You can use
it to create Bandpass objects that can perform various calculations,
especially the computation of synthetic photometry given a spectral model.
Some key attributes of each bandpass are pre-computed so that certain
operations can be done without needing to load the actual bandpass profile
(though so far none of these profiles are very large at all).

The bandpass definitions built into this module are:

* 2MASS (JHK)
* Bessell (UBVRI)
* GALEX (NUV, FUV)
* LMIRCam on LBT
* MEarth
* Mauna Kea Observatory (MKO) (JHKLM)
* SDSS (u' g' r' i' z')
* Swift (UVW1)
* WISE (1234)

**Classes:**

.. autosummary::
   AlreadyDefinedError
   Bandpass
   NotDefinedError
   Registry

**Functions:**

.. autosummary::
   get_std_registry

Various internal utilities may be useful for reference but are not documented here.

**Variables:**

.. autosummary::
   builtin_registrars

Example
-------

::

  from pwkit import synphot as ps, cgs as pc, msmt as pm
  reg = ps.get_std_registry()
  print(reg.telescopes()) # list known telescopes
  print(reg.bands('2MASS')) # list known 2MASS bands
  bp = reg.get('2MASS', 'Ks')
  mag = 12.83
  mjy = pm.repval(bp.mag_to_fnu(mag) * pc.jypercgs * 1e3)
  print('%.2f mag is %.2f mjy in 2MASS/Ks' % (mag, mjy))


Conventions
-----------

It is very important to maintain consistent conventions throughout.

Wavelengths are measured in angstroms. Flux densities are either
per-wavelength (f_λ, "flam") or per-frequency (f_ν, "fnu"). These are measured
in units of erg/s/cm²/Å and erg/s/cm²/Hz, respectively. Janskys can be
converted to f_ν by multiplying by cgs.cgsperjy. f_ν's and f_λ's can be
interconverted for a given filter if you know its "pivot wavelength". Some of
the routines below show how to calculate this and do the conversion. "AB
magnitudes" can be directly converted to Janskys and, thus, f_ν's.

Filter bandpasses can be expressed in two conventions: either "equal-energy"
(EE) or "quantum-efficiency" (QE). The former gives the response per unit
energy across the band, while the latter gives the response per photon. The EE
convention can be integrated directly against a model spectrum, so we store
all bandpasses internally in this convention. CCDs are photon-counting devices
and so their response curves are generally expressed in the QE convention.
Interconversion is easy: EE = QE * λ.

We don't expect any particular normalization of bandpass response curves.

The "width" of a bandpass is not a well-defined quantity, but is often needed
for display purposes or approximate calculations. We use the locations of the
half-maximum points (in the EE convention) to define the band edges.

This module requires Scipy and Pandas. It doesn't reeeeallllly need Pandas but
it's convenient.

References
----------

Casagrande & VandenBerg (2014; arxiv:1407.6095) has a lot of good stuff; see
also references therein.

References for specific bandpasses are given in their implementation
docstrings.

"""
from __future__ import absolute_import, division, print_function

__all__ = """
AlreadyDefinedError
Bandpass
NotDefinedError
Registry
builtin_registrars
get_std_registry""".split()

import numpy as np, pandas as pd, pkg_resources

from . import Holder, PKError, cgs, msmt


# Data loading


def bandpass_data_stream(name):
    return pkg_resources.resource_stream("pwkit", "data/bandpasses/" + name)


def bandpass_data_frame(name, colnames):
    a = np.loadtxt(bandpass_data_stream(name)).T
    mapped = dict((c, a[i]) for i, c in enumerate(colnames.split()))
    return pd.DataFrame(mapped)


def bandpass_data_fits(name):
    from astropy.io.fits import open

    return open(bandpass_data_stream(name))


# Simple, careful conversions


def fnu_cgs_to_flam_ang(fnu_cgs, pivot_angstrom):
    """erg/s/cm²/Hz → erg/s/cm²/Å"""
    return 1e8 * cgs.c / pivot_angstrom**2 * fnu_cgs


def flam_ang_to_fnu_cgs(flam_ang, pivot_angstrom):
    """erg/s/cm²/Å → erg/s/cm²/Hz"""
    return 1e-8 / cgs.c * pivot_angstrom**2 * flam_ang


def abmag_to_fnu_cgs(abmag):
    """Convert an AB magnitude to f_ν in erg/s/cm²/Hz."""
    return cgs.cgsperjy * 3631.0 * 10 ** (-0.4 * abmag)


def abmag_to_flam_ang(abmag, pivot_angstrom):
    """Convert an AB magnitude to f_λ in erg/s/cm²/Å. AB magnitudes are f_ν
    quantities, so a pivot wavelength is needed.

    """
    return fnu_cgs_to_flam_ang(abmag_to_fnu_cgs(abmag), pivot_angstrom)


def ghz_to_ang(ghz):
    """Convert a photon frequency in GHz to its wavelength in Ångström."""
    return 0.1 * cgs.c / ghz


def flat_ee_bandpass_pivot_wavelength(wavelen1, wavelen2):
    """Compute the pivot wavelength of a bandpass that's flat in equal-energy
    terms. It turns out to be their harmonic mean.

    """
    return np.sqrt(wavelen1 * wavelen2)


def pivot_wavelength_ee(bpass):
    """Compute pivot wavelength assuming equal-energy convention.

    `bpass` should have two properties, `resp` and `wlen`. The units of `wlen`
    can be anything, and `resp` need not be normalized in any particular way.

    """
    from scipy.integrate import simps

    return np.sqrt(
        simps(bpass.resp, bpass.wlen) / simps(bpass.resp / bpass.wlen**2, bpass.wlen)
    )


def pivot_wavelength_qe(bpass):
    """Compute pivot wavelength assuming quantum-efficiency convention. Note that
    this is NOT what we generally use in this module.

    `bpass` should have two properties, `resp` and `wlen`. The units of `wlen`
    can be anything, and `resp` need not be normalized in any particular way.

    """
    from scipy.integrate import simps

    return np.sqrt(
        simps(bpass.resp * bpass.wlen, bpass.wlen)
        / simps(bpass.resp / bpass.wlen, bpass.wlen)
    )


def interpolated_halfmax_points(x, y):
    """Given a curve y(x), find the x coordinates of points that have half the
    value of max(y), using linear interpolation. We're assuming that y(x) has
    a bandpass-ish shape, i.e., a single maximum and a drop to zero as we go
    to the edges of the function's domain. We also assume that x is sorted
    increasingly.

    """
    from scipy.interpolate import interp1d
    from scipy.optimize import fmin

    x = np.asarray(x)
    y = np.asarray(y)
    halfmax = 0.5 * y.max()

    # Guess from the actual samples.

    delta = y - halfmax
    guess1 = 0
    while delta[guess1] < 0:
        guess1 += 1
    guess2 = y.size - 1
    while delta[guess2] < 0:
        guess2 -= 1

    # Interpolate for fanciness.

    terp = interp1d(x, y, kind="linear", bounds_error=False, fill_value=0.0)
    x1 = fmin(lambda x: (terp(x) - halfmax) ** 2, x[guess1], disp=False)
    x2 = fmin(lambda x: (terp(x) - halfmax) ** 2, x[guess2], disp=False)

    x1 = np.asarray(x1).item()
    x2 = np.asarray(x2).item()

    if x1 == x2:
        raise PKError("halfmax finding failed")

    if x1 > x2:
        x1, x2 = x2, x1

    return x1, x2


# Organized storage of the bandpass info. This way we're extensible (ooh aah)
# and we don't have to run a bunch of code on module import.


class AlreadyDefinedError(PKError):
    """Raised when re-registering bandpass info."""


class NotDefinedError(PKError):
    """Raised when needed bandpass info is unavailable."""


class Bandpass(object):
    """Computations regarding a particular filter bandpass.

    The underlying bandpass shape is assumed to be sampled at discrete points.
    It is stored in ``_data`` and loaded on-demand. The object is a Pandas
    DataFrame containing at least the columns ``wlen`` and ``resp``. The
    former holds the wavelengths of the sample points, in Ångström and in
    ascending order. The latter gives the response curve in the EE convention.
    No particular normalization is assumed. Other columns may be present but
    are not used generically.

    """

    _data = None
    native_flux_kind = "none"
    "Which kind of flux this bandpass is calibrated to: 'flam', 'fnu', or 'none'."

    # These are set by the registry on construction:
    registry = None
    "This object's parent Registry instance."

    telescope = None
    "The name of this bandpass' associated telescope."

    band = None
    "The name of this bandpass' associated band."

    def _ensure_data(self):
        if self._data is None:
            self._data = self._load_data(self.band)
        return self._data

    def calc_pivot_wavelength(self):
        """Compute and return the bandpass' pivot wavelength.

        This value is computed directly from the bandpass data, not looked up
        in the Registry. Most of the values in the Registry were in fact
        derived from this function originally.

        """
        d = self._ensure_data()
        return pivot_wavelength_ee(d)

    def pivot_wavelength(self):
        """Get the bandpass' pivot wavelength.

        Unlike calc_pivot_wavelength(), this function will use a cached
        value if available.

        """
        wl = self.registry._pivot_wavelengths.get((self.telescope, self.band))
        if wl is not None:
            return wl

        wl = self.calc_pivot_wavelength()
        self.registry.register_pivot_wavelength(self.telescope, self.band, wl)
        return wl

    def calc_halfmax_points(self):
        """Calculate the wavelengths of the filter half-maximum values."""
        d = self._ensure_data()
        return interpolated_halfmax_points(d.wlen, d.resp)

    def halfmax_points(self):
        """Get the bandpass' half-maximum wavelengths. These can be used to
        compute a representative bandwidth, or for display purposes.

        Unlike calc_halfmax_points(), this function will use a cached value if
        available.

        """
        t = self.registry._halfmaxes.get((self.telescope, self.band))
        if t is not None:
            return t

        t = self.calc_halfmax_points()
        self.registry.register_halfmaxes(self.telescope, self.band, t[0], t[1])
        return t

    def mag_to_fnu(self, mag):
        """Convert a magnitude in this band to a f_ν flux density.

        It is assumed that the magnitude has been computed in the appropriate
        photometric system. The definition of "appropriate" will vary from
        case to case.

        """
        if self.native_flux_kind == "flam":
            return flam_ang_to_fnu_cgs(self.mag_to_flam(mag), self.pivot_wavelength())
        raise PKError(
            "dont't know how to get f_ν from mag for bandpass %s/%s",
            self.telescope,
            self.band,
        )

    def mag_to_flam(self, mag):
        """Convert a magnitude in this band to a f_λ flux density.

        It is assumed that the magnitude has been computed in the appropriate
        photometric system. The definition of "appropriate" will vary from
        case to case.

        """
        if self.native_flux_kind == "fnu":
            return fnu_cgs_to_flam_ang(self.mag_to_fnu(mag), self.pivot_wavelength())
        raise PKError(
            "dont't know how to get f_λ from mag for bandpass %s/%s",
            self.telescope,
            self.band,
        )

    def jy_to_flam(self, jy):
        """Convert a f_ν flux density measured in Janskys to a f_λ flux density.

        This conversion is bandpass-dependent because it depends on the pivot
        wavelength of the bandpass used to measure the flux density.

        """
        return fnu_cgs_to_flam_ang(cgs.cgsperjy * jy, self.pivot_wavelength())

    def synphot(self, wlen, flam):
        """`wlen` and `flam` give a tabulated model spectrum in wavelength and f_λ
        units. We interpolate linearly over both the model and the bandpass
        since they're both discretely sampled.

        Note that quadratic interpolation is both much slower and can blow up
        fatally in some cases. The latter issue might have to do with really large
        X values that aren't zero-centered, maybe?

        I used to use the quadrature integrator, but Romberg doesn't issue
        complaints the way quadrature did. I should probably acquire some idea
        about what's going on under the hood.

        """
        from scipy.interpolate import interp1d
        from scipy.integrate import romberg

        d = self._ensure_data()

        mflam = interp1d(wlen, flam, kind="linear", bounds_error=False, fill_value=0)

        mresp = interp1d(
            d.wlen, d.resp, kind="linear", bounds_error=False, fill_value=0
        )

        bmin = d.wlen.min()
        bmax = d.wlen.max()

        numer = romberg(lambda x: mresp(x) * mflam(x), bmin, bmax, divmax=20)
        denom = romberg(lambda x: mresp(x), bmin, bmax, divmax=20)
        return numer / denom

    def blackbody(self, T):
        """Calculate the contribution of a blackbody through this filter. *T* is the
        blackbody temperature in Kelvin. Returns a band-averaged spectrum in
        f_λ units.

        We use the composite Simpson's rule to integrate over the points at
        which the filter response is sampled. Note that this is a different
        technique than used by `synphot`, and so may give slightly different
        answers than that function.

        """
        from scipy.integrate import simps

        d = self._ensure_data()

        # factor of pi is going from specific intensity (sr^-1) to unidirectional
        # inner factor of 1e-8 is Å to cm
        # outer factor of 1e-8 is f_λ in cm^-1 to f_λ in Å^-1
        from .cgs import blambda

        numer_samples = d.resp * np.pi * blambda(d.wlen * 1e-8, T) * 1e-8

        numer = simps(numer_samples, d.wlen)
        denom = simps(d.resp, d.wlen)
        return numer / denom


class Registry(object):
    """A registry of known bandpass properties."""

    def __init__(self):
        self._pivot_wavelengths = {}
        self._halfmaxes = {}
        self._bpass_classes = {}
        self._seen_bands = {}

    def _note(self, telescope, band):
        q = self._seen_bands.setdefault(telescope, set())
        if band is not None:
            q.add(band)

    def telescopes(self):
        """Return a list of telescopes known to this registry."""
        return self._seen_bands.keys()

    def bands(self, telescope):
        """Return a list of bands associated with the specified telescope."""
        q = self._seen_bands.get(telescope)
        if q is None:
            return []
        return list(q)

    def register_pivot_wavelength(self, telescope, band, wlen):
        """Register precomputed pivot wavelengths."""
        if (telescope, band) in self._pivot_wavelengths:
            raise AlreadyDefinedError(
                "pivot wavelength for %s/%s already " "defined", telescope, band
            )
        self._note(telescope, band)
        self._pivot_wavelengths[telescope, band] = wlen
        return self

    def register_halfmaxes(self, telescope, band, lower, upper):
        """Register precomputed half-max points."""

        if (telescope, band) in self._halfmaxes:
            raise AlreadyDefinedError(
                "half-max points for %s/%s already " "defined", telescope, band
            )
        self._note(telescope, band)
        self._halfmaxes[telescope, band] = (lower, upper)
        return self

    def register_bpass(self, telescope, klass):
        """Register a Bandpass class."""

        if telescope in self._bpass_classes:
            raise AlreadyDefinedError(
                "bandpass class for %s already " "defined", telescope
            )
        self._note(telescope, None)
        self._bpass_classes[telescope] = klass
        return self

    def get(self, telescope, band):
        """Get a Bandpass object for a known telescope and filter."""

        klass = self._bpass_classes.get(telescope)
        if klass is None:
            raise NotDefinedError("bandpass data for %s not defined", telescope)

        bp = klass()
        bp.registry = self
        bp.telescope = telescope
        bp.band = band
        return bp


builtin_registrars = {}
"Hashtable of functions to register the builtin telescopes."


def get_std_registry():
    """Get a Registry object pre-filled with information for standard
    telescopes.

    """
    reg = Registry()
    for fn in builtin_registrars.values():
        fn(reg)
    return reg


# Now, builtin information for a variety of telescopes. Document these
# aggressively! I have these in alphabetical order. 2MASS is first.


class TwomassBandpass(Bandpass):
    native_flux_kind = "flam"

    def _load_data(self, band):
        df = bandpass_data_frame("filter_2mass_" + band + ".dat", "wlen resp")
        df.wlen *= 1e4  # micron to Angstrom
        return df

    _zeropoints = {
        # 2MASS Explanatory Supplement (VI.4.a) and Cohen+ 2003.
        # I've converted W/cm²/μm to erg/s/cm²/Å (factor of 1e3).
        "J": msmt.Uval.from_norm(3.129e-10, 5.464e-12),
        "H": msmt.Uval.from_norm(1.133e-10, 2.212e-12),
        "Ks": msmt.Uval.from_norm(4.283e-11, 8.053e-13),
    }

    def mag_to_flam(self, mag):
        return self._zeropoints[self.band] * 10 ** (-0.4 * mag)


def register_2mass(reg):
    reg.register_bpass("2MASS", TwomassBandpass)

    # Computed myself from the filter response curves.
    reg.register_pivot_wavelength("2MASS", "J", 12371.0)
    reg.register_pivot_wavelength("2MASS", "H", 16457.0)
    reg.register_pivot_wavelength("2MASS", "Ks", 21603.0)

    # Computed from filter responses using interpolator.
    reg.register_halfmaxes("2MASS", "J", 11316.0, 13465.0)
    reg.register_halfmaxes("2MASS", "H", 15182.0, 17792.0)
    reg.register_halfmaxes("2MASS", "Ks", 20242.0, 23026.0)


builtin_registrars["2MASS"] = register_2mass


# Standard Bessell filters reproducing the Johnson/Cousins UBVRI photometric
# system, using Blanton & Roweis (2007) AB corrections to get a flux density
# scale. We don't support U since it's a bit funky; see Bessell (1990).


class BessellBandpass(Bandpass):
    native_flux_kind = "fnu"

    def _load_data(self, band):
        """Bessell (1990) tries to determine standard filter responses that reproduce
        the Johnson/Cousins UBVRI photometric systems. Things are inherently
        imprecise because of the subtle differences between different workers'
        instruments and conventions, so it's not worth getting too worked up
        over precision.

        """
        return bandpass_data_frame("filter_bessell_" + band + ".dat", "wlen resp")

    _ab_corrections = {
        # Entries are m_AB - m_Vega. Data are from Blanton & Roweis (2007). We
        # skip ugriz and JHK_s since SDSS/2MASS-specific works should give
        # as-good or better results, I'd hope.
        "U": 0.79,
        "B": -0.09,
        "V": 0.02,
        "R": 0.21,
        "I": 0.45,
    }

    def mag_to_fnu(self, mag):
        """Convert a magnitude in the Johnson-Cousins system to a flux density. This
        is inherently not-so-precise since the actual conversion depends on
        the spectrum of the target and the actual passband of the filter used
        to make the observation, and a J-C magnitude is usually derived from
        instrumental magnitudes via some ad-hoc-ish transformation. But the
        following will be about right.

        """
        return abmag_to_fnu_cgs(mag + self._ab_corrections[self.band])


def register_bessell(reg):
    reg.register_bpass("Bessell", BessellBandpass)

    # I computed these myself from the per-energy response curves.
    reg.register_pivot_wavelength("Bessell", "B", 4370.0)
    reg.register_pivot_wavelength("Bessell", "V", 5478.0)
    reg.register_pivot_wavelength("Bessell", "R", 6496.0)
    reg.register_pivot_wavelength("Bessell", "I", 8020.0)

    # Ditto.
    reg.register_halfmaxes("Bessell", "B", 3885.0, 4832.0)
    reg.register_halfmaxes("Bessell", "V", 5013.0, 5865.0)
    reg.register_halfmaxes("Bessell", "R", 5653.0, 7220.0)
    reg.register_halfmaxes("Bessell", "I", 7283.0, 8826.0)


builtin_registrars["Bessell"] = register_bessell


# GALEX


class GalexBandpass(Bandpass):
    # TODO: there are GALEX magnitudes, but the data products give flux
    # densities directly, so I haven't bothered to look up the conversions.
    native_flux_kind = "none"

    def _load_data(self, band):
        """From Morrissey+ 2005, with the actual data coming from
        http://www.astro.caltech.edu/~capak/filters/. According to the latter,
        these are in QE units and thus need to be multiplied by the wavelength
        when integrating per-energy.

        """
        # `band` should be 'nuv' or 'fuv'
        df = bandpass_data_frame("filter_galex_" + band + ".dat", "wlen resp")
        df.resp *= df.wlen  # QE -> EE response convention.
        return df


def register_galex(reg):
    reg.register_bpass("GALEX", GalexBandpass)

    # I computed these myself from the per-energy response curves.
    reg.register_pivot_wavelength("GALEX", "nuv", 2305.0)
    reg.register_pivot_wavelength("GALEX", "fuv", 1537.0)

    # Ditto.
    reg.register_halfmaxes("GALEX", "nuv", 1956.0, 2746.0)
    reg.register_halfmaxes("GALEX", "fuv", 1415.0, 1646.0)


builtin_registrars["GALEX"] = register_galex


# LMIRCam on the LBT.


class LmircamBandpass(Bandpass):
    native_flux_kind = "fnu"

    def _load_data(self, band):
        """Filter responses for LBT's LMIRCam. Emailed to me privately by Jarron
        Leisenring on 2014 May 8.

        """
        # `band` should be 'L'.
        df = bandpass_data_frame("filter_lbt_lmircam_" + band + ".dat", "wlen resp")
        df.wlen *= 1e4  # micron to Angstrom
        df.resp *= df.wlen  # QE to equal-energy response.
        return df

    def mag_to_fnu(self, mag):
        """Compute F_ν for LBT LMIRCam data. This was for a one-off thing and I don't
        know if there's a reliable calibration of the photometric system to
        flux densities. It should be on an MKO-ish system, but who knows. I
        added this function to use the different pivot wavelength of the
        LMIRCam L filter, which is described as similar to, but not quite the
        same, as the MKO L'.

        """
        return (
            cgs.cgsperjy * MkoBandpass._zeropoints[self.band + "p"] * 10 ** (-0.4 * mag)
        )


def register_lbt(reg):
    # Numbers calculated manually.
    reg.register_bpass("LBT/LMIRCam", LmircamBandpass)
    reg.register_pivot_wavelength("LBT/LMIRCam", "L", 36696.0)
    reg.register_halfmaxes("LBT/LMIRCam", "L", 34142.0, 39947.0)


builtin_registrars["LBT"] = register_lbt


# MEarth. No absolute flux calibration available.


class MearthBandpass(Bandpass):
    native_flux_kind = "none"

    def _load_data(self, band):
        """Filter response the MEarth camera. I currently only have the CCD+RG715
        system, not the interference-filter setup that was tried briefly. The
        docs say that the filter responses are somewhat different before and
        after the interference-filter experiment, but I don't think the
        information we have is sensitive to changes on those levels. The data
        were fundamentally made by reading points off of data sheets so
        they're not going to be the most accurate.

        I computed the filter response file myself by multiplying spline
        approximations to the CCD and RG715 response curves emailed to me by
        Jonathan Irwin on 2014 Jul 11. I should write up something explaining
        what I did for posterity/reproducibility. AFAIK the response curves
        aren't published anywhere besides a hard-to-read plot in Nutzman &
        Charbonneau (2008).

        """
        df = bandpass_data_frame("filter_mearth_" + band + ".dat", "wlen resp")
        df.resp *= df.wlen  # QE to equal-energy response.
        return df


def register_mearth(reg):
    reg.register_bpass("MEarth", MearthBandpass)
    reg.register_pivot_wavelength("MEarth", "ccd715", 8286.0)
    reg.register_halfmaxes("MEarth", "ccd715", 7148.0, 9360.0)


builtin_registrars["MEarth"] = register_mearth


# The Mauna Kea Observatory (MKO) IR filter system.


class MkoBandpass(Bandpass):
    native_flux_kind = "fnu"

    def _load_data(self, band):
        """Filter responses for MKO NIR filters as specified in Tokunaga+ 2002 (see
        also Tokunaga+ 2005). I downloaded the L' profile from
        http://irtfweb.ifa.hawaii.edu/~nsfcam/hist/filters.2006.html.

        Pivot wavelengths from Tokunaga+ 2005 (Table 2) confirm that the
        profile is in QE convention, although my calculation of the pivot
        wavelength for L' is actually closer if I assume otherwise. M' and K_s
        are substantially better in QE convention, though, and based on the
        paper and nomenclature it seems more appropriate.

        """
        # `band` should be 'Lp'.
        df = bandpass_data_frame("filter_mko_" + band + ".dat", "wlen resp")
        # Put in increasing wavelength order:
        df = df[::-1]
        df.index = np.arange(df.shape[0])
        df.wlen *= 1e4  # micron to Angstrom
        df.resp *= df.wlen  # QE to equal-energy response.
        return df

    _zeropoints = {
        # From Tokunaga+ (2005), sort of. They list Vega flux densities for
        # different MKO filters, and note that IR magnitude systems are
        # usually defined with Vega = 0.0 or Vega = ~0.03 mag. We need to
        # actually know the photometric system in use to really accurately set
        # a zeropoint. But, assuming that people haven't done anything
        # ridiculously strange, setting Vega=0 will be pretty close.
        #
        # Values are in Jy.
        "J": 1560.0,
        "H": 1040.0,
        "Kp": 686.0,
        "Ks": 670.0,
        "K": 645.0,
        "Lp": 249.0,
        "Mp": 163.0,
    }

    def mag_to_fnu(self, mag):
        """Compute F_ν for an MKO IR filter band. There are some problems here since
        "MKO" is filters, not a photometric system, but people try to make
        Vega = 0.

        """
        return cgs.cgsperjy * self._zeropoints[self.band] * 10 ** (-0.4 * mag)


def register_mko(reg):
    reg.register_bpass("MKO", MkoBandpass)
    # From Tokunaga+ (2005), since my calculation is a fair bit different.
    reg.register_pivot_wavelength("MKO", "Lp", 37520.0)
    # Mine?
    reg.register_halfmaxes("MKO", "Lp", 34276.0, 41228.0)


builtin_registrars["MKO"] = register_mko


# Sloan Digital Sky Survey primed photometric system. We fake things a bit and
# use the bandpasses for the unprimed filters on the main survey telescope.


class SdssBandpass(Bandpass):
    native_flux_kind = "fnu"

    def _load_data(self, band):
        """Filter responses for SDSS. Data table from
        https://www.sdss3.org/binaries/filter_curves.fits, as linked from
        https://www.sdss3.org/instruments/camera.php#Filters. SHA1 hash of the
        file is d3f638c41e21489ba7d6dbe7bb8217d938f21184. "Determined by Jim
        Gunn in June 2001." Doi+ 2010 have updated estimates but these are
        per-column in the SDSS camera, which we don't care about.

        Note that these are for the main SDSS 2.5m telescope. Magnitudes in
        the primed SDSS system were determined on the "photometric telescope",
        and the whole reason for the existence of both primed and unprimed
        ugriz systems is that the two have filters with slightly different
        behavior. My current application involves an entirely different
        telescope emulating the primed SDSS photometric system, and their
        precise system response is neither going to be ultra-precisely
        characterized nor exactly equal to either of the SDSS systems. These
        responses will be good enough, though.

        Wavelengths are in Angstrom. Based on the pivot wavelengths listed in
        http://www.astro.ljmu.ac.uk/~ikb/research/mags-fluxes/, the data table
        stores QE responses, so we have to convert them to equal-energy
        responses. Responses both including and excluding the atmosphere are
        included; I use the former.

        """
        h = bandpass_data_fits("sdss3_filter_responses.fits")
        section = "ugriz".index(band[0]) + 1
        d = h[section].data
        if d.wavelength.dtype.isnative:
            df = pd.DataFrame({"wlen": d.wavelength, "resp": d.respt})
        else:
            df = pd.DataFrame(
                {
                    "wlen": d.wavelength.byteswap(True).newbyteorder(),
                    "resp": d.respt.byteswap(True).newbyteorder(),
                }
            )
        df.resp *= df.wlen  # QE to equal-energy response.
        return df

    def mag_to_fnu(self, mag):
        """SDSS *primed* magnitudes to F_ν. The primed magnitudes are the "USNO"
        standard-star system defined in Smith+ (2002AJ....123.2121S) and
        Fukugita+ (1996AJ....111.1748F). This system is anchored to the AB
        magnitude system, and as far as I can tell it is not known to have
        measurable offsets from that system. (As of DR10, the *unprimed* SDSS
        system is known to have small offsets from AB, but I do not believe
        that that necessarily has implications for u'g'r'i'z'.)

        However, as far as I can tell the filter responses of the USNO
        telescope are not published -- only those of the main SDSS 2.5m
        telescope. The whole reason for the existence of both the primed and
        unprimed ugriz systems is that their responses do not quite match. For
        my current application, which involves a completely different
        telescope anyway, the difference shouldn't matter.

        """
        # `band` should be 'up', 'gp', 'rp', 'ip', or 'zp'.
        if len(band) != 2 or band[1] != "p":
            raise ValueError("band: " + band)
        return abmag_to_fnu_cgs(mag)


def register_sdss(reg):
    reg.register_bpass("SDSS", SdssBandpass)

    # I computed these myself.
    reg.register_pivot_wavelength("SDSS", "up", 3557.0)
    reg.register_pivot_wavelength("SDSS", "gp", 4702.0)
    reg.register_pivot_wavelength("SDSS", "rp", 6176.0)
    reg.register_pivot_wavelength("SDSS", "ip", 7490.0)
    reg.register_pivot_wavelength("SDSS", "zp", 8947.0)

    # Ditto.
    reg.register_halfmaxes("SDSS", "up", 3295.0, 3861.0)
    reg.register_halfmaxes("SDSS", "gp", 4160.0, 5335.0)
    reg.register_halfmaxes("SDSS", "rp", 5622.0, 6753.0)
    reg.register_halfmaxes("SDSS", "ip", 6917.0, 8171.0)
    reg.register_halfmaxes("SDSS", "zp", 8291.0, 9290.0)


builtin_registrars["SDSS"] = register_sdss


# Swift.


class SwiftUvotBandpass(Bandpass):
    # Swift routines output flux densities automatically so I haven't bothered
    # to look up the conversion from their magnitude system.
    native_flux_kind = "none"

    _band_map = {"UVW1": "uw1"}

    def _load_data(self, band):
        """In-flight effective areas for the Swift UVOT, as obtained from the CALDB.
        See Breeveld+ 2011. XXX: confirm that these are equal-energy, not
        quantum-efficiency.

        """
        d = bandpass_data_fits("sw" + self._band_map[band] + "_20041120v106.arf")[
            1
        ].data

        # note:
        #   data.WAVE_MIN[i] < data.WAVE_MIN[i+1], but
        #   data.WAVE_MIN[i] > data.WAVE_MAX[i] (!)
        #   data.WAVE_MIN[i] = data.WAVE_MAX[i+1] (!)

        wmid = 0.5 * (d.WAVE_MIN + d.WAVE_MAX)  # in Ångström
        df = pd.DataFrame(
            {"wlen": wmid, "resp": d.SPECRESP, "wlo": d.WAVE_MAX, "whi": d.WAVE_MIN}
        )
        return df


def register_swift(reg):
    reg.register_bpass("Swift/UVOT", SwiftUvotBandpass)
    # Computed manually from Breeveld+2011 response.
    reg.register_pivot_wavelength("Swift/UVOT", "UVW1", 2517.0)
    reg.register_halfmaxes("Swift/UVOT", "UVW1", 2278.0, 2931.0)


builtin_registrars["Swift"] = register_swift


# WISE


class WiseBandpass(Bandpass):
    native_flux_kind = "fnu"

    _filter_subsets = {
        # The WISE filter tables are all on a common grid, which means that
        # some of them have responses that are largely essentially zero. We
        # manually clip the arrays using the numbers below.
        1: (10, 150),
        2: (140, 290),
        3: (440, 1520),
        4: (1640, 2550),
    }

    def _load_data(self, band):
        """From the WISE All-Sky Explanatory Supplement, IV.4.h.i.1, and Jarrett+
        2011. These are relative response per erg and so can be integrated
        directly against F_nu spectra. Wavelengths are in micron,
        uncertainties are in parts per thousand.

        """
        # `band` should be 1, 2, 3, or 4.
        df = bandpass_data_frame(
            "filter_wise_" + str(band) + ".dat", "wlen resp uncert"
        )
        df.wlen *= 1e4  # micron to Angstrom
        df.uncert *= df.resp / 1000.0  # parts per thou. to absolute values.
        lo, hi = self._filter_subsets[band]
        df = df[lo:hi]  # clip zero parts of response.
        return df

    _zeropoints = {
        # WISE Explanatory Supplement: IV.4.h.i.1; units are Jy. Color
        # corrections are necessary for sources with unusual spectra.
        1: 309.540,
        2: 171.787,
        3: 31.674,
        4: 8.363,
    }

    def mag_to_fnu(self, mag):
        return cgs.cgsperjy * self._zeropoints[self.band] * 10 ** (-0.4 * mag)


def register_wise(reg):
    reg.register_bpass("WISE", WiseBandpass)

    # I computed these myself from the per-energy response curves.
    reg.register_pivot_wavelength("WISE", 1, 33682.0)
    reg.register_pivot_wavelength("WISE", 2, 46179.0)
    reg.register_pivot_wavelength("WISE", 3, 120731.0)
    reg.register_pivot_wavelength("WISE", 4, 221942.0)

    # Ditto.
    reg.register_halfmaxes("WISE", 1, 31476.0, 37834.0)
    reg.register_halfmaxes("WISE", 2, 40906.0, 51980.0)
    reg.register_halfmaxes("WISE", 3, 100777.0, 163535.0)
    reg.register_halfmaxes("WISE", 4, 198530.0, 245927.0)


builtin_registrars["WISE"] = register_wise
