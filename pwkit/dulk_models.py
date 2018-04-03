# -*- mode: python; coding: utf-8 -*-
# Copyright 2016-2018 Peter Williams <peter@newton.cx> and collaborators.
# Licensed under the MIT License.

"""Model radio-wavelength radiative transfer using the Dulk (1985) equations.

Note that the gyrosynchrotron and relativistic synchrotron expressions can
give *very* different answers! For s=20, delta=3, theta=0.7, the results
differ by three orders of magnitude for η! The paper is a bit vague but
mentions that the gyrosynchrotron case is when "γ <~ 2 or 3". The synchrotron
functions give results compatible with Symphony/Rimphony; the gyrosynchrotron
ones do not, although I've only tentatively explored what happens if you given
Symphony/Rimphony very low cuts on their gamma values.

"""
from __future__ import absolute_import, division, print_function

__all__ = '''
calc_nu_b
calc_snu
calc_freefree_kappa
calc_freefree_eta
calc_freefree_snu_ujy
calc_gs_eta
calc_gs_kappa
calc_gs_nu_pk
calc_gs_snu_ujy
calc_gsff_snu_ujy
calc_synch_eta
calc_synch_kappa
calc_synch_nu_pk
calc_synch_snu_ujy
'''.split()

import numpy as np
from . import cgs


# Generic

def calc_nu_b(b):
    """Calculate the cyclotron frequency in Hz given a magnetic field strength in Gauss.

    This is in cycles per second not radians per second; i.e. there is a 2π in
    the denominator: ν_B = e B / (2π m_e c)

    """
    return cgs.e * b / (2 * cgs.pi * cgs.me * cgs.c)


def calc_snu(eta, kappa, width, elongation, dist):
    """Calculate the flux density S_ν given a simple physical configuration.

    This is basic radiative transfer as per Dulk (1985) equations 5, 6, and 11.

    eta
      The emissivity, in units of ``erg s^-1 Hz^-1 cm^-3 sr^-1``.
    kappa
      The absorption coefficient, in units of ``cm^-1``.
    width
      The characteristic cross-sectional width of the emitting region, in cm.
    elongation
      The the elongation of the emitting region; ``depth = width * elongation``.
    dist
      The distance to the emitting region, in cm.

    The return value is the flux density, in units of ``erg s^-1 cm^-2
    Hz^-1``. The angular size of the source is taken to be ``(width /
    dist)**2``.

    """
    omega = (width / dist)**2
    depth = width * elongation
    tau = depth * kappa
    sourcefn = eta / kappa
    return 2 * omega * sourcefn * (1 - np.exp(-tau))


# Free-free (bremsstrahlung) emission

def calc_freefree_kappa(ne, t, hz):
    """Dulk (1985) eq 20, assuming pure hydrogen."""
    return 9.78e-3 * ne**2 * hz**-2 * t**-1.5 * (24.5 + np.log(t) - np.log(hz))


def calc_freefree_eta(ne, t, hz):
    """Dulk (1985) equations 7 and 20, assuming pure hydrogen."""
    kappa = calc_freefree_kappa(ne, t, hz)
    return kappa * cgs.k * t * hz**2 / cgs.c**2


def calc_freefree_snu_ujy(ne, t, width, elongation, dist, ghz):
    """Calculate a flux density from pure free-free emission.

    """
    hz = ghz * 1e9
    eta = calc_freefree_eta(ne, t, hz)
    kappa = calc_freefree_kappa(ne, t, hz)
    snu = calc_snu(eta, kappa, width, elongation, dist)
    ujy = snu * cgs.jypercgs * 1e6
    return ujy


# Gyrosynchrotron

def calc_gs_eta(b, ne, delta, sinth, nu):
    """Calculate the gyrosynchrotron emission coefficient η_ν.

    This is Dulk (1985) equation 35, which is a fitting function assuming a
    power-law electron population. Arguments are:

    b
      Magnetic field strength in Gauss
    ne
      The density of electrons per cubic centimeter with energies greater than 10 keV.
    delta
      The power-law index defining the energy distribution of the electron population,
      with ``n(E) ~ E^(-delta)``. The equation is valid for ``2 <~ delta <~ 7``.
    sinth
      The sine of the angle between the line of sight and the magnetic field direction.
      The equation is valid for θ > 20° or ``sinth > 0.34`` or so.
    nu
      The frequency at which to calculate η, in Hz. The equation is valid for
      ``10 <~ nu/nu_b <~ 100``, which sets a limit on the ratio of ``nu`` and ``b``.

    The return value is the emission coefficient (AKA "emissivity"), in units
    of ``erg s^-1 Hz^-1 cm^-3 sr^-1``.

    No complaints are raised if you attempt to use the equation outside of its
    range of validity.

    """
    s = nu / calc_nu_b(b)
    return (b * ne *
            3.3e-24 *
            10**(-0.52 * delta) *
            sinth**(-0.43 + 0.65 * delta) *
            s**(1.22 - 0.90 * delta))


def calc_gs_kappa(b, ne, delta, sinth, nu):
    """Calculate the gyrosynchrotron absorption coefficient κ_ν.

    This is Dulk (1985) equation 36, which is a fitting function assuming a
    power-law electron population. Arguments are:

    b
      Magnetic field strength in Gauss
    ne
      The density of electrons per cubic centimeter with energies greater than 10 keV.
    delta
      The power-law index defining the energy distribution of the electron population,
      with ``n(E) ~ E^(-delta)``. The equation is valid for ``2 <~ delta <~ 7``.
    sinth
      The sine of the angle between the line of sight and the magnetic field direction.
      The equation is valid for θ > 20° or ``sinth > 0.34`` or so.
    nu
      The frequency at which to calculate η, in Hz. The equation is valid for
      ``10 <~ nu/nu_b <~ 100``, which sets a limit on the ratio of ``nu`` and ``b``.

    The return value is the absorption coefficient, in units of ``cm^-1``.

    No complaints are raised if you attempt to use the equation outside of its
    range of validity.

    """
    s = nu / calc_nu_b(b)
    return (ne / b *
            1.4e-9 *
            10**(-0.22 * delta) *
            sinth**(-0.09 + 0.72 * delta) *
            s**(-1.30 - 0.98 * delta))


def calc_gs_nu_pk(b, ne, delta, sinth, depth):
    """Calculate the frequency of peak synchrotron emission, ν_pk.

    This is Dulk (1985) equation 39, which is a fitting function assuming a
    power-law electron population. Arguments are:

    b
      Magnetic field strength in Gauss
    ne
      The density of electrons per cubic centimeter with energies greater than 10 keV.
    delta
      The power-law index defining the energy distribution of the electron population,
      with ``n(E) ~ E^(-delta)``. The equation is valid for ``2 <~ delta <~ 7``.
    sinth
      The sine of the angle between the line of sight and the magnetic field direction.
      The equation is valid for θ > 20° or ``sinth > 0.34`` or so.
    depth
      The path length through the emitting medium, in cm.

    The return value is peak frequency in Hz.

    No complaints are raised if you attempt to use the equation outside of its
    range of validity.

    """
    coldens = ne * depth
    return (2.72e3 *
            10**(0.27 * delta) *
            sinth**(0.41 + 0.03 * delta) *
            coldens**(0.32 - 0.03 * delta) *
            b**(0.68 + 0.03 * delta))


def calc_gs_snu_ujy(b, ne, delta, sinth, width, elongation, dist, ghz):
    """Calculate a flux density from pure gyrosynchrotron emission.

    This combines Dulk (1985) equations 35 and 36, which are fitting functions
    assuming a power-law electron population, with standard radiative transfer
    through a uniform medium. Arguments are:

    b
      Magnetic field strength in Gauss
    ne
      The density of electrons per cubic centimeter with energies greater than 10 keV.
    delta
      The power-law index defining the energy distribution of the electron population,
      with ``n(E) ~ E^(-delta)``. The equation is valid for ``2 <~ delta <~ 7``.
    sinth
      The sine of the angle between the line of sight and the magnetic field direction.
      The equation is valid for θ > 20° or ``sinth > 0.34`` or so.
    width
      The characteristic cross-sectional width of the emitting region, in cm.
    elongation
      The the elongation of the emitting region; ``depth = width * elongation``.
    dist
      The distance to the emitting region, in cm.
    ghz
      The frequencies at which to evaluate the spectrum, **in GHz**.

    The return value is the flux density **in μJy**. The arguments can be
    Numpy arrays.

    No complaints are raised if you attempt to use the equations outside of
    their range of validity.

    """
    hz = ghz * 1e9
    eta = calc_gs_eta(b, ne, delta, sinth, hz)
    kappa = calc_gs_kappa(b, ne, delta, sinth, hz)
    snu = calc_snu(eta, kappa, width, elongation, dist)
    ujy = snu * cgs.jypercgs * 1e6
    return ujy


# Full-on highly relativistic synchrotron

def calc_synch_eta(b, ne, delta, sinth, nu, E0=1.):
    """Calculate the relativistic synchrotron emission coefficient η_ν.

    This is Dulk (1985) equation 40, which is an approximation assuming a
    power-law electron population. Arguments are:

    b
      Magnetic field strength in Gauss
    ne
      The density of electrons per cubic centimeter with energies greater than E0.
    delta
      The power-law index defining the energy distribution of the electron population,
      with ``n(E) ~ E^(-delta)``. The equation is valid for ``2 <~ delta <~ 5``.
    sinth
      The sine of the angle between the line of sight and the magnetic field direction.
      It's not specified for what range of values the expressions work well.
    nu
      The frequency at which to calculate η, in Hz. The equation is valid for
      It's not specified for what range of values the expressions work well.
    E0
      The minimum energy of electrons to consider, in MeV. Defaults to 1 so that
      these functions can be called identically to the gyrosynchrotron functions.

    The return value is the emission coefficient (AKA "emissivity"), in units
    of ``erg s^-1 Hz^-1 cm^-3 sr^-1``.

    No complaints are raised if you attempt to use the equation outside of its
    range of validity.

    """
    s = nu / calc_nu_b(b)
    return (b * ne *
            8.6e-24 * (delta - 1) * sinth *
            (0.175 * s / (E0**2 * sinth))**(0.5 * (1 - delta)))


def calc_synch_kappa(b, ne, delta, sinth, nu, E0=1.):
    """Calculate the relativstic synchrotron absorption coefficient κ_ν.

    This is Dulk (1985) equation 41, which is a fitting function assuming a
    power-law electron population. Arguments are:

    b
      Magnetic field strength in Gauss
    ne
      The density of electrons per cubic centimeter with energies greater than E0.
    delta
      The power-law index defining the energy distribution of the electron population,
      with ``n(E) ~ E^(-delta)``. The equation is valid for ``2 <~ delta <~ 5``.
    sinth
      The sine of the angle between the line of sight and the magnetic field direction.
      It's not specified for what range of values the expressions work well.
    nu
      The frequency at which to calculate η, in Hz. The equation is valid for
      It's not specified for what range of values the expressions work well.
    E0
      The minimum energy of electrons to consider, in MeV. Defaults to 1 so that
      these functions can be called identically to the gyrosynchrotron functions.

    The return value is the absorption coefficient, in units of ``cm^-1``.

    No complaints are raised if you attempt to use the equation outside of its
    range of validity.

    """
    s = nu / calc_nu_b(b)
    return (ne / b *
            8.7e-12 * (delta - 1) / sinth *
            E0**(delta - 1) *
            (8.7e-2 * s / sinth)**(-0.5 * (delta + 4)))


def calc_synch_nu_pk(b, ne, delta, sinth, depth, E0=1.):
    """Calculate the frequency of peak synchrotron emission, ν_pk.

    This is Dulk (1985) equation 43, which is a fitting function assuming a
    power-law electron population. Arguments are:

    b
      Magnetic field strength in Gauss
    ne
      The density of electrons per cubic centimeter with energies greater than 10 keV.
    delta
      The power-law index defining the energy distribution of the electron population,
      with ``n(E) ~ E^(-delta)``. The equation is valid for ``2 <~ delta <~ 5``.
    sinth
      The sine of the angle between the line of sight and the magnetic field direction.
      It's not specified for what range of values the expressions work well.
    depth
      The path length through the emitting medium, in cm.
    E0
      The minimum energy of electrons to consider, in MeV. Defaults to 1 so that
      these functions can be called identically to the gyrosynchrotron functions.

    The return value is peak frequency in Hz.

    No complaints are raised if you attempt to use the equation outside of its
    range of validity.

    """
    coldens = ne * depth
    return (3.2e7 * sinth *
            E0**((2 * delta - 2) / (delta + 4)) *
            (8.7e-12 * (delta - 1) * coldens / sinth)**(2./(delta + 4)) *
            b**((delta + 2) / (delta + 4)))


def calc_synch_snu_ujy(b, ne, delta, sinth, width, elongation, dist, ghz, E0=1.):
    """Calculate a flux density from pure gyrosynchrotron emission.

    This combines Dulk (1985) equations 40 and 41, which are fitting functions
    assuming a power-law electron population, with standard radiative transfer
    through a uniform medium. Arguments are:

    b
      Magnetic field strength in Gauss
    ne
      The density of electrons per cubic centimeter with energies greater than 10 keV.
    delta
      The power-law index defining the energy distribution of the electron population,
      with ``n(E) ~ E^(-delta)``. The equation is valid for ``2 <~ delta <~ 5``.
    sinth
      The sine of the angle between the line of sight and the magnetic field direction.
      It's not specified for what range of values the expressions work well.
    width
      The characteristic cross-sectional width of the emitting region, in cm.
    elongation
      The the elongation of the emitting region; ``depth = width * elongation``.
    dist
      The distance to the emitting region, in cm.
    ghz
      The frequencies at which to evaluate the spectrum, **in GHz**.
    E0
      The minimum energy of electrons to consider, in MeV. Defaults to 1 so that
      these functions can be called identically to the gyrosynchrotron functions.

    The return value is the flux density **in μJy**. The arguments can be
    Numpy arrays.

    No complaints are raised if you attempt to use the equations outside of
    their range of validity.

    """
    hz = ghz * 1e9
    eta = calc_synch_eta(b, ne, delta, sinth, hz, E0=E0)
    kappa = calc_synch_kappa(b, ne, delta, sinth, hz, E0=E0)
    snu = calc_snu(eta, kappa, width, elongation, dist)
    ujy = snu * cgs.jypercgs * 1e6
    return ujy


# Diagnostics

def calc_gsff_snu_ujy(b, ne_energetic, delta, sinth, ne_thermal, t, width, elongation, dist, ghz):
    hz = ghz * 1e9

    gs_eta = calc_gs_eta(b, ne_energetic, delta, sinth, hz)
    gs_kappa = calc_gs_kappa(b, ne_energetic, delta, sinth, hz)

    ff_kappa = calc_freefree_kappa(ne_thermal, t, hz)
    ff_eta = calc_freefree_eta(ne_thermal, t, hz)

    snu = calc_snu(gs_eta + ff_eta, gs_kappa + ff_kappa, width, elongation, dist)
    ujy = snu * cgs.jypercgs * 1e6
    return ujy
