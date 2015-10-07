# -*- mode: python; coding: utf-8 -*-
# Copyright 2012-2014 Peter Williams <peter@newton.cx> and collaborators.
# Licensed under the MIT License.

"""pwkit.cgs - Physical constants in CGS.

Specifically, ESU-CGS im which the electron charge is measured in esu ≡
Franklin ≡ statcoulomb.

a0       - Bohr radius [cm]
alpha    - Fine structure constant [ø]
arad     - Radiation constant [erg/cm³/K⁴]
aupercm  - AU per cm
c        - Speed of light [cm/s]
cgsperjy - [erg/s/cm²/Hz] per Jy
cmperau  - cm per AU
cmperpc  - cm per parsec
conjaaev - eV/Angstrom conjugation factor: AA = conjaaev / eV [Å·eV]
e        - electron charge [esu]
ergperev - erg per eV
euler    - Euler's constant (2.71828...) [ø]
evpererg - eV per erg
G        - Gravitational constant [cm³/g/s²]
h        - Planck's constant [erg s]
hbar     - Reduced Planck's constant [erg·s]
jypercgs - Jy per [erg/s/cm²/Hz]
k        - Boltzmann's constant [erg/K]
lsun     - Luminosity of the Sun [erg/s]
me       - Mass of the electron [g]
mearth   - Mass of the Earth [g]
mjup     - Mass of Jupiter [g]
mp       - Mass of the proton [g]
msun     - Mass of the Sun [g]
mu_e     - Magnetic moment of the electron [esu·cm²/s]
pcpercm  - parsec per cm
pi       - Pi [ø]
r_e      - Classical radius of the electron [cm]
rearth   - Radius of the earth [cm]
rjup     - Radius of Jupiter [cm]
rsun     - Radius of the Sun [cm]
ryd1     - Rydberg energy [erg]
sigma    - Stefan-Boltzmann constant [erg/s/K⁴]
sigma_T  - Thomson cross section of the electron [cm²]
spersyr  - Seconds per sidereal year
syrpers  - Sidereal years per second
tsun     - Effective temperature of the Sun [K]

Functions:

blambda  - Planck function (Hz, K) -> erg/s/cm²/Hz/sr.
bnu      - Planck function (cm, K) -> erg/s/cm²/cm/sr.
exp      - Numpy exp() function.
log      - Numpy log() function.
log10    - Numpy log10() function.
sqrt     - Numpy sqrt() function.

For reference: the esu has dimensions of g^(1/2) cm^(3/2) s^-1. Electric and
magnetic field have g^(1/2) cm^(-1/2) s^-1. [esu * field] = dyne.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = str ('''
a0
alpha
arad
aupercm
blambda
bnu
c
cgsperjy
cmperau
cmperpc
conjaaev
e
ergperev
euler
evpererg
exp
G
h
hbar
jypercgs
k
log
log10
lsun
me
mearth
mjup
mp
msun
mu_e
pcpercm
pi
r_e
rearth
rjup
rsun
ryd1
sigma
sigma_T
spersyr
sqrt
syrpers
tsun
''').split ()

# make e the electron charge
from numpy import pi, e as euler, exp, sqrt, log, log10

c = 2.99792458e10 # cm / s
h = 6.6260755e-27 # erg s
me = 9.1093897e-28 # g
mp = 1.6726231e-24 # g
e = 4.80320425e-10 # esu
G = 6.67259e-8 # cm^3 g^-1 s^-2
k = 1.3806505e-16 # erg / K
hbar = h / 2 / pi
alpha = e**2  / hbar / c # dimensionless
sigma = pi**2 * k ** 4 * hbar**-3 * c**-2 / 60 # g s^-3 K^-4
arad = 4 * sigma / c # radiation constant, erg cm^-3 K^-4
a0 = hbar**2 / (me * e**2) # cm
r_e = e**2 / (me * c**2) # cm
ryd1 = e**2 / (2 * a0) # erg
mu_e = e * hbar / (2 * me * c) # magnetic moment units, whatever those are
sigma_T = 8 * pi * r_e**2 / 3 # cm^2
ergperev = 1e8 * e / c # erg / eV [dimensionless]
evpererg = 1. / ergperev
conjaaev = 1e8 * c * h / ergperev # eV*Angstrom ; lambda(AA) = conjaaev/E(eV)
cmperpc = 3.08568025e18 # cm / pc [dimensionless]
pcpercm = 1. / cmperpc
cmperau = 1.49598e13 # cm / AU [dimensionless]
aupercm = 1. / cmperau
spersyr = 31558150. # s / sidereal yr [dimensionless]
syrpers = 1. / spersyr

# Astro
msun = 1.989e33 # g
rsun = 6.9599e10 # cm
lsun = 3.826e33 # erg s^-1
tsun = 5770 # K
mearth = 5.974e27 # g
rearth = 6.378e8 # cm
mjup = 1.8986e30 # g
rjup = 7.1492e9 # cm (equatorial)

jypercgs = 1e23 # jy per (erg s^-1 cm^-2 Hz^-1) [dimensionless]
cgsperjy = 1e-23

bnu = lambda nu, T: 2 * h * nu**3 * c**-2 / (exp (h * nu / k / T) - 1)
blambda = lambda lam, T: 2 * h * lam**-5 * c**2 / (exp (h * c / lam / k / T) - 1)
