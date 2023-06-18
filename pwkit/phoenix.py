# -*- mode: python; coding: utf-8 -*-
# Copyright 2014, 2017 Peter Williams <peter@newton.cx> and collaborators.
# Licensed under the MIT License.

"""pwkit.phoenix - Working with Phoenix atmospheric models.

Functions:

- load_spectrum - Load a model spectrum into a Pandas DataFrame.

Requires Pandas.

Individual data files for the BT-Settl models are about 120 MB, and there are
a million variations, so we do not consider bundling them with pwkit.
Therefore, we can safely expect that the model will be accessible as a path on
the filesystem.

Current BT-Settl models may be downloaded from a SPECTRA directory within `the
BT-Settl download site ``https://phoenix.ens-lyon.fr/Grids/BT-Settl/`` (see the
README). E.g.::

  https://phoenix.ens-lyon.fr/Grids/BT-Settl/CIFIST2011bc/SPECTRA/

File names are generally::

  lte{Teff/100}-{Logg}{[M/H]}a[alpha/H].GRIDNAME.spec.7.[gz|bz2|xz]

The first three columns are wavelength in Å, log10(F_λ), and log10(B_λ), where
the latter is the blackbody flux for the given Teff. The fluxes can nominally
be converted into absolute units with an offset of 8 in log space, but I doubt
that can be trusted much. Subsequent columns are related to various spectral
lines. See ``https://phoenix.ens-lyon.fr/Grids/FORMAT``.

The files do not come sorted!

"""
from __future__ import absolute_import, division, print_function

__all__ = "load_spectrum".split()

import numpy as np, pandas as pd


def load_spectrum(path, smoothing=181, DF=-8.0):
    """Load a Phoenix model atmosphere spectrum.

    path : string
      The file path to load.
    smoothing : integer
      Smoothing to apply. If None, do not smooth. If an integer, smooth with a
      Hamming window. Otherwise, the variable is assumed to be a different
      smoothing window, and the data will be convolved with it.
    DF: float
      Numerical factor used to compute the emergent flux density.

    Returns a Pandas DataFrame containing the columns:

    wlen
      Sample wavelength in Angstrom.
    flam
      Flux density in erg/cm²/s/Å. See `pwkit.synphot` for related tools.

    The values of *flam* returned by this function are computed from the
    second column of the data file as specified in the documentation: ``flam =
    10**(col2 + DF)``. The documentation states that the default value, -8, is
    appropriate for most modern models; but some older models use other
    values.

    Loading takes about 5 seconds on my current laptop. Un-smoothed spectra
    have about 630,000 samples.

    """
    try:
        ang, lflam = np.loadtxt(path, usecols=(0, 1)).T
    except ValueError:
        # In some files, the numbers in the first columns fill up the
        # whole 12-character column width, and are given in exponential
        # notation with a 'D' character, so we must be more careful:
        with open(path, "rb") as f:

            def lines():
                for line in f:
                    yield line.replace(b"D", b"e")

            ang, lflam = np.genfromtxt(lines(), delimiter=(13, 12)).T

    # Data files do not come sorted!
    z = ang.argsort()
    ang = ang[z]
    flam = 10 ** (lflam[z] + DF)
    del z

    if smoothing is not None:
        if isinstance(smoothing, int):
            smoothing = np.hamming(smoothing)
        else:
            smoothing = np.asarray(smoothing)

        wnorm = np.convolve(np.ones_like(smoothing), smoothing, mode="valid")
        smoothing = smoothing / wnorm  # do not alter original array.
        smooth = lambda a: np.convolve(a, smoothing, mode="valid")[:: smoothing.size]
        ang = smooth(ang)
        flam = smooth(flam)

    return pd.DataFrame({"wlen": ang, "flam": flam})
