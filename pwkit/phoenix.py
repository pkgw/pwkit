# -*- mode: python; coding: utf-8 -*-
# Copyright 2014 Peter Williams <peter@newton.cx> and collaborators.
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
BT-Settl download site <http://phoenix.ens-lyon.fr/Grids/BT-Settl/>`_ (see the
README). E.g.::

  http://phoenix.ens-lyon.fr/Grids/BT-Settl/CIFIST2011bc/SPECTRA/

File names are generally::

  lte{Teff/100}-{Logg}{[M/H]}a[alpha/H].GRIDNAME.spec.7.[gz|bz2|xz]

The first three columns are wavelength in Å, log10(F_λ), and log10(B_λ), where
the latter is the blackbody flux for the given Teff. The fluxes can nominally
be converted into absolute units with an offset of 8 in log space, but I doubt
that can be trusted much. Subsequent columns are related to various spectral
lines. See http://phoenix.ens-lyon.fr/Grids/FORMAT .

The files do not come sorted!

"""
from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = str ('load_spectrum').split ()

import numpy as np, pandas as pd


def load_spectrum (path, smoothing=181):
    """Load a Phoenix model atmosphere spectrum.

    path : string
      The file path to load.
    smoothing : integer
      Smoothing to apply. If None, do not smooth. If an integer, smooth with a
      Hamming window. Otherwise, the variable is assumed to be a different
      smoothing window, and the data will be convolved with it.

    Returns a Pandas DataFrame containing the columns:

    wlen
      Sample wavelength in Angstrom.
    flam
      Flux density in erg/cm²/s/Å. See `pwkit.synphot` for related tools.

    Loading takes about 5 seconds on my current laptop. Un-smoothed spectra
    have about 630,000 samples.

    """
    ang, lflam = np.loadtxt (path, usecols=(0,1)).T

    # Data files do not come sorted!
    z = ang.argsort ()
    ang = ang[z]
    flam = 10**lflam[z]
    del z

    if smoothing is not None:
        if isinstance (smoothing, int):
            smoothing = np.hamming (smoothing)
        else:
            smoothing = np.asarray (smoothing)

        wnorm = np.convolve (np.ones_like (smoothing), smoothing, mode='valid')
        smoothing = smoothing / wnorm # do not alter original array.
        smooth = lambda a: np.convolve (a, smoothing, mode='valid')[::smoothing.size]
        ang = smooth (ang)
        flam = smooth (flam)

    return pd.DataFrame ({'wlen': ang, 'flam': flam})
