# -*- mode: python; coding: utf-8 -*-
# Copyright 2012-2014 Peter Williams <peter@newton.cx> and collaborators.
# Licensed under the MIT License.

"""pwkit.radio_cal_models - models of radio calibrator flux densities.

From the command line::

    python -m pwkit.radio_cal_models [-f] <source> <freq[mhz]>
    python -m pwkit.radio_cal_models [-f] CasA     <freq[mhz]> <year>

Print the flux density of the specified calibrator at the specified frequency,
in Janskys.

Arguments:

<source>
  the source name (e.g., 3c348)
<freq>
  the observing frequency in MHz (e.g., 1420)
<year>
  is the decimal year of the observation (e.g., 2007.8).
  Only needed if <source> is CasA.
``-f``
  activates "flux" mode, where a three-item string is
  printed that can be passed to MIRIAD tasks that accept a
  model flux and spectral index argument.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = str ('cas_a commandline init_cas_a models spindexes').split ()


import six
import numpy as np

from . import PKError

models = {}
spindexes = {}


def cas_a (freq_mhz, year):
    """Return the flux of Cas A given a frequency and the year of observation.
    Based on the formula given in Baars et al., 1977.

    Parameters:

    freq - Observation frequency in MHz.
    year - Year of observation. May be floating-point.

    Returns: s, flux in Jy.

    """
    # The snu rule is right out of Baars et al. The dnu is corrected
    # for the frequency being measured in MHz, not GHz.

    snu = 10. ** (5.745 - 0.770 * np.log10 (freq_mhz)) # Jy
    dnu = 0.01 * (0.07 - 0.30 * np.log10 (freq_mhz)) # percent per yr.
    loss = (1 - dnu) ** (year - 1980.)
    return snu * loss


def init_cas_a (year):
    """Insert an entry for Cas A into the table of models. Need to specify the
    year of the observations to account for the time variation of Cas A's
    emission.

    """
    year = float (year)
    models['CasA'] = lambda f: cas_a (f, year)


# Other models from Baars et al. 1977 -- data from Table 5 in that paper. Some
# of these will be overwritten by VLA models below.

def _add_generic_baars (src, a, b, c, fmin, fmax):
    def fluxdens (freq_mhz):
        if np.any (freq_mhz < fmin) or np.any (freq_mhz > fmax):
            raise PKError ('going beyond frequency limits of model: want '
                           '%f, but validity is [%f, %f]', freq_mhz, fmin, fmax)
        lf = np.log10 (freq_mhz)
        return 10.**(a + b * lf + c * lf**2)

    def spindex (freq_mhz):
        if np.any (freq_mhz < fmin) or np.any (freq_mhz > fmax):
            raise PKError ('going beyond frequency limits of model: want '
                           '%f, but validity is [%f, %f]', freq_mhz, fmin, fmax)
        return b + 2 * c * np.log10 (freq_mhz)

    models[src] = fluxdens
    spindexes[src] = spindex


baars_parameters = {
    '3c48': (2.345, 0.071, -0.138, 405., 15000.),
    '3c123': (2.921, -0.002, -0.124, 405., 15000.),
    '3c147': (1.766, 0.447, -0.184, 405., 15000.),
    '3c161': (1.633, 0.498, -0.194, 405., 10700.),
    '3c218': (4.497, -0.910, 0.0, 405., 10700.),
    '3c227': (3.460, -0.827, 0.0, 405, 15000.),
    '3c249.1': (1.230, 0.288, -0.176, 405., 15000.),
    '3c286': (1.480, 0.292, -0.124, 405., 15000.),
    '3c295': (1.485, 0.759, -0.255, 405., 15000.),
    '3c348': (4.963, -1.052, 0., 405., 10700.),
    '3c353': (2.944, -0.034, -0.109, 405., 10700.),
    'DR21': (1.81, -0.122, 0., 7000., 31000.),
    'NGC7027': (1.32, -0.127, 0., 10000., 31000.)
}

def _init_generic_baars ():
    for src, info in six.iteritems (baars_parameters):
        _add_generic_baars (src, *info)

_init_generic_baars ()


# VLA models of calibrators: see
# http://www.vla.nrao.edu/astro/calib/manual/baars.html These are the 1999.2
# values. This makes them pretty out of date, but still a lot more recent than
# Baars.

def _add_vla_model (src, a, b, c, d):
    def fluxdens (freq_mhz):
        if np.any (freq_mhz < 300) or np.any (freq_mhz > 50000):
            raise PKError ('going beyond frequency limits of model: want '
                           '%f, but validity is [300, 50000]', freq_mhz)
        lghz = np.log10 (freq_mhz) - 3
        return 10.**(a + b * lghz + c * lghz**2 + d * lghz**3)

    def spindex (freq_mhz):
        if np.any (freq_mhz < 300) or np.any (freq_mhz > 50000):
            raise PKError ('going beyond frequency limits of model: want '
                           '%f, but validity is [300, 50000]', freq_mhz)
        lghz = np.log10 (freq_mhz) - 3
        return b + 2 * c * lghz + 3 * d * lghz**2

    models[src] = fluxdens
    spindexes[src] = spindex


vla_parameters = {
    '3c48': (1.31752, -0.74090, -0.16708, +0.01525),
    '3c138': (1.00761, -0.55629, -0.11134, -0.01460),
    '3c147': (1.44856, -0.67252, -0.21124, +0.04077),
    '3c286': (1.23734, -0.43276, -0.14223, +0.00345),
    '3c295': (1.46744, -0.77350, -0.25912, +0.00752)
}

def _init_vla ():
    for src, info in six.iteritems (vla_parameters):
        _add_vla_model (src, *info)

_init_vla ()


# Crappier power-law modeling based on VLA Calibrator Manual catalog. It is
# not clear whether values in the catalog should be taken to supersede those
# given in the analytic models above, for those five sources that have
# analytic models. The catalog entries do not seem to necessarily be more
# recent than the analytic models.

def add_from_vla_obs (src, Lband, Cband):
    """Add an entry into the models table for a source based on L-band and
    C-band flux densities.

    """
    if src in models:
        raise PKError ('already have a model for ' + src)

    fL = np.log10 (1425)
    fC = np.log10 (4860)

    lL = np.log10 (Lband)
    lC = np.log10 (Cband)

    A = (lL - lC) / (fL - fC)
    B = lL - A * fL

    def fluxdens (freq_mhz):
        return 10. ** (A * np.log10 (freq_mhz) + B)

    def spindex (freq_mhz):
        return A

    models[src] = fluxdens
    spindexes[src] = spindex


add_from_vla_obs ('3c84', 23.9, 23.3)


# If we're executed as a program, print out a flux given a source name.

def commandline (argv):
    from . import cli

    cli.unicode_stdio ()
    cli.check_usage (__doc__, argv, usageifnoargs='long')
    flux_mode = cli.pop_option ('f', argv)
    source = argv[1]

    if source == 'CasA':
        if len (argv) != 4:
            cli.wrong_usage (__doc__, 'must give exactly three arguments '
                             'when modeling Cas A')

        try:
            init_cas_a (float (argv[3]))
        except Exception as e:
            cli.die ('unable to parse year "%s": %s', argv[3], e)
    elif len (argv) != 3:
        cli.wrong_usage (__doc__, 'must give exactly two arguments unless '
                         'modeling Cas A')

    try:
        freq = float (argv[2])
    except Exception as e:
        cli.die ('unable to parse frequency "%s": %s', argv[2], e)

    if source not in models:
        cli.die ('unknown source "%s"; known sources are: CasA, %s', source,
                 ', '.join (sorted (models.keys ())))

    try:
        flux = models[source] (freq)
    except Exception as e:
        # Catch, e.g, going beyond frequency limits.
        cli.die ('error finding flux of %s at %f MHz: %s', source, freq, e)

    if not flux_mode:
        print ('%g' % (flux, ))
        return 0

    try:
        spindex = spindexes[source] (freq)
    except Exception as e:
        cli.warn ('error finding spectral index of %s at %f MHz: %s',
                  source, freq, e)
        spindex = 0

    print ('%g,%g,%g' % (flux, freq * 1e-3, spindex))
    return 0


if __name__ == '__main__':
    from sys import argv, exit
    exit (commandline (argv))
