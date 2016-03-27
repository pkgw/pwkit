# -*- mode: python; coding: utf-8 -*-
# Copyright 2016 Peter Williams <peter@newton.cx> and collaborators
# Licensed under the MIT License

"""Compute a table of expected background counts in different X-ray energy
bands.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = str('''
get_region_area
count_events
compute_bgband
''').split ()


def get_region_area (env, evtpath, region):
    with env.slurp (argv=['dmlist', '%s[sky=%s]' % (evtpath, region), 'subspace'], linebreak=True) as s:
        for etype, payload in s:
            if etype != 'stdout':
                continue

            if 'Region area' not in payload:
                continue

            return float (payload.split ()[-1])

    raise Exception ('parsing of dmlist output failed')


def count_events (env, evtpath, filter):
    """TODO: this can probably be replaced with simply reading the file
    ourselves!

    """
    with env.slurp (argv=['dmstat', '%s%s[cols energy]' % (evtpath, filter)], linebreak=True) as s:
        for etype, payload in s:
            if etype != 'stdout':
                continue

            if 'good:' not in payload:
                continue

            return int (payload.split ()[-1])

    raise Exception ('parsing of dmlist output failed')


def compute_bgband (evtpath, srcreg, bkgreg, ebins, env=None):
    """Compute background information for a source in one or more energy bands.

    evtpath
      Path to a CIAO events file
    srcreg
      String specifying the source region to consider; use 'region(path.reg)' if you
      have the region saved in a file.
    bkgreg
      String specifying the background region to consider; same format as srcreg
    ebins
      Iterable of 2-tuples giving low and high bounds of the energy bins to
      consider, measured in eV.
    env
      An optional CiaoEnvironment instance; default settings are used if unspecified.

    Returns a DataFrame containing at least the following columns:

    elo
      The low bound of this energy bin, in eV.
    ehi
      The high bound of this energy bin, in eV.
    ewidth
      The width of the bin in eV; simply `abs(ehi - elo)`.
    nsrc
      The number of events within the specified source region and energy range.
    nbkg
      The number of events within the specified background region and energy range.
    nbkg_scaled
      The number of background events scaled to the source area; not an integer.
    nsrc_subbed
      The estimated number of non-background events in the source region; simply
      `nsrc - nbkg_scaled`.
    log_prob_bkg
      The logarithm of the probability that all counts in the source region are due
      to background events.
    src_sigma
      The confidence of source detection in sigma inferred from log_prob_bkg.

    The probability of backgrounditude is computed as:

      b^s * exp (-b) / s!

    where `b` is `nbkg_scaled` and `s` is `nsrc`. The confidence of source detection is
    computed as:

      sqrt(2) * erfcinv (prob_bkg)

    where `erfcinv` is the inverse complementary error function.

    """
    import numpy as np
    import pandas as pd
    from scipy.special import erfcinv, gammaln

    if env is None:
        from . import CiaoEnvironment
        env = CiaoEnvironment ()

    srcarea = get_region_area (env, evtpath, srcreg)
    bkgarea = get_region_area (env, evtpath, bkgreg)

    srccounts = [count_events (env, evtpath, '[sky=%s][energy=%d:%d]' % (srcreg, elo, ehi))
                 for elo, ehi in ebins]
    bkgcounts = [count_events (env, evtpath, '[sky=%s][energy=%d:%d]' % (bkgreg, elo, ehi))
                 for elo, ehi in ebins]

    df = pd.DataFrame ({
        'elo': [t[0] for t in ebins],
        'ehi': [t[1] for t in ebins],
        'nsrc': srccounts,
        'nbkg': bkgcounts
    })

    df['ewidth'] = np.abs (df['ehi'] - df['elo'])
    df['nbkg_scaled'] = df['nbkg'] * srcarea / bkgarea
    df['log_prob_bkg'] = df['nsrc'] * np.log (df['nbkg_scaled']) - df['nbkg_scaled'] - gammaln (df['nsrc'] + 1)
    df['src_sigma'] = np.sqrt (2) * erfcinv (np.exp (df['log_prob_bkg']))
    df['nsrc_subbed'] = df['nsrc'] - df['nbkg_scaled']
    return df
