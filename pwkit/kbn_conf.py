# -*- mode: python; coding: utf-8 -*-
# Copyright 2014 Peter Williams <peter@newton.cx> and collaborators.
# Licensed under the MIT License.

"""pwkit.kbn_conf - calculate Poisson-like confidence intervals assuming a background

This module implements the Bayesian confidence intervals for Poisson processes
in a background using the approach described in Kraft, Burrows, & Nousek
(1991). That paper provides tables of values; this module can calculate
intervals for arbitrary inputs. Requires `scipy`.

This implementation almost directly transcribes the equations. We do, however,
work in log-gamma space to try to avoid overflows with large values of N or B.

Functions:

kbn_conf     - Compute a single confidence limit.
vec_kbn_conf - Vectorized version of `kbn_conf`.

TODO: tests!

"""
from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = str("kbn_conf vec_kbn_conf").split()

from numpy import exp, log, vectorize
from scipy.special import gammaln
from scipy.integrate import quad
from scipy.optimize import newton


def _cconst(N, B):
    s = 0.0
    lnb = log(B)

    for n in range(N + 1):
        s += exp(-B + n * lnb - gammaln(n + 1))

    return 1.0 / s


def _fcnb(C, N, B, S):
    return C * exp(-(S + B) + N * log(S + B) - gammaln(N + 1))


def _fnb(N, B, S):
    return _fcnb(_cconst(N, B), N, B, S)


def kbn_conf(N, B, CL):
    """Given a (integer) number of observed Poisson events `N` and a (real)
    expected number of background events `B` and a confidence limit `CL`
    (between 0 and 1), return the confidence interval on the source event
    rate.

    Returns: (Smin, Smax)

    This interval is calculated using the Bayesian formalism of Kraft, Burrows, &
    Nousek (1991), which assumes no uncertainty in `B` and returns the smallest
    possible interval that satisfies the above properties.

    Example: in a certain time interval, 3 events were recorded. Based on
    external knowledge, it is expected that on average 0.5 background events will
    be recorded in the same interval. The 95% confidence interval on the source
    event rate is

    >>> kbn_conf.kbn_conf (3, 0.5, 0.95)
    <<< (0.22156, 7.40188)

    which agrees with the entry in Table 2 of KBN91.

    Reference info: 1991ApJ...374..344K, doi:10.1086/170124

    """

    tol = 1e-6

    origN = N
    try:
        N = int(N)
        assert N == origN
    except Exception:
        raise ValueError("N must be an integer")

    CL = float(CL)
    if CL <= 0.0 or CL >= 1.0:
        raise ValueError("CL must be between 0 and 1, noninclusive")

    B = float(B)
    if B < 0:
        raise ValueError("B must be nonnegative")

    # OK, arg-checking is out of the way. Precompute some things ...

    C = _cconst(N, B)
    f = lambda s: _fcnb(C, N, B, s)

    # The goal is find Smin and Smax such that the integral of _fnb between
    # Smin and Smax is CL, and _fnb (Smin) = _fnb (Smax). Follow the
    # suggestion in Kraft, Burrows, & Nousek (1991) and start at the
    # maximum-probability value, integrating outwards trying to maintain the
    # constraints. We have to be careful because smin cannot go below zero,
    # and to ignore the enormous typo ("local maximum at S = B + N"!) in the
    # paper!

    smin = smax = max(N - B, 0.0)
    fmin = f(smin)
    fmax = f(smax)
    conf = 0.0

    while conf < CL:
        if smin == 0.0 or fmin < fmax:
            stepsize = max(0.2 * abs(CL - conf) / CL / fmax, tol)
            conf += quad(f, smax, smax + stepsize)[0]
            smax += stepsize
            fmax = f(smax)
        else:
            stepsize = max(min(0.2 * abs(CL - conf) / CL / fmin, 0.1 * smin), tol)
            if smin - stepsize < tol:
                conf += quad(f, 0, smin)[0]
                smin = 0.0
            else:
                conf += quad(f, smin - stepsize, smin)[0]
                smin -= stepsize
            fmin = f(smin)

    return smin, smax


vec_kbn_conf = vectorize(
    kbn_conf,
    otypes=[float, float],
    doc="""Vectorized form of `kbn_conf`.

All three inputs must be broadcastable to a common shape.""",
)
