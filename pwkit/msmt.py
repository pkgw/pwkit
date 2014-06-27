# -*- mode: python; coding: utf-8 -*-
# Copyright 2013-2014 Peter Williams <peter@newton.cx> and collaborators.
# Licensed under the MIT license.

"""
pwkit.msmt - Working with uncertain measurements.

Classes:

Uval - An empirical uncertain value represented by numerical samples.

Miscellaneous functions:

find_gamma_params    - Compute reasonable Γ distribution parameters given mode/stddev.
pk_scoreatpercentile - Simplified version of scipy.stats.scoreatpercentile.
sample_gamma         - Sample from a Γ distribution with α/β parametrization.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = (b'Uval find_gamma_params pk_scoreatpercentile sample_gamma '
           b'uval_dtype uval_nsamples').split ()

import operator

import numpy as np

from . import unicode_to_str


uval_nsamples = 1024
uval_dtype = np.double


# This is a copy of scipy.stats' scoreatpercentile() function with simplified
# functionality. This way we don't depend on scipy, and we can count on the
# ability to handle vector `per`, which earlier versions incorrectly didn't.
# I'm also tempted to make percentiles be in [0, 1], not [0, 100], but
# gratuitous incompatibilities seem unwise.

def pk_scoreatpercentile (a, per):
    asort = np.sort (a)
    vper = np.atleast_1d (per)

    if np.any ((vper < 0) | (vper > 100)):
        raise ValueError ('`per` must be in the range [0, 100]')

    fidx = vper / 100. * (asort.size - 1)
    # clipping iidx here gets the right behavior for per = 100:
    iidx = np.minimum (fidx.astype (np.int), asort.size - 2)
    res = (iidx + 1 - fidx) * asort[iidx] + (fidx - iidx) * asort[iidx + 1]

    if np.isscalar (per):
        return res[0]
    return res


# Utilities for gamma distributions. We want these for values that are
# presented as being (skew)normal but must be positive. If the uncertainty is
# not much smaller than the value, the samples can cross zero in significant
# numbers and lead to all sorts of bad behavior in some kinds of computation
# (e.g., taking logarithms, which we do a lot).

def sample_gamma (alpha, beta, size):
    """This is mostly about recording the conversion between Numpy/Scipy
    conventions and Wikipedia conventions. Some equations:

    mean = alpha / beta
    variance = alpha / beta**2
    mode = (alpha - 1) / beta [if alpha > 1; otherwise undefined]
    skewness = 2 / sqrt (alpha)
    """

    if alpha <= 0:
        raise ValueError ('alpha must be positive; got %e' % alpha)
    if beta <= 0:
        raise ValueError ('beta must be positive; got %e' % beta)
    return np.random.gamma (alpha, scale=1./beta, size=size)


def find_gamma_params (mode, std):
    """Given a modal value and a standard deviation, compute corresponding
    parameters for the gamma distribution.

    Intended to be used to replace normal distributions when the value must be
    positive and the uncertainty is comparable to the best value. Conversion
    equations determined from the relations given in the sample_gamma()
    docs.

    """
    if mode < 0:
        raise ValueError ('input mode must be positive for gamma; got %e' % mode)

    var = std**2
    beta = (mode + np.sqrt (mode**2 + 4 * var)) / (2 * var)
    j = 2 * var / mode**2
    alpha = (j + 1 + np.sqrt (2 * j + 1)) / j

    if alpha <= 1:
        raise ValueError ('couldn\'t compute self-consistent gamma parameters: '
                          'mode=%e std=%e alpha=%e beta=%e' % (mode, std, alpha, beta))

    return alpha, beta


# The fundamental "Uval" class.

def _make_uval_operator (opfunc):
    def uvalopfunc (uval, other):
        if isinstance (other, Uval):
            otherd = other.d
        else:
            try:
                otherd = float (other)
            except Exception:
                return NotImplemented

        r = Uval (noinit=True)
        r.d = opfunc (uval.d, otherd)
        return r

    return uvalopfunc


def _make_uval_rev_operator (opfunc):
    def uvalopfunc (uval, other):
        if isinstance (other, Uval):
            otherd = other.d
        else:
            try:
                otherd = float (other)
            except Exception:
                return NotImplemented

        r = Uval (noinit=True)
        r.d = opfunc (otherd, uval.d)
        return r

    return uvalopfunc


def _make_uval_inpl_operator (opfunc):
    def uvalopfunc (uval, other):
        if isinstance (other, Uval):
            otherd = other.d
        else:
            try:
                otherd = float (other)
            except Exception:
                return NotImplemented
        uval.d = opfunc (uval.d, otherd)
        return uval
    return uvalopfunc


class Uval (object):
    __slots__ = ('d', )

    # Initialization.

    def __init__ (self, noinit=False):
        if not noinit:
            # Precisely zero.
            self.d = np.zeros (uval_nsamples, dtype=uval_dtype)

    @staticmethod
    def from_fixed (v):
        u = Uval (noinit=False)
        u.d.fill (v)
        return u

    @staticmethod
    def from_norm (val, uncert):
        if uncert < 0:
            raise ValueError ('uncert must be positive')
        u = Uval (noinit=True)
        u.d = np.random.normal (val, uncert, uval_nsamples)
        return u

    @staticmethod
    def from_unif (lower_incl, upper_excl):
        if upper_excl <= lower_incl:
            raise ValueError ('upper_excl must be greater than lower_incl')
        u = Uval (noinit=True)
        u.d = np.random.uniform (lower_incl, upper_excl, uval_nsamples)
        return u

    @staticmethod
    def from_gamma (alpha, beta):
        if alpha <= 0:
            raise ValueError ('gamma parameter `alpha` must be positive')
        if beta <= 0:
            raise ValueError ('gamma parameter `beta` must be positive')
        u = Uval (noinit=True)
        u.d = sample_gamma (alpha, beta, uval_nsamples)
        return u

    @staticmethod
    def from_pcount (nevents):
        """We assume a Poisson process. nevents is the number of events in
        some interval. The distribution of values is the distribution of the
        Poisson rate parameter given this observed number of events, where the
        "rate" is in units of events per interval of the same duration. The
        max-likelihood value is nevents, but the mean value is nevents + 1.
        The gamma distribution is obtained by assuming an improper, uniform
        prior for the rate between 0 and infinity."""
        if nevents < 0:
            raise ValueError ('Poisson parameter `nevents` must be nonnegative')
        u = Uval (noinit=True)
        u.d = np.random.gamma (nevents + 1, size=uval_nsamples)
        return u

    # Interrogation. Would be nice to have a way to estimate the
    # distribution's mode -- when a scientist writes V = X +- Y, I think they
    # usually think of X as the maximum likelihood value, which is the modal
    # value. Maybe a kernel density estimator do this sensibly? Anyway, I
    # don't think we're in a position to unilaterally declare what the "best"
    # representative value is, so we provide options and let the caller
    # decide.

    def repvals (self, method):
        """Compute representative statistical values for this Uval. `method`
        may be either 'pct' or 'gauss'.

        Returns (best, minus_one_sigma, plus_one_sigma), where `best` is the
        "best" value in some sense, and the others correspond to values at
        the ~16 and 84 percentile limits, respectively. Because of the
        sampled nature of the Uval system, there is no single method to
        compute these numbers.

        The "pct" method returns the 50th, 15.866th, and 84.134th percentile
        values.

        The "gauss" method computes the mean μ and standard deviation σ of the
        samples and returns [μ, μ-σ, μ+σ].

        """
        if method == 'pct':
            return pk_scoreatpercentile (self.d, [50., 15.866, 84.134])
        if method == 'gauss':
            m, s = self.d.mean (), self.d.std ()
            return np.asarray ([m, m - s, m + s])
        raise ValueError ('unknown representative-value method "%s"' % method)


    # Textualization.

    def text_pieces (self, method, uplaces=2):
        """Return (main, dhigh, dlow, sharedexponent), all as strings. The
        delta terms do not have sign indicators. Any item except the first
        may be None.

        `method` is passed to Uval.repvals() to compute representative
        statistical limits.

        """
        md, lo, hi = self.repvals (method)

        if hi == lo:
            return '%g' % lo, None, None, None

        if not np.isfinite ([lo, md, hi]).all ():
            raise ValueError ('got nonfinite values when formatting Uval')

        # Deltas. Round to limited # of places because we don't actually know
        # the fourth moment of the thing we're trying to describe.

        from numpy import abs, ceil, floor, log10

        dh = hi - md
        dl = md - lo

        if dh <= 0:
            raise ValueError ('strange problem formatting Uval; '
                              'hi=%g md=%g dh=%g' % (hi, md, dh))
        if dl <= 0:
            raise ValueError ('strange problem formatting Uval; '
                              'lo=%g md=%g dl=%g' % (lo, md, dl))

        p = int (ceil (log10 (dh)))
        rdh = round (dh * 10**(-p), uplaces) * 10**p
        p = int (ceil (log10 (dl)))
        rdl = round (dl * 10**(-p), uplaces) * 10**p

        # The least significant place to worry about is the L.S.P. of one of
        # the deltas, which we can find relative to its M.S.P. Any precision
        # in the datum beyond this point is false.

        lsp = int (ceil (log10 (min (rdh, rdl)))) - uplaces

        # We should round the datum since it might be something like
        # 0.999+-0.1 and we're about to try to decide what its most
        # significant place is. Might get -1 rather than 0.

        rmd = round (md, -lsp)

        if rmd == -0.: # 0 = -0, too, but no problem there.
            rmd = 0.

        # The most significant place to worry about is the M.S.P. of any of
        # the datum or the deltas. rdl and rdl must be positive, but not
        # necessarily rmd.

        msp = int (floor (log10 (max (abs (rmd), rdh, rdl))))

        # If we're not very large or very small, don't use scientific
        # notation.

        if msp > -3 and msp < 3:
            srmd = '%.*f' % (-lsp, rmd)
            srdh = '%.*f' % (-lsp, rdh)
            srdl = '%.*f' % (-lsp, rdl)
            return srmd, srdh, srdl, None

        # Use scientific notation. Adjust values, then format.

        armd = rmd * 10**-msp
        ardh = rdh * 10**-msp
        ardl = rdl * 10**-msp
        prec = msp - lsp

        sarmd = '%.*f' % (prec, armd)
        sardh = '%.*f' % (prec, ardh)
        sardl = '%.*f' % (prec, ardl)
        return sarmd, sardh, sardl, str (msp)


    def format (self, method, parenexp=True, uplaces=2):
        main, dh, dl, exp = self.text_pieces (method, uplaces=uplaces)

        if exp is not None and not parenexp:
            main += 'e' + exp
            if dh is not None:
                dh += 'e' + exp
            if dl is not None:
                dl += 'e' + exp

        if dh is None:
            pmterm = ''
        elif dh == dl:
            pmterm = 'pm' + dh
        else:
            pmterm = ''.join (['p', dh, 'm', dl])

        if exp is not None and parenexp:
            return '(%s%s)e%s' % (main, pmterm, exp)

        return main + pmterm


    def __str__ (self):
        try:
            return self.format ('pct')
        except ValueError:
            return '{bad samples}'

    __unicode__ = unicode_to_str


    def __repr__ (self):
        formatted = self.format ('pct')
        v = pk_scoreatpercentile (self.d, [0, 2.5, 50, 97.5, 100])
        return '<Uval %s [min=%g l95=%g med=%g u95=%g max=%g]>' % \
            ((formatted, ) + tuple (v))


    def latex (self, method, uplaces=1):
        main, dh, dl, exp = self.text_pieces (method, uplaces=uplaces)

        if dh is None:
            return r'$%s$' % main

        if dh == dl:
            pmterm = r'\pm %s' % dh
        else:
            pmterm = r'^{%s}_{%s}' % (dh, dl)

        if exp is None:
            return '$%s %s$' % (main, pmterm)

        return r'$(%s %s) \times 10^{%s}$' % (main, pmterm, exp)


    def ulatex3col (self, method, uplaces=1):
        main, dh, dl, exp = self.text_pieces (method, uplaces=uplaces)

        if dh is None:
            return r'\multicolumn{3}{c}{$%s$}' % main

        if dh == dl:
            pmterm = r'$\pm\,%s$' % dh
        else:
            pmterm = r'$\pm\,^{%s}_{%s}$' % (dh, dl)

        if '.' not in main:
            mainterm = r'$%s$ & ' % main
        else:
            mainterm = r'$%s$ & $.%s$' % tuple (main.split ('.'))

        if exp is None:
            return mainterm + ' & ' + pmterm

        return ''.join (['$($', mainterm, ' & ', pmterm,
                         r'$) \times 10^{%s}$' % exp])


    # math -- http://docs.python.org/2/reference/datamodel.html#emulating-numeric-types

    __add__ = _make_uval_operator (operator.add)
    __sub__ = _make_uval_operator (operator.sub)
    __mul__ = _make_uval_operator (operator.mul)
    __floordiv__ = _make_uval_operator (operator.floordiv)
    __mod__ = _make_uval_operator (operator.mod)
    __divmod__ = _make_uval_operator (divmod)
    __pow__ = _make_uval_operator (operator.pow)
    # skipped: lshift, rshift, and, xor, or
    __div__ = _make_uval_operator (operator.div)
    __truediv__ = _make_uval_operator (operator.truediv)

    __radd__ = _make_uval_rev_operator (operator.add)
    __rsub__ = _make_uval_rev_operator (operator.sub)
    __rmul__ = _make_uval_rev_operator (operator.mul)
    __rfloordiv__ = _make_uval_rev_operator (operator.floordiv)
    __rmod__ = _make_uval_rev_operator (operator.mod)
    __rdivmod__ = _make_uval_rev_operator (divmod)
    __rpow__ = _make_uval_rev_operator (operator.pow)
    # skipped: rlshift, rrshift, rand, rxor, ror
    __rdiv__ = _make_uval_rev_operator (operator.div)
    __rtruediv__ = _make_uval_rev_operator (operator.truediv)

    __iadd__ = _make_uval_inpl_operator (operator.iadd)
    __isub__ = _make_uval_inpl_operator (operator.isub)
    __imul__ = _make_uval_inpl_operator (operator.imul)
    __ifloordiv__ = _make_uval_inpl_operator (operator.ifloordiv)
    __imod__ = _make_uval_inpl_operator (operator.imod)
    __ipow__ = _make_uval_inpl_operator (operator.ipow)
    # skipped: ilshift, irshift, iand, ixor, ior
    __idiv__ = _make_uval_inpl_operator (operator.idiv)
    __itruediv__ = _make_uval_inpl_operator (operator.itruediv)

    def __neg__ (self):
        self.d = -self.d
        return self

    def __pos__ (self):
        self.d = +self.d
        return self

    def __abs__ (self):
        self.d = np.abs (self.d)
        return self

    def __invert__ (self):
        self.d = ~self.d
        return self.d

    def __complex__ (self):
        # TODO: allow if we're actually a precise scalar, and suggest
        # a method that gives the median.
        raise TypeError ('uncertain value cannot be reduced to a complex scalar')

    def __int__ (self):
        raise TypeError ('uncertain value cannot be reduced to an integer scalar')

    def __long__ (self):
        raise TypeError ('uncertain value cannot be reduced to a long-integer scalar')

    def __float__ (self):
        raise TypeError ('uncertain value cannot be reduced to a float scalar')

    # skipped: oct, hex, index, coerce

    def __lt__ (self, other):
        raise TypeError ('uncertain value does not have a well-defined "<" comparison')

    def __le__ (self, other):
        raise TypeError ('uncertain value does not have a well-defined "<" comparison')

    def __eq__ (self, other):
        raise TypeError ('uncertain value does not have a well-defined "==" comparison')

    def __ne__ (self, other):
        raise TypeError ('uncertain value does not have a well-defined "!=" comparison')

    def __gt__ (self, other):
        raise TypeError ('uncertain value does not have a well-defined ">" comparison')

    def __ge__ (self, other):
        raise TypeError ('uncertain value does not have a well-defined ">=" comparison')

    def __cmp__ (self, other):
        raise TypeError ('uncertain value does not have a well-defined __cmp__ comparison')

    __hash__ = None

    def __nonzero__ (self):
        raise TypeError ('uncertain value does not have a well-defined boolean value')


    def debug_distribution (self):
        import omega as om

        v = pk_scoreatpercentile (self.d, [50, 0, 0.270, 2.5, 97.5, 99.730, 100])
        median = v[0]
        v = v[1:]

        print ('median=%g mean=%g'
               % (median, self.d.mean ()))
        print ('   abs: min=%g l3σ=%g l95%%=%g .. u95%%=%g u3σ=%g max=%g'
               % tuple (v))
        print ('   rel: min=%g l3σ=%g l95%%=%g .. u95%%=%g u3σ=%g max=%g'
               % tuple (v - median))
        print ('   scl: min=%.2f l3σ=%.2f l95%%=%.2f .. u95%%=%.2f u3σ=%.2f max=%.2f'
               % tuple ((v - median) / np.abs (median)))
        return om.quickHist (self.d, bins=25)
