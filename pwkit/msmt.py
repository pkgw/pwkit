# -*- mode: python; coding: utf-8 -*-
# Copyright 2013-2014 Peter Williams <peter@newton.cx> and collaborators.
# Licensed under the MIT license.

"""
pwkit.msmt - Working with uncertain measurements.

Classes:

Uval       - An empirical uncertain value represented by numerical samples.
LimitError - Raised on illegal operations on upper/lower limits.
Lval       - Container for either precise values or upper/lower limits.
Textual    - A measurement recorded in textual form.

Generic unary functions on measurements:

absolute   - abs(x)
arccos     - As named.
arcsin     - As named.
arctan     - As named.
cos        - As named.
errinfo    - Get (limtype, repval, plus_1_sigma, minus_1_sigma)
expm1      - exp(x) - 1
exp        - As named.
fmtinfo    - Get (typetag, text, is_imprecise) for textual round-tripping.
isfinite   - True if the value is well-defined and finite.
liminfo    - Get (limtype, repval)
limtype    - -1 if the datum is an upper limit; 1 if lower; 0 otherwise.
log10      - As named.
log1p      - log(1+x)
log2       - As named.
log        - As named.
negative   - -x
reciprocal - 1/x
repval     - Get a "representative" value if x (in case it is uncertain).
sin        - As named.
sqrt       - As named.
square     - x**2
tan        - As named.
unwrap     - Get a version of x on which algebra can be performed.

Generic binary mathematical-ish functions:

add         - x + y
divide      - x / y, never with floor-integer division
floor_divide- x // y
multiply    - x * y
power       - x ** y
subtract    - x - y
true_divide - x / y, never with floor-integer division
typealign   - Return (x*, y*) cast to same algebra-friendly type: float, Uval, or Lval.

Miscellaneous functions:

is_measurement       - Check whether an object is numerical
find_gamma_params    - Compute reasonable Γ distribution parameters given mode/stddev.
pk_scoreatpercentile - Simplified version of scipy.stats.scoreatpercentile.
sample_double_norm   - Sample from a quasi-normal distribution with asymmetric variances.
sample_gamma         - Sample from a Γ distribution with α/β parametrization.

Variables:

lval_unary_math            - Dict of unary math functions operating on Lvals.
parsers                    - Dict of type tag to parsing functions.
scalar_unary_math          - Dict of unary math functions operating on scalars.
textual_unary_math         - Dict of unary math functions operating on Textuals.
UQUANT_UNCERT              - Scale of uncertainty assumed for in cases where it's unquantified.
uval_default_repval_method - Default method for computing Uval representative values.
uval_dtype                 - The Numpy dtype of Uval data (often ignored!)
uval_nsamples              - Number of samples used when constructing Uvals
uval_unary_math            - Dict of unary math functions operating on Uvals.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = str(
    """LimitError Lval Textual Uval absolute arccos arcsin arctan cos errinfo
    expm1 exp fmtinfo isfinite is_measurement liminfo limtype log10 log1p log2
    log negative reciprocal repval sin sqrt square tan unwrap add divide floor_divide
    multiply power subtract true_divide typealign find_gamma_params
    pk_scoreatpercentile sample_double_norm sample_gamma lval_unary_math
    parsers scalar_unary_math textual_unary_math UQUANT_UNCERT
    uval_default_repval_method uval_dtype uval_nsamples uval_unary_math"""
).split()

import operator

import numpy as np

from . import PKError, unicode_to_str


uval_nsamples = 1024
uval_dtype = np.double
uval_default_repval_method = "pct"


# This is a copy of scipy.stats' scoreatpercentile() function with simplified
# functionality. This way we don't depend on scipy, and we can count on the
# ability to handle vector `per`, which earlier versions incorrectly didn't.
# I'm also tempted to make percentiles be in [0, 1], not [0, 100], but
# gratuitous incompatibilities seem unwise.


def pk_scoreatpercentile(a, per):
    asort = np.sort(a)
    vper = np.atleast_1d(per)

    if np.any((vper < 0) | (vper > 100)):
        raise ValueError("`per` must be in the range [0, 100]")

    fidx = vper / 100.0 * (asort.size - 1)
    # clipping iidx here gets the right behavior for per = 100:
    iidx = np.minimum(fidx.astype(int), asort.size - 2)
    res = (iidx + 1 - fidx) * asort[iidx] + (fidx - iidx) * asort[iidx + 1]

    if np.isscalar(per):
        return res[0]
    return res


# Double-normal distribution -- that is, pasting together two normal
# distributions with unequal left and right variances. Skew-normal distributions
# are mathematically purer but turn out to be just obnoxiously hard to work with.
# Double-normals are ad-hoc but also much more tractable.


def sample_double_norm(mean, std_upper, std_lower, size):
    """Note that this function requires Scipy."""
    from scipy.special import erfinv

    # There's probably a better way to do this. We first draw percentiles
    # uniformly between 0 and 1. We want the peak of the distribution to occur
    # at `mean`. However, if we assign 50% of the samples to the lower half
    # and 50% to the upper half, the side with the smaller variance will be
    # overrepresented because of the 1/sigma normalization of the Gaussian
    # PDF. Therefore we need to divide points between the two halves with a
    # fraction `cutoff` (defined below) going to the lower half. Having
    # partitioned them this way, we can then use the standard Gaussian
    # quantile function to go from percentiles to sample values -- except that
    # we must remap from [0, cutoff] to [0, 0.5] and from [cutoff, 1] to [0.5,
    # 1].

    samples = np.empty(size)
    percentiles = np.random.uniform(0.0, 1.0, size)
    cutoff = std_lower / (std_lower + std_upper)

    w = percentiles < cutoff
    percentiles[w] *= 0.5 / cutoff
    samples[w] = mean + np.sqrt(2) * std_lower * erfinv(2 * percentiles[w] - 1)

    w = ~w
    percentiles[w] = 1 - (1 - percentiles[w]) * 0.5 / (1 - cutoff)
    samples[w] = mean + np.sqrt(2) * std_upper * erfinv(2 * percentiles[w] - 1)

    return samples


# Utilities for gamma distributions. We want these for values that are
# presented as being (skew)normal but must be positive. If the uncertainty is
# not much smaller than the value, the samples can cross zero in significant
# numbers and lead to all sorts of bad behavior in some kinds of computation
# (e.g., taking logarithms, which we do a lot).


def sample_gamma(alpha, beta, size):
    """This is mostly about recording the conversion between Numpy/Scipy
    conventions and Wikipedia conventions. Some equations:

    mean = alpha / beta
    variance = alpha / beta**2
    mode = (alpha - 1) / beta [if alpha > 1; otherwise undefined]
    skewness = 2 / sqrt(alpha)
    """

    if alpha <= 0:
        raise ValueError("alpha must be positive; got %e" % alpha)
    if beta <= 0:
        raise ValueError("beta must be positive; got %e" % beta)
    return np.random.gamma(alpha, scale=1.0 / beta, size=size)


def find_gamma_params(mode, std):
    """Given a modal value and a standard deviation, compute corresponding
    parameters for the gamma distribution.

    Intended to be used to replace normal distributions when the value must be
    positive and the uncertainty is comparable to the best value. Conversion
    equations determined from the relations given in the sample_gamma()
    docs.

    """
    if mode < 0:
        raise ValueError("input mode must be positive for gamma; got %e" % mode)

    var = std**2
    beta = (mode + np.sqrt(mode**2 + 4 * var)) / (2 * var)
    j = 2 * var / mode**2
    alpha = (j + 1 + np.sqrt(2 * j + 1)) / j

    if alpha <= 1:
        raise ValueError(
            "couldn't compute self-consistent gamma parameters: "
            "mode=%e std=%e alpha=%e beta=%e" % (mode, std, alpha, beta)
        )

    return alpha, beta


# Scalar math.
#
# What's going on here is that we want to provide a library of standard math
# functions that can operate on any kind of measurement. Rather than writing
# several large and complicated handle-anything functions, we implement them
# independently for each data type. At the bottom of this module we then have
# some generic code the uses the type of its argument to determine which
# function to invoke. This declutters things and also lets the implementations
# of math operators take advantage of the unary functions, as is done
# extensively in the Lval implementation.
#
# For scalars, we just delegate to Numpy.

scalar_unary_math = {
    "absolute": np.absolute,
    "arccos": np.arccos,
    "arcsin": np.arcsin,
    "arctan": np.arctan,
    "cos": np.cos,
    "expm1": np.expm1,
    "exp": np.exp,
    "isfinite": np.isfinite,
    "log10": np.log10,
    "log1p": np.log1p,
    "log2": np.log2,
    "log": np.log,
    "negative": np.negative,
    "reciprocal": lambda x: 1.0
    / x,  # numpy reciprocal barfs on ints. I don't want that.
    "sin": np.sin,
    "sqrt": np.sqrt,
    "square": np.square,
    "tan": np.tan,
}


# The fundamental "Uval" class.
#
# These have a more extensive math/operator library than Lval and Textual
# since it's so easy to implement things.


def _to_uval_info(value):
    if isinstance(value, Uval):
        return value.d
    return float(value)  # broadcasting FTW


def _make_uval_operator(opfunc):
    def uvalopfunc(uval, other):
        try:
            otherd = _to_uval_info(other)
        except Exception:
            return NotImplemented
        return Uval(opfunc(uval.d, otherd))

    return uvalopfunc


def _make_uval_rev_operator(opfunc):
    def uvalopfunc(uval, other):
        try:
            otherd = _to_uval_info(other)
        except Exception:
            return NotImplemented
        return Uval(opfunc(otherd, uval.d))

    return uvalopfunc


def _make_uval_inpl_operator(opfunc):
    def uvalopfunc(uval, other):
        try:
            otherd = _to_uval_info(other)
        except Exception:
            return NotImplemented
        uval.d = opfunc(uval.d, otherd)
        return uval

    return uvalopfunc


class Uval(object):
    """An empirical uncertain value, represented by samples.

    Constructors are:

    - :meth:`Uval.from_other`
    - :meth:`Uval.from_fixed`
    - :meth:`Uval.from_norm`
    - :meth:`Uval.from_unif`
    - :meth:`Uval.from_double_norm`
    - :meth:`Uval.from_gamma`
    - :meth:`Uval.from_pcount`

    Key methods are:

    - :meth:`repvals`
    - :meth:`text_pieces`
    - :meth:`format`
    - :meth:`debug_distribution`

    Supported operations are:
    ``unicode() str() repr() [latexification]  + -(sub) * // / % ** += -= *= //= %= /= **= -(neg) ~ abs()``

    """

    __slots__ = ("d",)

    # Initialization.

    def __init__(self, data):
        self.d = data

    @staticmethod
    def from_other(o):
        if isinstance(o, Uval):
            return Uval(o.d.copy())
        if np.isscalar(o):
            return Uval.from_fixed(o)
        raise ValueError("cannot convert %r to a Uval" % o)

    @staticmethod
    def from_fixed(v):
        return Uval(np.zeros(uval_nsamples, dtype=uval_dtype) + v)

    @staticmethod
    def from_norm(mean, std):
        if std < 0:
            raise ValueError("std must be positive")
        return Uval(np.random.normal(mean, std, uval_nsamples))

    @staticmethod
    def from_unif(lower_incl, upper_excl):
        if upper_excl <= lower_incl:
            raise ValueError("upper_excl must be greater than lower_incl")
        return Uval(np.random.uniform(lower_incl, upper_excl, uval_nsamples))

    @staticmethod
    def from_double_norm(mean, std_upper, std_lower):
        if std_upper <= 0:
            raise ValueError("double-norm upper stddev must be positive")
        if std_lower <= 0:
            raise ValueError("double-norm lower stddev must be positive")
        return Uval(sample_double_norm(mean, std_upper, std_lower, uval_nsamples))

    @staticmethod
    def from_gamma(alpha, beta):
        if alpha <= 0:
            raise ValueError("gamma parameter `alpha` must be positive")
        if beta <= 0:
            raise ValueError("gamma parameter `beta` must be positive")
        return Uval(sample_gamma(alpha, beta, uval_nsamples))

    @staticmethod
    def from_pcount(nevents):
        """We assume a Poisson process. nevents is the number of events in
        some interval. The distribution of values is the distribution of the
        Poisson rate parameter given this observed number of events, where the
        "rate" is in units of events per interval of the same duration. The
        max-likelihood value is nevents, but the mean value is nevents + 1.
        The gamma distribution is obtained by assuming an improper, uniform
        prior for the rate between 0 and infinity."""
        if nevents < 0:
            raise ValueError("Poisson parameter `nevents` must be nonnegative")
        return Uval(np.random.gamma(nevents + 1, size=uval_nsamples))

    # Interrogation. Would be nice to have a way to estimate the
    # distribution's mode -- when a scientist writes V = X +- Y, I think they
    # usually think of X as the maximum likelihood value, which is the modal
    # value. Maybe a kernel density estimator do this sensibly? Anyway, I
    # don't think we're in a position to unilaterally declare what the "best"
    # representative value is, so we provide options and let the caller
    # decide.

    def repvals(self, method):
        """Compute representative statistical values for this Uval. `method`
        may be either 'pct' or 'gauss'.

        Returns (best, plus_one_sigma, minus_one_sigma), where `best` is the
        "best" value in some sense, and the others correspond to values at
        the ~84 and 16 percentile limits, respectively. Because of the
        sampled nature of the Uval system, there is no single method to
        compute these numbers.

        The "pct" method returns the 50th, 15.866th, and 84.134th percentile
        values.

        The "gauss" method computes the mean μ and standard deviation σ of the
        samples and returns [μ, μ+σ, μ-σ].

        """
        if method == "pct":
            return pk_scoreatpercentile(self.d, [50.0, 84.134, 15.866])
        if method == "gauss":
            m, s = self.d.mean(), self.d.std()
            return np.asarray([m, m + s, m - s])
        raise ValueError('unknown representative-value method "%s"' % method)

    # Textualization.

    def text_pieces(self, method, uplaces=2, use_exponent=True):
        """Return (main, dhigh, dlow, sharedexponent), all as strings. The
        delta terms do not have sign indicators. Any item except the first
        may be None.

        `method` is passed to Uval.repvals() to compute representative
        statistical limits.

        """
        md, hi, lo = self.repvals(method)

        if hi == lo:
            return "%g" % lo, None, None, None

        if not np.isfinite([lo, md, hi]).all():
            raise ValueError("got nonfinite values when formatting Uval")

        # Deltas. Round to limited # of places because we don't actually know
        # the fourth moment of the thing we're trying to describe.

        from numpy import abs, ceil, floor, log10

        dh = hi - md
        dl = md - lo

        if dh <= 0:
            raise ValueError(
                "strange problem formatting Uval; " "hi=%g md=%g dh=%g" % (hi, md, dh)
            )
        if dl <= 0:
            raise ValueError(
                "strange problem formatting Uval; " "lo=%g md=%g dl=%g" % (lo, md, dl)
            )

        p = int(ceil(log10(dh)))
        rdh = round(dh * 10 ** (-p), uplaces) * 10**p
        p = int(ceil(log10(dl)))
        rdl = round(dl * 10 ** (-p), uplaces) * 10**p

        # The least significant place to worry about is the L.S.P. of one of
        # the deltas, which we can find relative to its M.S.P. Any precision
        # in the datum beyond this point is false.

        lsp = int(ceil(log10(min(rdh, rdl)))) - uplaces

        # We should round the datum since it might be something like
        # 0.999+-0.1 and we're about to try to decide what its most
        # significant place is. Might get -1 rather than 0.

        rmd = round(md, -lsp)

        if rmd == -0.0:  # 0 = -0, too, but no problem there.
            rmd = 0.0

        # The most significant place to worry about is the M.S.P. of any of
        # the datum or the deltas. rdl and rdl must be positive, but not
        # necessarily rmd.

        msp = int(floor(log10(max(abs(rmd), rdh, rdl))))

        # If we're not very large or very small, or it's been explicitly
        # disabled, don't use scientific notation.

        if (msp > -3 and msp < 3) or not use_exponent:
            srmd = "%.*f" % (-lsp, rmd)
            srdh = "%.*f" % (-lsp, rdh)
            srdl = "%.*f" % (-lsp, rdl)
            return srmd, srdh, srdl, None

        # Use scientific notation. Adjust values, then format.

        armd = rmd * 10**-msp
        ardh = rdh * 10**-msp
        ardl = rdl * 10**-msp
        prec = msp - lsp

        sarmd = "%.*f" % (prec, armd)
        sardh = "%.*f" % (prec, ardh)
        sardl = "%.*f" % (prec, ardl)
        return sarmd, sardh, sardl, str(msp)

    def format(self, method, parenexp=True, uplaces=2, use_exponent=True):
        main, dh, dl, exp = self.text_pieces(
            method, uplaces=uplaces, use_exponent=use_exponent
        )

        if exp is not None and not parenexp:
            main += "e" + exp
            if dh is not None:
                dh += "e" + exp
            if dl is not None:
                dl += "e" + exp

        if dh is None:
            pmterm = ""
        elif dh == dl:
            pmterm = "pm" + dh
        else:
            pmterm = "".join(["p", dh, "m", dl])

        if exp is not None and parenexp:
            return "(%s%s)e%s" % (main, pmterm, exp)

        return main + pmterm

    def __unicode__(self):
        try:
            return self.format(uval_default_repval_method)
        except ValueError:
            return "{bad samples}"

    __str__ = unicode_to_str

    def __repr__(self):
        formatted = self.format(uval_default_repval_method)
        v = pk_scoreatpercentile(self.d, [0, 2.5, 50, 97.5, 100])
        return "<Uval %s [min=%g l95=%g med=%g u95=%g max=%g]>" % (
            (formatted,) + tuple(v)
        )

    def __pk_fmtinfo__(self):
        return "u", self.format("pct", parenexp=False), True

    def __pk_latex__(self, method=None, uplaces=1, use_exponent=True, **kwargs):
        if method is None:
            method = uval_default_repval_method
        main, dh, dl, exp = self.text_pieces(
            method, uplaces=uplaces, use_exponent=use_exponent
        )

        if dh is None:
            return r"$%s$" % main

        if dh == dl:
            pmterm = r"\pm %s" % dh
        else:
            pmterm = r"^{%s}_{%s}" % (dh, dl)

        if exp is None:
            return "$%s %s$" % (main, pmterm)

        return r"$(%s %s) \times 10^{%s}$" % (main, pmterm, exp)

    def __pk_latex_l3col__(self, method=None, **kwargs):
        if method is None:
            method = uval_default_repval_method
        v = self.repvals(method)[0]

        from .latex import latexify_n2col

        return b"$\\sim$ & " + latexify_n2col(v, **kwargs)

    def __pk_latex_u3col__(self, method=None, uplaces=1, use_exponent=True, **kwargs):
        if method is None:
            method = uval_default_repval_method
        main, dh, dl, exp = self.text_pieces(
            method, uplaces=uplaces, use_exponent=use_exponent
        )

        if dh is None:
            return r"\multicolumn{3}{c}{$%s$}" % main

        if dh == dl:
            pmterm = r"$\pm\,%s$" % dh
        else:
            pmterm = r"$\pm\,^{%s}_{%s}$" % (dh, dl)

        if "." not in main:
            mainterm = r"$%s$ & " % main
        else:
            mainterm = r"$%s$ & $.%s$" % tuple(main.split("."))

        if exp is None:
            return mainterm + " & " + pmterm

        return "".join(["$($", mainterm, " & ", pmterm, r"$) \times 10^{%s}$" % exp])

    # math -- http://docs.python.org/2/reference/datamodel.html#emulating-numeric-types

    __add__ = _make_uval_operator(operator.add)
    __sub__ = _make_uval_operator(operator.sub)
    __mul__ = _make_uval_operator(operator.mul)
    __floordiv__ = _make_uval_operator(operator.floordiv)
    __mod__ = _make_uval_operator(operator.mod)
    __divmod__ = _make_uval_operator(divmod)
    __pow__ = _make_uval_operator(operator.pow)
    # skipped: lshift, rshift, and, xor, or
    # used to do div too; Python 3 has no operator.div
    __truediv__ = _make_uval_operator(operator.truediv)

    __radd__ = _make_uval_rev_operator(operator.add)
    __rsub__ = _make_uval_rev_operator(operator.sub)
    __rmul__ = _make_uval_rev_operator(operator.mul)
    __rfloordiv__ = _make_uval_rev_operator(operator.floordiv)
    __rmod__ = _make_uval_rev_operator(operator.mod)
    __rdivmod__ = _make_uval_rev_operator(divmod)
    __rpow__ = _make_uval_rev_operator(operator.pow)
    # skipped: rlshift, rrshift, rand, rxor, ror
    # as above, used to do rdiv too
    __rtruediv__ = _make_uval_rev_operator(operator.truediv)

    __iadd__ = _make_uval_inpl_operator(operator.iadd)
    __isub__ = _make_uval_inpl_operator(operator.isub)
    __imul__ = _make_uval_inpl_operator(operator.imul)
    __ifloordiv__ = _make_uval_inpl_operator(operator.ifloordiv)
    __imod__ = _make_uval_inpl_operator(operator.imod)
    __ipow__ = _make_uval_inpl_operator(operator.ipow)
    # skipped: ilshift, irshift, iand, ixor, ior
    # as above, used to do idiv too
    __itruediv__ = _make_uval_inpl_operator(operator.itruediv)

    def __neg__(self):
        self.d = -self.d
        return self

    def __pos__(self):
        self.d = +self.d
        return self

    def __abs__(self):
        self.d = np.abs(self.d)
        return self

    def __invert__(self):
        self.d = ~self.d
        return self

    def __nonzero__(self):
        raise TypeError("uncertain value cannot be reduced to a boolean scalar")

    def __complex__(self):
        raise TypeError("uncertain value cannot be reduced to a complex scalar")

    def __int__(self):
        raise TypeError("uncertain value cannot be reduced to an integer scalar")

    def __long__(self):
        raise TypeError("uncertain value cannot be reduced to a long-integer scalar")

    def __float__(self):
        raise TypeError("uncertain value cannot be reduced to a float scalar")

    # skipped: oct, hex, index, coerce

    def __lt__(self, other):
        raise TypeError('uncertain value does not have a well-defined "<" comparison')

    def __le__(self, other):
        raise TypeError('uncertain value does not have a well-defined "<" comparison')

    def __eq__(self, other):
        raise TypeError('uncertain value does not have a well-defined "==" comparison')

    def __ne__(self, other):
        raise TypeError('uncertain value does not have a well-defined "!=" comparison')

    def __gt__(self, other):
        raise TypeError('uncertain value does not have a well-defined ">" comparison')

    def __ge__(self, other):
        raise TypeError('uncertain value does not have a well-defined ">=" comparison')

    def __cmp__(self, other):
        raise TypeError(
            "uncertain value does not have a well-defined __cmp__ comparison"
        )

    __hash__ = None

    def debug_distribution(self):
        import omega as om

        v = pk_scoreatpercentile(self.d, [50, 0, 0.270, 2.5, 97.5, 99.730, 100])
        median = v[0]
        v = v[1:]

        print("median=%g mean=%g" % (median, self.d.mean()))
        print("   abs: min=%g l3σ=%g l95%%=%g .. u95%%=%g u3σ=%g max=%g" % tuple(v))
        print(
            "   rel: min=%g l3σ=%g l95%%=%g .. u95%%=%g u3σ=%g max=%g"
            % tuple(v - median)
        )
        print(
            "   scl: min=%.2f l3σ=%.2f l95%%=%.2f .. u95%%=%.2f u3σ=%.2f max=%.2f"
            % tuple((v - median) / np.abs(median))
        )
        return om.quickHist(self.d, bins=25)


def _make_uval_unary_math(scalarfunc):
    def uval_unary_math(v):
        return Uval(scalarfunc(_to_uval_info(v)))

    return uval_unary_math


def _uval_unary_isfinite(v):
    return np.all(np.isfinite(_to_uval_info(v)))


uval_unary_math = {
    "absolute": _make_uval_unary_math(np.absolute),
    "arccos": _make_uval_unary_math(np.arccos),
    "arcsin": _make_uval_unary_math(np.arcsin),
    "arctan": _make_uval_unary_math(np.arctan),
    "cos": _make_uval_unary_math(np.cos),
    "expm1": _make_uval_unary_math(np.expm1),
    "exp": _make_uval_unary_math(np.exp),
    "isfinite": _uval_unary_isfinite,
    "log10": _make_uval_unary_math(np.log10),
    "log1p": _make_uval_unary_math(np.log1p),
    "log2": _make_uval_unary_math(np.log2),
    "log": _make_uval_unary_math(np.log),
    "negative": _make_uval_unary_math(np.negative),
    "reciprocal": _make_uval_unary_math(lambda x: 1.0 / x),
    "sin": _make_uval_unary_math(np.sin),
    "sqrt": _make_uval_unary_math(np.sqrt),
    "square": _make_uval_unary_math(np.square),
    "tan": _make_uval_unary_math(np.tan),
}


# Now, limiting values. I tried to do this within the context of the Uval
# system, but it just never worked in a way that gave the results that people
# would naively expect. Lvals are one level "above" Uvals: Lvals know about
# Uvals, but not the other way around.
#
# It turns out that we have to take a somewhat complicated approach here. Say
# X is a limiting value: X < 4. If X is really any real number < 4, 1/X is
# undefined because we pass through zero and could be anywhere between
# positive and negative infinity. On the other hand, if we really mean 0 < X <
# 4, 1/X should work out to >0.25.
#
# Practical math with limits requires both possibilities. In particular, we
# sometimes want to take reciprocals or logs of numbers known to be positive,
# and sometimes we'll want to exponentiate numbers that are known to be logs.
# After several false starts, the system I've devised below seems to allow
# sane operation in the majority of cases.

_lval_pos_sigils = {
    "exact": "",
    "uncertain": "~",
    "toinf": ">",
    "tozero": "<",
    "pastzero": "<<",
    "undef": "!",
}

_lval_kmap_reciprocal = {
    "toinf": "tozero",
    "tozero": "toinf",
    "pastzero": "undef",
}

_lval_kmap_add_unconditional = {
    ("exact", "exact"): "exact",
    ("exact", "tozero"): "undef",
    ("exact", "uncertain"): "uncertain",
    ("tozero", "uncertain"): "undef",
    ("uncertain", "uncertain"): "uncertain",
}

_lval_kmap_mul = {
    ("exact", "exact"): "exact",
    ("exact", "pastzero"): "pastzero",
    ("exact", "toinf"): "toinf",
    ("exact", "tozero"): "tozero",
    ("exact", "uncertain"): "uncertain",
    ("pastzero", "pastzero"): "undef",
    ("pastzero", "toinf"): "undef",
    ("pastzero", "tozero"): "pastzero",
    ("pastzero", "uncertain"): "pastzero",
    ("toinf", "toinf"): "toinf",
    # ('toinf', 'tozero'): special case -> >0
    ("toinf", "uncertain"): "toinf",
    ("tozero", "tozero"): "tozero",
    ("tozero", "uncertain"): "tozero",
    ("uncertain", "uncertain"): "uncertain",
}

_lval_kmap_pow_zero_to_one = {
    "pastzero": "toinf",
    "toinf": "tozero",
    "tozero": "undef",  # this yields a value in [v**l, 1], which is inexpressible.
}

_lval_kmap_pow_above_one = {
    "pastzero": "tozero",
    "toinf": "toinf",
    "tozero": "undef",  # this yields a value in [1, v**l], which is inexpressible.
}

_lval_kmap_exp = _lval_kmap_pow_above_one  # same behavior

_lval_kmap_log = {
    "tozero": "pastzero",
}


class LimitError(PKError):
    def __init__(self):
        super(LimitError, self).__init__("forbidden operation on a limit value")


def _ordpair(v1, v2):
    if v1 > v2:
        return (v2, v1)
    return (v1, v2)


def _lval_add_towards_polarity(x, polarity):
    """Compute the appropriate Lval "kind" for the limit of value `x` towards
    `polarity`. Either 'toinf' or 'pastzero' depending on the sign of `x` and
    the infinity direction of polarity.

    """
    if x < 0:
        if polarity < 0:
            return Lval("toinf", x)
        return Lval("pastzero", x)
    elif polarity > 0:
        return Lval("toinf", x)
    return Lval("pastzero", x)


class Lval(object):
    """A container for either precise values or upper/lower limits. Constructed as
    ``Lval(kind, value)``, where *kind* is ``"exact"``, ``"uncertain"``,
    ``"toinf"``, ``"tozero"``, ``"pastzero"``, or ``"undef"``. Most easily
    constructed via :meth:`Textual.parse`. Can also be constructed with
    :meth:`Lval.from_other`.

    Supported operations are
    ``unicode() str() repr() -(neg) abs() + - * / ** += -= *= /= **=``.

    """

    __slots__ = ("kind", "value")

    def __init__(self, kind, value):
        if kind not in _lval_pos_sigils:
            raise ValueError("unrecognized Lval kind %r" % kind)
        if not np.isscalar(value):
            raise ValueError("Lvals must be scalars; got %r" % value)
        self.kind = kind
        self.value = value

    @staticmethod
    def from_other(o):
        if isinstance(o, Lval):
            from copy import copy

            return Lval(o.kind, copy(o.value))
        if isinstance(o, Uval):
            return Lval("uncertain", o.repvals(uval_default_repval_method)[0])
        if np.isscalar(o):
            return Lval("exact", float(o))
        raise ValueError("cannot convert %r to an Lval" % o)

    # Textualization.

    def __unicode__(self):
        s = _lval_pos_sigils[self.kind]
        if self.value < 0:
            if s == ">":
                s = "<"
            else:
                s = s.replace("<", ">")
        return "%s%g" % (s, self.value)

    __str__ = unicode_to_str

    def __repr__(self):
        return "Lval(%r, %r)" % (self.kind, self.value)

    def __pk_fmtinfo__(self):
        # Only certain kinds of Lval can successfully be roundtripped through
        # text. Positive 'tozero' values need the 'P' flag; negative tozeros
        # are inexpressible.
        if self.kind == "undef" or (self.kind == "tozero" and self.value < 0):
            raise ValueError("no fmtinfo textualization of %r is possible" % self)

        if self.kind == "tozero":
            return "Pu", "<%g" % self.value, True

        s = _lval_pos_sigils[self.kind][0]  # note: truncating << pastzero mode.
        if self.value < 0:
            if s == ">":  # toinf, but we're negative.
                s = "<"
            else:  # tozero disallowed, so we must be pastzero
                s = ">"

        return "u", "%s%g" % (s, self.value), True

    def __pk_latex__(self, undefok=False, **kwargs):
        from .latex import latexify

        base = latexify(self.value, **kwargs)

        if self.kind == "undef":
            if undefok:
                return b""
            raise ValueError("tried to LaTeXify undefined Lval")

        if self.kind == "exact":
            return b"" + base

        if self.kind == "uncertain":
            return b"$\\sim$" + base

        s = _lval_pos_sigils[self.kind][0]  # note: truncating << pastzero mode.
        if self.value < 0:
            if s == ">":
                s = "<"
            else:
                s = ">"

        return b"$%s$%s" % (s, base)

    def __pk_latex_l3col__(self, undefok=False, **kwargs):
        from .latex import latexify_n2col

        base = latexify_n2col(self.value, **kwargs)

        if self.kind == "undef":
            if undefok:
                return b" & & "
            raise ValueError("tried to LaTeXify undefined Lval")

        if self.kind == "exact":
            return b"& " + base

        if self.kind == "uncertain":
            return b"$\\sim$ & " + base

        s = _lval_pos_sigils[self.kind][0]  # note: truncating << pastzero mode.
        if self.value < 0:
            if s == ">":
                s = "<"
            else:
                s = ">"

        return b"$%s$ & %s" % (s, base)

    def __pk_latex_u3col__(self, **kwargs):
        return rb"\multicolumn{3}{c}{%s}" % self.__pk_latex__(**kwargs)

    # Math. We start with addition. It gets complicated!

    def __neg__(self):
        return _lval_unary_negative(self)

    def __abs__(self):
        return _lval_unary_absolute(self)

    def _polarity(self):
        # -2  --  limit towards -infinity
        # -1  --  limit from a negative value to zero
        # 0   --  not a limit
        # +1  --  limit from a positive value to zero
        # +2  --  limit towards +infinity

        assert self.kind != "undef"

        if self.kind in ("uncertain", "exact"):
            return 0

        if self.value < 0:
            if self.kind == "toinf":
                return -2
            if self.kind == "tozero":
                return -1
            return +2

        if self.kind == "toinf":
            return +2
        if self.kind == "tozero":
            return +1
        return -2

    def __add__(self, other):
        v1 = self
        v2 = Lval.from_other(other)
        tot = v1.value + v2.value

        # Rule 1: undef trumps all.
        if v1.kind == "undef" or v2.kind == "undef":
            return Lval("undef", tot)

        # Rule(s) 2: some combinations with exact/uncert values require no
        # checking of the kind or polarity.
        k = _lval_kmap_add_unconditional.get(_ordpair(v1.kind, v2.kind))
        if k is not None:
            return Lval(k, tot)

        # Rule 3: if values have same sign and same kind, we can add
        # without needing to worry about changing the kind.
        s1, s2 = np.sign(v1.value), np.sign(v2.value)
        if s1 == 0:
            s1 = 1.0
        if s2 == 0:
            s2 = 1.0

        if s1 == s2 and v1.kind == v2.kind:
            return Lval(v1.kind, tot)

        # Undefs and exact-ish pairs were dealt with in Rules 1 and 2; to-zero
        # and exact-ish were dealt with in Rule 2, and same-sign-same-kind was
        # dealt with in Rule 3. Therefore if we have two to-zeros, they must
        # be of opposite polarity. Rule 4: this goes to undef.

        p1, p2 = v1._polarity(), v2._polarity()

        if max(abs(p1), abs(p2)) == 1:
            assert p1 == -p2
            return Lval("undef", tot)

        # The only remaining possibility is a combination of a limit to
        # infinity and something else. Make sure that p1 holds an infinity
        # limit.

        if abs(p1) < 2:
            v1, v2 = v2, v1
            s1, s2 = s2, s1
            p1, p2 = p2, p1

        # Rule 5: to-infs of opposite signs go to undef.
        if p2 == -p1:
            return Lval("undef", tot)

        # Rule 6: to-infs of same sign are add-and-normalize.
        if p1 == p2:
            return _lval_add_towards_polarity(tot, p1)

        # Rule 7: to-inf and to-zero of same polarity give the to-inf.
        # Rule 8: to-inf and to-zero of opposite polarity are add-and-normalize.
        if np.abs(p2) == 1:
            if p1 * p2 > 0:
                return v1
            return _lval_add_towards_polarity(tot, p1)

        # Rule 9: to-inf and constant-ish are add-and-normalize.
        if p2 == 0:
            return _lval_add_towards_polarity(tot, p1)

        assert False, "not reached"

    __radd__ = __add__

    def __sub__(self, other):
        other = Lval.from_other(other)
        other = -other
        return self + other

    def __rsub__(self, other):
        other = Lval.from_other(other)
        return other + -self

    def __iadd__(self, other):
        other = Lval.from_other(other)
        tmp = self + other
        self.kind, self.value = tmp.kind, tmp.value
        return self

    def __isub__(self, other):
        other = Lval.from_other(other)
        other = -other
        tmp = self + other
        self.kind, self.value = tmp.kind, tmp.value
        return self

    # Multiplication!

    def __mul__(self, other):
        v1 = self
        v2 = Lval.from_other(other)
        negative = False

        if v1.value < 0:
            negative = not negative
            v1 = -v1

        if v2.value < 0:
            negative = not negative
            v2 = -v2

        prod = v1.value * v2.value

        if v1.kind == "undef" or v2.kind == "undef":
            return Lval("undef", prod)

        ordkind = _ordpair(v1.kind, v2.kind)

        if ordkind == ("toinf", "tozero"):
            rv = Lval("toinf", 0.0)
        else:
            rv = Lval(_lval_kmap_mul[ordkind], prod)

        if negative:
            return -rv
        return rv

    __rmul__ = __mul__

    def __div__(self, other):
        other = Lval.from_other(other)
        other = _lval_unary_reciprocal(other)
        return self * other

    def __rdiv__(self, other):
        other = Lval.from_other(other)
        tmp = _lval_unary_reciprocal(self)
        return other * tmp

    __truediv__ = __div__

    __rtruediv__ = __rdiv__

    def __imul__(self, other):
        other = Lval.from_other(other)
        tmp = self * other
        self.kind, self.value = tmp.kind, tmp.value
        return self

    def __idiv__(self, other):
        other = Lval.from_other(other)
        other = _lval_unary_reciprocal(other)
        tmp = self * other
        self.kind, self.value = tmp.kind, tmp.value
        return self

    __itruediv__ = __idiv__

    # Exponentiation.

    def __pow__(self, other, modulo=None):
        if modulo is not None:
            raise ValueError("powmod behavior forbidden with Lvals")

        try:
            v = float(other)
        except TypeError:
            raise ValueError("Lvals can only be exponentiated by exact values")

        if self.kind == "undef":
            # It's not worth trying to get a reasonable value in this case.
            return Lval("undef", np.nan)

        if v == 0:
            return Lval("exact", 1.0)

        reciprocate = v < 0
        if reciprocate:
            v = -v

        i = int(v)

        if v != i:
            # For us, fractional powers are only defined on positive numbers,
            # which gives us a fairly small number of valid cases to worry
            # about.
            if self.value <= 0 or self.kind == "pastzero":
                return Lval("undef", np.nan)
            rv = Lval(self.kind, self.value**v)
        else:
            # We can deal with integer exponentiation as a series of
            # multiplies. Not the most efficient, but reduces the chance for
            # bugs.
            rv = Lval.from_other(self)
            for _ in range(i - 1):
                rv *= self

        if reciprocate:
            rv = _lval_unary_reciprocal(rv)
        return rv

    def __rpow__(self, other, modulo=None):
        if modulo is not None:
            raise ValueError("powmod behavior forbidden with Lvals")

        if self.kind == "undef":
            return Lval("undef", np.nan)

        if self.kind == "exact":
            # In this very special case, we can delegate.
            return other**self.value

        # In all other cases, we're exponentiating by a fractional value,
        # so we're only valid for nonnegative numbers.

        try:
            v = float(other)
        except TypeError:
            raise ValueError("Lvals can only exponentiate exact values")

        if v < 0:
            raise ValueError("Lvals can only exponentiate nonnegative values")

        reciprocate = self.value < 0
        exponent = self  # NOTE: no use of 'self' from here on out!
        if reciprocate:
            exponent = -exponent

        if v == 0:
            if exponent.kind == "pastzero":
                return Lval("undef", np.nan)
            # We ignore the fact that 0**0 = 1.
            rv = Lval("exact", 0.0)
        elif v < 1:
            k = _lval_kmap_pow_zero_to_one.get(exponent.kind, exponent.kind)
            rv = Lval(k, v**exponent.value)
        else:
            k = _lval_kmap_pow_above_one.get(exponent.kind, exponent.kind)
            rv = Lval(k, v**exponent.value)

        if reciprocate:
            rv = _lval_unary_reciprocal(rv)
        return rv

    def __ipow__(self, other, modulo=None):
        tmp = pow(self, other, modulo)
        self.kind, self.value = tmp.kind, tmp.value
        return self

    __hash__ = None


def _make_lval_unary_math_nolimits(scalarfunc):
    def lval_unary_math_nolimits(v):
        v = Lval.from_other(v)
        if v.kind == "upper" or v.kind == "lower":
            raise LimitError()
        return Lval(v.kind, scalarfunc(value))

    return lval_unary_math_nolimits


def _lval_unary_absolute(v):
    v = Lval.from_other(v)

    if v.kind == "pastzero":
        return Lval("toinf", 0.0)  # can't argue with this!
    return Lval(v.kind, abs(v.value))


def _lval_unary_exp(v):
    v = Lval.from_other(v)

    reciprocate = v.value < 0
    if reciprocate:
        v = -v

    rv = Lval(_lval_kmap_exp.get(v.kind, v.kind), np.exp(v.value))
    if reciprocate:
        rv = _lval_unary_reciprocal(rv)
    return rv


def _lval_unary_isfinite(v):
    v = Lval.from_other(v)

    if v.kind == "undef":
        return False
    return np.isfinite(v.value)


def _make_lval_unary_log(scalarfunc):
    def lval_unary_log(v):
        v = Lval.from_other(v)
        if v.value <= 0 or v.kind in ("undef", "pastzero"):
            return Lval("undef", np.nan)
        return Lval(_lval_kmap_log.get(v.kind, v.kind), scalarfunc(v.value))

    return lval_unary_log


def _lval_unary_negative(v):
    # In this convenient case, the `kind` doesn't change.
    v = Lval.from_other(v)
    return Lval(v.kind, -v.value)


def _lval_unary_reciprocal(v):
    v = Lval.from_other(v)
    if v.value == 0:
        return Lval("undef", np.nan)
    return Lval(_lval_kmap_reciprocal.get(v.kind, v.kind), 1.0 / v.value)


def _lval_unary_sqrt(v):
    v = Lval.from_other(v)
    return v**0.5


def _lval_unary_square(v):
    v = Lval.from_other(v)
    return v * v


lval_unary_math = {
    # The 'nolimits' entries could be improved with special-cased
    # implementations, but I'm not going to write them until the need arises.
    "absolute": _lval_unary_absolute,
    "arccos": _make_lval_unary_math_nolimits(np.arccos),
    "arcsin": _make_lval_unary_math_nolimits(np.arcsin),
    "arctan": _make_lval_unary_math_nolimits(np.arctan),
    "cos": _make_lval_unary_math_nolimits(np.cos),
    "expm1": _make_lval_unary_math_nolimits(np.expm1),
    "exp": _lval_unary_exp,
    "isfinite": _lval_unary_isfinite,
    "log10": _make_lval_unary_log(np.log10),
    "log1p": _make_lval_unary_math_nolimits(np.log1p),
    "log2": _make_lval_unary_log(np.log2),
    "log": _make_lval_unary_log(np.log),
    "negative": _lval_unary_negative,
    "reciprocal": _lval_unary_reciprocal,
    "sin": _make_lval_unary_math_nolimits(np.sin),
    "sqrt": _lval_unary_sqrt,
    "square": _lval_unary_square,
    "tan": _make_lval_unary_math_nolimits(np.tan),
}


# Finally, measurements represented in textual form. Textual forms can
# represent exact values, Uvals, or Lvals.

UQUANT_UNCERT = 0.2
"""Some values are known to be uncertain, but their uncertainties have not been
quantified. This is lame but it happens. In this case, assume a 20%
uncertainty.

We could infer uncertainties from the number of written digits: i.e., assuming
"1.2" is uncertain by 0.05 or so, while "1.2000" is uncertain by 0.00005 or
so. But there are many cases in astronomy where people just list values that
are 20% uncertain and give them to multiple decimal places. I'd rather be
conservative with these values than overly optimistic.

Code to do the appropriate parsing is in the Python uncertainties package, in
its __init__.py:parse_error_in_parentheses().

"""
_tkinds = frozenset(("none", "log10", "positive"))
_dkinds = frozenset(("exact", "symm", "asymm", "uncertain", "upper", "lower", "unif"))
_noextra_dkinds = frozenset(("exact", "uncertain", "upper", "lower"))
_yesextra_dkinds = frozenset(("symm", "asymm"))


def _split_decimal_col(floattext):
    if "." not in floattext:
        return "$%s$ & " % floattext
    return "$%s$ & $.%s$ " % tuple(floattext.split("."))


class Textual(object):
    """A measurement recorded in textual form.

    Textual.from_exact(text, tkind='none') - `text` is passed to float()
    Textual.parse(text, tkind='none')      - `text` as described below.

    Transformation kinds are 'none', 'log10', or 'positive'. Expressions for
    values take the form '1.234', '<2', '>3', '~7', '6to8', '7pm0.1', or
    '12p1m0.3'.

    Methods:

    unparse()              - Return parsed text (but not tkind!)
    unwrap()               - Express as float/Uval/Lval as appropriate.
    repval(limitsok=False) - Get single scalar "representative" value.
    limtype()              - -1 if upper limit; +1 if lower; 0 otherwise.

    Supported operations: unicode() str() repr() [latexification] -(neg) abs()
    + - * / **

    """

    __slots__ = ("tkind", "dkind", "data")

    def __init__(self, tkind, dkind, data):
        if tkind not in _tkinds:
            raise ValueError('unrecognized transformation kind "%s"' % tkind)
        if dkind not in _dkinds:
            raise ValueError('unrecognized distribution kind "%s"' % dkind)
        # FIXME: could/should check `data`.

        self.tkind = tkind
        self.dkind = dkind
        self.data = data

    @staticmethod
    def from_exact(text, tkind="none"):
        float(text)  # check float-parseability.
        return Textual(tkind, "exact", text)

    @staticmethod
    def parse(text, tkind="none"):
        # freestanding float() calls below are used to check
        # float-parseability of strings.
        # XXX: we do not check sanity when tkind is 'positive'!

        if text[0] == "~":
            dkind = "uncertain"
            data = text[1:]
            float(data)
        elif text[0] == "<":
            dkind = "upper"
            data = text[1:]
            float(data)
        elif text[0] == ">":
            dkind = "lower"
            data = text[1:]
            float(data)
        elif "to" in text:
            lower, upper = text.split("to")
            f_lower = float(lower)
            f_upper = float(upper)

            if f_lower > f_upper:
                upper, lower = lower, upper
                f_upper, f_lower = f_lower, f_upper

            if f_lower < 0 and tkind == "positive":
                raise ValueError(
                    "uniform interval is forced positive, but " 'got "%s"' % text
                )

            dkind = "unif"
            data = (lower, upper)
        elif "pm" in text:
            val, uncert = text.split("pm")
            float(val)
            f_uncert = float(uncert)
            if f_uncert <= 0.0:
                raise ValueError('uncertainty values must be positive; got "%s"' % text)

            dkind = "symm"
            data = (val, uncert)
        elif "p" in text:
            val, rhs = text.split("p", 1)
            high, low = rhs.split("m", 1)
            float(val)  # checks parseability
            f_high = float(high)
            f_low = float(low)

            if f_high <= 0:
                raise ValueError("asymmetrical upper uncertainty must be positive")
            if f_low <= 0:
                raise ValueError("asymmetrical lower uncertainty must be positive")

            dkind = "asymm"
            data = (val, high, low)
        else:
            try:  # plain float treated as unquantified
                dkind = "uncertain"
                data = text
                float(data)
            except ValueError:
                raise ValueError("don't know how to parse measurement text: %s" % text)

        return Textual(tkind, dkind, data)

    # Textualization -- keep this up here since this is so closely tied to
    # construction via parse(). Note that unparse() loses the `tkind` info.

    def unparse(self):
        if self.dkind == "exact":
            return self.data
        elif self.dkind == "uncertain":
            return "~" + self.data
        elif self.dkind == "symm":
            return self.data[0] + "pm" + self.data[1]
        elif self.dkind == "asymm":
            return self.data[0] + "p" + self.data[1] + "m" + self.data[2]
        elif self.dkind == "upper":
            return "<" + self.data
        elif self.dkind == "lower":
            return ">" + self.data
        elif self.dkind == "unif":
            return self.data[0] + "to" + self.data[1]

    def __repr__(self):
        if self.tkind == "none":
            ttext = ""
        else:
            ttext = ", %r" % (self.tkind,)

        if self.dkind == "exact":
            return "Textual.from_exact (%r%s)" % (self.data, ttext)
        return "Textual.parse(%r%s)" % (self.unparse(), ttext)

    def __unicode__(self):
        if self.tkind == "none":
            return self.unparse()
        return self.unparse() + ":" + self.tkind

    __str__ = unicode_to_str

    def __pk_fmtinfo__(self):
        t = self.unparse()

        if self.tkind == "log10":
            ttag = "L"
        elif self.tkind == "positive":
            ttag = "P"
        else:
            ttag = ""

        if self.dkind == "exact":
            dtag = "f"
        else:
            dtag = "u"

        return ttag + dtag, t, False

    # "Unwrapping" -- conversion into either a scalar, Uval, or Lval. The
    # ability to apply various data transforms complicates this process.

    def _unwrap_pos(self):
        dkind = self.dkind

        # Deal with the easy cases ...

        if dkind == "exact":
            return float(self.data)

        if dkind == "upper":
            # Important case here: since we're positivized, the appropriate
            # Lval kind is 'tozero' rather than 'pastzero'. This allows
            # the caller to safely take the log or the reciprocal.
            return Lval("tozero", float(self.data))

        if dkind == "lower":
            return Lval("toinf", float(self.data))

        if dkind == "unif":
            # Limits should have been checked upon construction.
            lower, upper = map(float, self.data)
            return Uval.from_unif(lower, upper)

        # We have to get careful with the Uvals.

        if dkind == "symm":
            val = float(self.data[0])
            uncert = float(self.data[1])
            v = Uval.from_norm(val, uncert)
        elif dkind == "uncertain":
            val = float(self.data)
            uncert = UQUANT_UNCERT * abs(val)
            v = Uval.from_norm(val, uncert)
        elif dkind == "asymm":
            val, dhigh, dlow = map(float, self.data)
            v = Uval.from_double_norm(val, dhigh, dlow)

        nnonpos = np.where(v.d <= 0)[0].size

        if nnonpos == 0:
            return v

        if nnonpos < 3:
            # Yay arbitrary cutoff. This is close enough to
            # positive that we just force things.
            v.d = np.abs(v.d)
            return v

        # There are enough negative values that we're not comfortable with
        # drawing from a (double) normal distribution. Draw from a gamma
        # distribution instead.

        if dkind == "asymm":
            # The gamma distribution only has two parameters, so what else can
            # we do?
            uncert = 0.5 * (dhigh + dlow)

        alpha, beta = find_gamma_params(val, uncert)
        return Uval.from_gamma(alpha, beta)

    def _unwrap_log(self):
        dkind = self.dkind

        if dkind == "exact":
            return 10 ** float(self.data)

        if dkind == "upper":
            # As with positive Textuals, it's important that we can return
            # a tozero limit here.
            return Lval("tozero", 10 ** float(self.data))

        if dkind == "lower":
            return Lval("toinf", 10 ** float(self.data))

        if dkind == "uncertain":
            # Assume UQUANT_UNCERT in (10**x), not in x itself
            val = 10 ** float(self.data)
            return Uval.from_norm(val, UQUANT_UNCERT * abs(val))

        if dkind == "unif":
            # We'll yield a uniform distribution in log10(x), not x. I think
            # this is more desirable if someone writes "foo:Lu = 3.5to4.5".
            lower, upper = map(float, self.data)
            return 10 ** Uval.from_unif(lower, upper)

        if dkind == "symm":
            val = float(self.data[0])
            uncert = float(self.data[1])
            return 10 ** Uval.from_norm(val, uncert)

        assert dkind == "asymm"
        val, dhigh, dlow = map(float, self.data)
        return 10 ** Uval.from_double_norm(val, dhigh, dlow)

    def unwrap(self):
        if self.tkind == "log10":
            return self._unwrap_log()
        if self.tkind == "positive":
            return self._unwrap_pos()

        # No transformations applied:
        dkind = self.dkind

        if dkind == "exact":
            return float(self.data)

        if dkind == "upper":
            # Limits of magnitude-type quantities should always be of tkind
            # 'log10' or 'positive', so that we can return a 'tozero' Lval
            # rather than 'pastzero'. This is important for taking reciprocals
            # and/or logarithms.
            v = float(self.data)
            if v < 0:
                return Lval("toinf", v)
            return Lval("pastzero", v)

        if dkind == "lower":
            v = float(self.data)
            if v < 0:
                return Lval("pastzero", v)
            return Lval("toinf", v)

        if dkind == "unif":
            lower, upper = map(float, self.data)
            return Uval.from_unif(lower, upper)

        if dkind == "uncertain":
            val = float(self.data)
            return Uval.from_norm(val, UQUANT_UNCERT * abs(val))

        if dkind == "symm":
            val = float(self.data[0])
            uncert = float(self.data[1])
            return Uval.from_norm(val, uncert)

        assert dkind == "asymm"
        val, dhigh, dlow = map(float, self.data)
        return Uval.from_double_norm(val, dhigh, dlow)

    # Other numerical helpers.

    def repval(self, limitsok=False):
        """Get a best-effort representative value as a float. This can be
        DANGEROUS because it discards limit information, which is rarely wise."""

        if not limitsok and self.dkind in ("lower", "upper"):
            raise LimitError()

        if self.dkind == "unif":
            lower, upper = map(float, self.data)
            v = 0.5 * (lower + upper)
        elif self.dkind in _noextra_dkinds:
            v = float(self.data)
        elif self.dkind in _yesextra_dkinds:
            v = float(self.data[0])
        else:
            raise RuntimeError("can't happen")

        if self.tkind == "log10":
            return 10**v
        return v

    def limtype(self):
        """Return -1 if this value is an upper limit, 1 if it is a lower
        limit, 0 otherwise."""

        if self.dkind == "upper":
            return -1
        if self.dkind == "lower":
            return 1
        return 0

    # Latexification.

    def __pk_latex__(self):
        if self.dkind == "exact":
            return r"$%s$" % self.data
        if self.dkind == "uncertain":
            return r"$\sim$$%s$" % self.data
        if self.dkind == "symm":
            return r"$%s \pm %s$" % self.data
        if self.dkind == "asymm":
            return r"$%s^{+%s}_{-%s}$" % self.data
        if self.dkind == "upper":
            return r"$<$$%s$" % self.data
        if self.dkind == "lower":
            return r"$>$$%s$" % self.data
        if self.dkind == "unif":
            return r"$%s$--$%s$" % self.data

    def __pk_latex_u3col__(self):
        if self.dkind == "exact":
            return r"\multicolumn{3}{c}{$%s$}" % self.data
        if self.dkind == "uncertain":
            return r"\multicolumn{3}{c}{$\sim$$%s$}" % self.data
        if self.dkind == "symm":
            return r"%s & $\pm\,%s$" % (_split_decimal_col(self.data[0]), self.data[1])
        if self.dkind == "asymm":
            return r"%s & $\pm\,^{%s}_{%s}$" % (
                _split_decimal_col(self.data[0]),
                self.data[1],
                self.data[2],
            )
        if self.dkind == "upper":
            return r"\multicolumn{3}{c}{$<$$%s$}" % self.data
        if self.dkind == "lower":
            return r"\multicolumn{3}{c}{$>$$%s$}" % self.data
        if self.dkind == "unif":
            return r"\multicolumn{3}{c}{$%s$--$%s$}" % self.data

    # Unary math -- we do the same thing as _make_textual_unary_math_generic()
    # below. Unlike Uval and Lval, algebra on Textuals is emphatically not
    # closed -- the result is always a non-Textual.

    def __neg__(self):
        return _dispatch_unary_math("negative", False, self.unwrap())

    def __abs__(self):
        return _dispatch_unary_math("absolute", False, self.unwrap())

    # Binary math -- we delegate to the functions that are defined below.

    def __add__(self, other):
        return add(self, other)

    def __sub__(self, other):
        return subtract(self, other)

    def __mul__(self, other):
        return multiply(self, other)

    def __div__(self, other):
        return divide(self, other)

    def __truediv__(self, other):
        return true_divide(self, other)

    def __pow__(self, other, module=None):
        if modulo is not None:
            raise ValueError("powmod behavior forbidden with Textuals")
        return power(self, other)

    def __radd__(self, other):
        return add(other, self)

    def __rsub__(self, other):
        return subtract(other, self)

    def __rmul__(self, other):
        return multiply(other, self)

    def __rdiv__(self, other):
        return divide(other, self)

    def __rtruediv__(self, other):
        return true_divide(other, self)

    def __rpow__(self, other, module=None):
        if modulo is not None:
            raise ValueError("powmod behavior forbidden with Textuals")
        return power(other, self)


def _dispatch_unary_math(name, check_textual, value):
    if np.isscalar(value):
        table = scalar_unary_math
    elif isinstance(value, Uval):
        table = uval_unary_math
    elif isinstance(value, Lval):
        table = lval_unary_math
    elif check_textual and isinstance(value, Textual):
        table = textual_unary_math
    else:
        raise ValueError("cannot treat %r as a scalar for %s" % (value, name))

    func = table.get(name)
    if func is None:
        raise ValueError("no implementation of %s for %r" % (name, value))
    return func(value)


def _make_textual_unary_math_generic(name):
    # N.B.: unlike Uval and Lval functions, this assumes that `v` is in fact a
    # Textual.
    def textual_unary_math_generic(val):
        return _dispatch_unary_math(name, False, val.unwrap())

    return textual_unary_math_generic


def _textual_unary_log10(val):
    if val.tkind == "log10":
        return Textual("none", val.dkind, val.data)
    return _dispatch_unary_math("log10", False, val.unwrap())


textual_unary_math = {
    "absolute": _make_textual_unary_math_generic("absolute"),
    "arccos": _make_textual_unary_math_generic("arccos"),
    "arcsin": _make_textual_unary_math_generic("arcsin"),
    "arctan": _make_textual_unary_math_generic("arctan"),
    "cos": _make_textual_unary_math_generic("cos"),
    "expm1": _make_textual_unary_math_generic("expm1"),
    "exp": _make_textual_unary_math_generic("exp"),
    "isfinite": lambda v: True,  # legal Textuals can never yield inf or nan
    "log10": _textual_unary_log10,
    "log1p": _make_textual_unary_math_generic("log1p"),
    "log2": _make_textual_unary_math_generic("log2"),
    "log": _make_textual_unary_math_generic("log"),
    "negative": _make_textual_unary_math_generic("negative"),
    "reciprocal": _make_textual_unary_math_generic("reciprocal"),
    "sin": _make_textual_unary_math_generic("sin"),
    "sqrt": _make_textual_unary_math_generic("sqrt"),
    "square": _make_textual_unary_math_generic("square"),
    "tan": _make_textual_unary_math_generic("tan"),
}


# Now, a library of metadata-esque functions that will handle anything you
# throw at them: scalars, Uvals, Lvals, and Textuals.


def is_measurement(obj):
    return np.isscalar(obj) or isinstance(obj, (Uval, Lval, Textual))


def unwrap(msmt):
    """Convert the value into the most basic representation that we can do
    math on: float if possible, then Uval, then Lval."""

    if np.isscalar(msmt):
        return float(msmt)
    if isinstance(msmt, (Uval, Lval)):
        return msmt
    if isinstance(msmt, Textual):
        return msmt.unwrap()
    raise ValueError("don't know how to treat %r as a measurement" % msmt)


def typealign(origmsmt1, origmsmt2):
    msmt1 = origmsmt1
    if isinstance(msmt1, Textual):
        msmt1 = msmt1.unwrap()
    msmt2 = origmsmt2
    if isinstance(msmt2, Textual):
        msmt2 = msmt2.unwrap()

    if isinstance(msmt1, Lval):
        return msmt1, Lval.from_other(msmt2)
    if isinstance(msmt2, Lval):
        return Lval.from_other(msmt1), msmt2
    if isinstance(msmt1, Uval):
        return msmt1, Uval.from_other(msmt2)
    if isinstance(msmt2, Uval):
        return Uval.from_other(msmt1), msmt2

    try:
        return float(msmt1), float(msmt2)
    except Exception:
        raise ValueError(
            "cannot treat %r and %r as numeric types" % (origmsmt1, origmsmt2)
        )


def repval(msmt, limitsok=False):
    """Get a best-effort representative value as a float. This is DANGEROUS
    because it discards limit information, which is rarely wise. m_liminfo()
    or m_unwrap() are recommended instead."""

    if np.isscalar(msmt):
        return float(msmt)
    if isinstance(msmt, Uval):
        return msmt.repvals(uval_default_repval_method)[0]
    if isinstance(msmt, Lval):
        if not limitsok and msmt.kind in ("tozero", "toinf", "pastzero"):
            raise LimitError()
        return msmt.value
    if isinstance(msmt, Textual):
        return msmt.repval(limitsok=limitsok)

    raise ValueError("don't know how to treat %r as a measurement" % msmt)


def limtype(msmt):
    """Return -1 if this value is some kind of upper limit, 1 if this value
    is some kind of lower limit, 0 otherwise."""

    if np.isscalar(msmt):
        return 0
    if isinstance(msmt, Uval):
        return 0
    if isinstance(msmt, Lval):
        if msmt.kind == "undef":
            raise ValueError("no simple limit type for Lval %r" % msmt)

        # Quasi-hack here: limits of ('tozero', [positive number]) are
        # reported as upper limits. In a plot full of fluxes this would be
        # what makes sense, but note that this would be misleading if the
        # quantity in question was something that could go negative.
        p = msmt._polarity()
        if p == -2 or p == 1:
            return -1
        if p == 2 or p == -1:
            return 1
        return 0
    if isinstance(msmt, Textual):
        return msmt.limtype()
    raise ValueError("don't know how to treat %r as a measurement" % msmt)


def liminfo(msmt):
    """Return (limtype, repval). `limtype` is -1 for upper limits, 1 for lower
    limits, and 0 otherwise; repval is a best-effort representative scalar value
    for this measurement."""

    return limtype(msmt), repval(msmt, limitsok=True)


def errinfo(msmt):
    """Return (limtype, repval, errval1, errval2). Like m_liminfo, but also
    provides error bar information for values that have it."""

    if isinstance(msmt, Textual):
        msmt = msmt.unwrap()

    if np.isscalar(msmt):
        return 0, msmt, msmt, msmt

    if isinstance(msmt, Uval):
        rep, plus1, minus1 = msmt.repvals(uval_default_repval_method)
        return 0, rep, plus1, minus1

    if isinstance(msmt, Lval):
        return limtype(msmt), msmt.value, msmt.value, msmt.value

    raise ValueError("don't know how to treat %r as a measurement" % msmt)


# Unary numerical functions.
#
# Here we just have to look up the appropriate table of unary math operations
# and delegate. The implemented functions are those in the scalar_unary_math
# dict.
#
# Potentially useful Numpy functions that I've skipped:
#
# absolute angle arccosh arcsinh arctanh around ceil clip conj copysign cosh
# deg2rad degrees fabs fix floor frexp hypot i0 imag ldexp modf rad2deg
# radians real rint round_ sign signbit sinc sinh tanh trunc unwrap
#
# Skipped core Python unary operators:
#
# abs coerce complex hex index int invert float long neg nonzero oct pos


def _make_wrapped_unary_math(name):
    def unary_mathfunc(val):
        rv = _dispatch_unary_math(name, True, val)
        if not _dispatch_unary_math("isfinite", True, rv):
            raise ValueError("out-of-bounds input %r to %s" % (val, name))
        return rv

    return unary_mathfunc


def _init_unary_math():
    g = globals()

    for name in scalar_unary_math.keys():
        if name == "isfinite":
            g[name] = lambda v: _dispatch_unary_math("isfinite", True, v)
        else:
            g[name] = _make_wrapped_unary_math(name)


_init_unary_math()


# Binary numerical functions.
#
# Here we have to coerce the arguments to the ~"highest-level" type that's
# relevant. Then we can delegate to the class's __op__ functions. Note that
# there's no reasonable binary operator that can take two Textuals to another
# Textual (unlike the very special case of unary log10() on a Textual), so we
# can unwrap those without loss of information.
#
# Potentially useful Numpy functions that I've skipped:
#
# fmod maximum minimum mod=remainder logaddexp2 logaddexp
#
# Skipped core Python binary operators:
#
# and cmp eq ge gt le lshift lt ne or rshift xor


def _make_wrapped_binary_math(opfunc):
    def binary_mathfunc(val1, val2):
        a1, a2 = typealign(val1, val2)
        return opfunc(a1, a2)

    return binary_mathfunc


add = _make_wrapped_binary_math(operator.add)
floor_divide = _make_wrapped_binary_math(operator.floordiv)
multiply = _make_wrapped_binary_math(operator.mul)
power = _make_wrapped_binary_math(operator.pow)
subtract = _make_wrapped_binary_math(operator.sub)
true_divide = _make_wrapped_binary_math(operator.truediv)
divide = true_divide  # are we supposed to respect Py2 plain-div semantics?


# Parsing and formatting of measurements and other quantities.


def _parse_bool(text):
    if not len(text):
        return False
    if text == "y":
        return True
    raise ValueError(
        'illegal bool textualization: expect empty or "y"; ' 'got "%s"' % text
    )


def _maybe(subparse):
    def parser(text):
        if not len(text):
            return None
        return subparse(text)

    return parser


_ttkinds = {"": "none", "L": "log10", "P": "positive"}


def _maybe_parse_exact(text, tkind):
    if not len(text):
        return None
    return Textual.from_exact(text, _ttkinds[tkind])


def _maybe_parse_uncert(text, tkind):
    if not len(text):
        return None
    return Textual.parse(text, _ttkinds[tkind])


parsers = {
    # maps 'type tag string' to 'parsing function'.
    "x": None,
    "b": _parse_bool,
    "i": _maybe(int),
    "s": _maybe(str),
    "f": lambda t: _maybe_parse_exact(t, ""),
    "Lf": lambda t: _maybe_parse_exact(t, "L"),
    "Pf": lambda t: _maybe_parse_exact(t, "P"),
    "u": lambda t: _maybe_parse_uncert(t, ""),
    "Lu": lambda t: _maybe_parse_uncert(t, "L"),
    "Pu": lambda t: _maybe_parse_uncert(t, "P"),
}


def fmtinfo(value):
    """Returns (typetag, text, is_imprecise). Unlike other functions that operate
    on measurements, this also operates on bools, ints, and strings.

    """
    if value is None:
        raise ValueError("cannot format None!")

    if isinstance(value, str):
        return "", value, False

    if isinstance(value, bool):
        # Note: isinstance(True, int) = True, so this must come before the next case.
        if value:
            return "b", "y", False
        return "b", "", False

    if isinstance(value, int):
        return "i", str(value), False

    if isinstance(value, float):
        return "f", str(value), True

    if hasattr(value, "__pk_fmtinfo__"):
        return value.__pk_fmtinfo__()

    raise ValueError("don't know how to format %r as a measurement" % value)
