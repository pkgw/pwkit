# -*- mode: python; coding: utf-8 -*-
# Copyright 2014-2015 Peter Williams <peter@newton.cx> and collaborators.
# Licensed under the MIT License.

"""
Peter Williams' toolkit for science and astronomy.

Modules in this package:

  (this one)       - Exceptions, Holder class, Python 3 compat help.
  astimage         - Generic I/O interface for astronomical images (FITS, CASA, MIRIAD).
  astutil          - Miscellaneous astronomy-related constants and functions.
  bblocks          - Bayesian Blocks analysis for binning time-tagged events.
  casautil         - Helpers for saner usage of CASA in Python.
  cgs              - Physical constants in CGS.
  cli              - Utilities for command-line programs.
  colormaps        - Mapping scalars into a color palette for visualization.
  contours         - Tracing contours in functions and data.
  data_gui_helpers - Helpers for GUIs for investigating data arrays.
  ellipses         - Computations with ellipses in several parametrizations.
  immodel          - Analytical modeling of astronomical images.
  inifile          - Simple ini-format file parser.
  io               - Utilities for input and output.
  kbn_conf         - Calculate Poisson-like confidence intervals assuming a background.
  kwargv           - Keyword-style argument parsing.
  latex            - Tools for interacting with the LaTeX typesetting system.
  lmmin            - Levenberg-Marquardt least-squares function minimizer.
  lsqmdl           - Model data with least-squares fitting.
  method_decorator - Utility for writing decorators that go on methods in classes.
  msmt             - Framework for working with uncertain measurements.
  ndshow_gtk2      - Visualize data arrays as interactive images, using Gtk+2.
  ndshow_gtk2      - Visualize data arrays as interactive images, using Gtk+3.
  numutil          - Basic NumPy and generic numerical utilities.
  parallel         - Utilities for parallel processing.
  pdm              - Finding periods in data with Phase Dispersion Minimization.
  phoenix          - Working with Phoenix-based model atmospheres.
  radio_cal_models - Models of radio-wavelength calibrator flux densities.
  synphot          - Synthetic photometry and database of instrumental bandpasses.
  tabfile          - I/O on typed tabular files containing uncertain measurements.
  tinifile         - I/O on typed ini-format files containing uncertain measurements.
  ucd_physics      - Estimating physical quantities for M stars and UCDs.
  unicode_to_latex - Rendering Unicode to LaTeX.


Classes in the toplevel module:

  Holder      - A "namespace object" that just lets you assign attributes.
  PKError     - Base exception class for PWKit
  binary_type - The binary data type: either `str` (Python 2.x) or `bytes` (Python 3.x)
  text_type   - The text data type: either `unicode` (Python 2.x) or `str` (Python 3.x)

Functions in the toplevel module:

  reraise_context - Reraise an exception with additional contextual information
                    in the message.
  unicode_to_str  - Write `def __unicode__ (self): ... ; __str__ = unicode_to_str`
                    in classes.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = (b'Holder PKError binary_type reraise_context text_type '
           b'unicode_to_str').split ()

__version__ = '0.4.1' # also edit ../setup.py!

# Lightly-exercised simultaneous Python 2 and 3 compat.
import sys
if sys.version < '3':
    text_type = unicode
    binary_type = str
    unicode_to_str = lambda s: s.__unicode__ ().encode ('utf8')
else:
    text_type = str
    binary_type = bytes
    unicode_to_str = lambda s: s.__unicode__ ()


class PKError (Exception):
    def __init__ (self, fmt, *args):
        if not len (args):
            self.args = (text_type (fmt), )
        else:
            self.args = (text_type (fmt) % args, )

    def __unicode__ (self):
        return self.args[0]

    __str__ = unicode_to_str

    def __repr__ (self):
        return 'PKError(' + repr (self.args[0]) + ')'


def reraise_context (fmt, *args):
    """Reraise an exception with its message modified to specify additional
    context.

    """
    if len (args):
        cstr = fmt % args
    else:
        cstr = text_type (fmt)

    ex = sys.exc_info ()[1]
    if len (ex.args):
        cstr = '%s: %s' % (cstr, ex.args[0])
    ex.args = (cstr, ) + ex.args[1:]
    raise


class Holder (object):
    """A basic 'namespace class' that provides you a place to easily stick named
    data with a minimum of fuss.

    Holder (name1=val1, name2=val2, ...)

    Provides nice str/unicode/repr representations, and basic manipulations:

    set(**kwargs)           - Set named keys as a group
    setone (name, value)    - Set a specific key
    get (name, defval=None) - Retrieve a key with an optional default
    copy ()                 - Make a shallow clone of the Holder

    This class may also be used as a decorator on a class definition to transform
    its contents into a Holder instance. Writing:

        @Holder
        class mydata ():
            a = 1
            b = 'hello'

    creates a Holder instance named 'mydata' containing names 'a' and 'b'.
    This can be a convenient way to populate one-off data structures.

    """
    def __init__ (self, __decorating=None, **kwargs):
        import types

        if __decorating is None:
            values = kwargs
        elif isinstance (__decorating, (type, types.ClassType)):
            # We're decorating a class definition. Transform the definition
            # into a Holder instance thusly:
            values = dict (kv for kv in __decorating.__dict__.iteritems ()
                           if not kv[0].startswith ('__'))
        else:
            # You could imagine allowing @Holder on a function and doing
            # something with its return value, but I can't think of a use that
            # would be more sensible than just creating and returning a Holder
            # directly.
            raise ValueError ('unexpected use of Holder as a decorator (on %r)'
                              % __decorating)

        self.set (**values)

    def __unicode__ (self):
        d = self.__dict__
        s = sorted (d.iterkeys ())
        return '{' + ', '.join ('%s=%s' % (k, d[k]) for k in s) + '}'

    __str__ = unicode_to_str

    def __repr__ (self):
        d = self.__dict__
        s = sorted (d.iterkeys ())
        return b'%s(%s)' % (self.__class__.__name__,
                            b', '.join (b'%s=%r' % (k, d[k]) for k in s))

    def set (self, **kwargs):
        self.__dict__.update (kwargs)
        return self

    def get (self, name, defval=None):
        return self.__dict__.get (name, defval)

    def setone (self, name, value):
        self.__dict__[name] = value
        return self

    def has (self, name):
        return name in self.__dict__

    def copy (self):
        new = self.__class__ ()
        new.__dict__ = dict (self.__dict__)
        return new
