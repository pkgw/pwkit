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
  cgs              - Physical constants in CGS.
  cli              - Utilities for command-line programs.
  colormaps        - Mapping scalars into a color palette for visualization.
  contours         - Tracing contours in functions and data.
  data_gui_helpers - Helpers for GUIs for investigating data arrays.
  ellipses         - Computations with ellipses in several parametrizations.
  environments     - Interfacing with external software environments (SAS, CIAO, etc).
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
  slurp            - Streaming output from sub-programs.
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

# In Python 2, the items in __all__ should be bytes strings. In Python 3, they
# should be Unicode. (http://stackoverflow.com/a/19913680/3760486) If you
# don't use __future__.unicode_literals, you can just write `__all__ =
# ["foo"]` and it's fine, but we do, which causes problems on Py 2.
# Fortunately, to work on both cases we just need to do this:
__all__ = str ('''Holder PKError binary_type reraise_context text_type unicode_to_str''').split ()

__version__ = '0.6.99' # also edit ../setup.py, ../docs/source/conf.py!

# Simultaneous Python 2/3 compatibility through the 'six' module. I started
# out hoping that I could do this all "in-house" without adding the dep, but
# it became clear that 'six' was going to end up being helpful.

import six
from six import binary_type, text_type

if six.PY2:
    unicode_to_str = lambda s: s.__unicode__ ().encode ('utf8')
else:
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
    import sys

    if len (args):
        cstr = fmt % args
    else:
        cstr = text_type (fmt)

    ex = sys.exc_info ()[1]

    if isinstance (ex, EnvironmentError):
        ex.strerror = '%s: %s' % (cstr, ex.strerror)
        ex.args = (ex.errno, ex.strerror)
    else:
        if len (ex.args):
            cstr = '%s: %s' % (cstr, ex.args[0])
        ex.args = (cstr, ) + ex.args[1:]

    raise


class Holder (object):
    """A basic 'namespace class' that provides you a place to easily stick named
    data with a minimum of fuss.

    Holder (name1=val1, name2=val2, ...)

    Provides nice str/unicode/repr representations, and basic manipulations:

    set(**kwargs)            - Set named keys as a group
    set_one (name, value)    - Set a specific key
    get (name, defval=None)  - Retrieve a key with an optional default
    has (name)               - Test whether a key is in the Holder
                               (can also test `name in holderobj`)
    copy ()                  - Make a shallow clone of the Holder
    to_dict ()               - Return a copy of the contents as a dict
    to_pretty (format='str') - Return aligned, multi-line stringification

    Iterating over a Holder yields its contents in the form of a sequence of
    (name, value) tuples.

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
        elif isinstance (__decorating, six.class_types):
            # We're decorating a class definition. Transform the definition
            # into a Holder instance thusly:
            values = dict (kv for kv in six.iteritems (__decorating.__dict__)
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
        s = sorted (six.iterkeys (d))
        return '{' + ', '.join ('%s=%s' % (k, d[k]) for k in s) + '}'

    __str__ = unicode_to_str

    def __repr__ (self):
        d = self.__dict__
        s = sorted (six.iterkeys (d))
        return b'%s(%s)' % (self.__class__.__name__,
                            b', '.join (b'%s=%r' % (k, d[k]) for k in s))

    def __iter__ (self):
        return six.iteritems (self.__dict__)

    def __contains__ (self, key):
        return key in self.__dict__

    def set (self, **kwargs):
        self.__dict__.update (kwargs)
        return self

    def get (self, name, defval=None):
        return self.__dict__.get (name, defval)

    def set_one (self, name, value):
        self.__dict__[name] = value
        return self

    def has (self, name):
        return name in self.__dict__

    def copy (self):
        new = self.__class__ ()
        new.__dict__ = dict (self.__dict__)
        return new

    def to_dict (self):
        return self.__dict__.copy ()

    def to_pretty (self, format='str'):
        if format == 'str':
            template = '%-*s = %s'
        elif format == 'repr':
            template = '%-*s = %r'
        else:
            raise ValueError ('unrecognied value for "format": %r' % format)

        d = self.__dict__
        maxlen = 0

        for k in six.iterkeys (d):
            maxlen = max (maxlen, len (k))

        return '\n'.join (template % (maxlen, k, d[k])
                          for k in sorted (six.iterkeys (d)))
