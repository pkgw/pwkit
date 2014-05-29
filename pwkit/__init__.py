# -*- mode: python; coding: utf-8 -*-
# Copyright 2014 Peter Williams <peter@newton.cx> and collaborators.
# Licensed under the MIT License.

"""
Peter Williams' toolkit for science and astronomy.

>>> import pwkit as pk

Modules:

  (this one)       - Exceptions, Holder class, Python 3 compat help.
  cli              - Utilities for command-line programs.
  inifile          - Simple ini-format file parser.
  io               - Utilities for input and output.
  kwargv           - Keyword-style argument parsing.
  unicode_to_latex - Rendering Unicode to LaTeX.

"""

from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = ('Holder PKError binary_type reraise_context text_type '
           'unicode_to_str').split ()


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
    def __init__ (self, **kwargs):
        self.set (**kwargs)

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
