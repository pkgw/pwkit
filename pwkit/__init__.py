# -*- mode: python; coding: utf-8 -*-
# Copyright 2014 Peter Williams <peter@newton.cx> and collaborators.
# Licensed under the MIT License.

"""
Peter Williams' toolkit for science and astronomy.

>>> import pwkit as pk

"""

from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = ('PKError binary_type text_type unicode_to_str').split ()


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
