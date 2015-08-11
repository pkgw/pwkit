# -*- mode: python; coding: utf-8 -*-
# Copyright 2014-2015 Peter Williams and collaborators
# Licensed under the MIT License.

from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = [b'enumeration']

def enumeration (cls):
    """A very simple decorator for creating enumerations. Unlike Python 3.4
    enumerations, this just gives a way to use a class declaration to create
    an immutable object containing only the values specified in the class.

    If __pickle_compat__ is set to True in the decorated class, the resulting
    enumeration value will be callable such that EnumClass(x) = x. This is
    needed to unpickle enumeration values that were previously implemented
    using `enum.Enum`.

    """
    from pwkit import unicode_to_str
    name = cls.__name__
    pickle_compat = getattr (cls, '__pickle_compat__', False)

    def __unicode__ (self):
        return '<enumeration holder %s>' % name

    def getattr_error (self, attr):
        raise AttributeError ('enumeration %s does not contain attribute %s' % (name, attr))

    def modattr_error (self, *args, **kwargs):
        raise AttributeError ('modification of %s enumeration not allowed' % name)

    clsdict = {
        '__doc__': cls.__doc__,
        '__slots__': (),
        '__unicode__': __unicode__,
        '__str__': unicode_to_str,
        '__repr__': unicode_to_str,
        '__getattr__': getattr_error,
        '__setattr__': modattr_error,
        '__delattr__': modattr_error,
        }

    for key in dir (cls):
        if not key.startswith ('_'):
            clsdict[key] = getattr (cls, key)

    if pickle_compat:
        clsdict['__call__'] = lambda self, x: x

    enumcls = type (name, (object, ), clsdict)
    return enumcls ()
