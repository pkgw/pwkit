# -*- mode: python; coding: utf-8 -*-
# Copyright 2014-2015 Peter Williams and collaborators
# Licensed under the MIT License.

"""The :mod:`pwkit.simpleenum` module contains a single decorator function for
creating “enumerations”, by which we mean a group of named, un-modifiable
values. For example::

  from pwkit.simpleenum import enumeration

  @enumeration
  class Constants (object):
    period_days = 2.771
    period_hours = period_days * 24
    n_iters = 300
    # etc

  def myfunction ():
    print ('the period is', Constants.period_hours, 'hours')

The ``class`` declaration syntax is handy here because it lets you define new
values in relation to old values. In the above example, you cannot change any
of the properties of ``Constants`` once it is constructed.

.. important:: If you populate an enumeration with a mutable data type,
   however, we’re unable to prevent you from modifying it. For instance, if
   you do this::

     @enumeration
     class Dangerous (object):
       mutable = [1, 2]
       immutable = (1, 2)

   You can then do something like write ``Dangerous.mutable.append (3)`` and
   modify the value stored in the enumeration. If you’re concerned about this,
   make sure to populate the enumeration with immutable classes such as
   :class:`tuple`, :class:`frozenset`, :class:`int`, and so on.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = str ('enumeration').split ()


def enumeration (cls):
    """A very simple decorator for creating enumerations. Unlike Python 3.4
    enumerations, this just gives a way to use a class declaration to create
    an immutable object containing only the values specified in the class.

    If the attribute ``__pickle_compat__`` is set to True in the decorated
    class, the resulting enumeration value will be callable such that
    ``EnumClass(x) = x``. This is needed to unpickle enumeration values that
    were previously implemented using :class:`enum.Enum`.

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
