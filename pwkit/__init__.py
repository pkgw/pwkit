# -*- mode: python; coding: utf-8 -*-
# Copyright 2014-2016 Peter Williams <peter@newton.cx> and collaborators.
# Licensed under the MIT License.

"""A toolkit for science and astronomy in Python.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

# In Python 2, the items in __all__ should be bytes strings. In Python 3, they
# should be Unicode. (http://stackoverflow.com/a/19913680/3760486) If you
# don't use __future__.unicode_literals, you can just write `__all__ =
# ["foo"]` and it's fine, but we do, which causes problems on Py 2.
# Fortunately, to work on both cases we just need to do this:
__all__ = str ('''Holder PKError binary_type reraise_context text_type unicode_to_str''').split ()

__version__ = '0.8.4.99' # also edit ../setup.py, ../docs/source/conf.py!

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
    """A generic base class for exceptions.

    All custom exceptions raised by :mod:`pwkit` modules should be subclasses
    of this class.

    The constructor automatically applies old-fashioned ``printf``-like
    (``%``-based) string formatting if more than one argument is given::

      PKError ('my format string says %r, %d', myobj, 12345)
      # has text content equal to:
      'my format string says %r, %d' % (myobj, 12345)

    If only a single argument is given, the exception text is its
    stringification without applying ``printf``-style formatting.

    """
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

    This function tries to help provide context when a piece of code
    encounters an exception while trying to get something done, and it wishes
    to propagate contextual information farther up the call stack. It only
    makes sense in Python 2, which does not provide Python 3’s `exception
    chaining <https://www.python.org/dev/peps/pep-3134/>`_ functionality.
    Instead of that more sophisticated infrastructure, this function just
    modifies the textual message associated with the exception being raised.

    If only a single argument is supplied, the exception text prepended with
    the stringification of that argument. If multiple arguments are supplied,
    the first argument is treated as an old-fashioned ``printf``-type
    (``%``-based) format string, and the remaining arguments are the formatted
    values.

    Example usage::

      from pwkit import reraise_context
      from pwkit.io import Path

      filename = 'my-filename.txt'

      try:
        f = Path (filename).open ('rt')
        for line in f.readlines ():
          # do stuff ...
      except Exception as e:
        reraise_context ('while reading "%r"', filename)
        # The exception is reraised and so control leaves this function.

    If an exception with text ``"bad value"`` were to be raised inside the
    ``try`` block in the above example, its text would be modified to read
    ``"while reading \"my-filename.txt\": bad value"``.

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
    """Create a new :class:`Holder`. Any keyword arguments will be assigned as
    properties on the object itself, for instance, ``o = Holder (foo=1)``
    yields an object such that ``o.foo`` is 1.

    The *__decorating* keyword is used to implement the :class:`Holder`
    decorator functionality, described below.

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
        """For each keyword argument, sets an attribute on this :class:`Holder` to its
        value.

        Equivalent to::

          for key, value in kwargs.iteritems ():
            setattr (self, key, value)

        Returns *self*.

        """
        self.__dict__.update (kwargs)
        return self

    def get (self, name, defval=None):
        """Get an attribute on this :class:`Holder`.

        Equivalent to ``getattr (self, name, defval)``.

        """
        return self.__dict__.get (name, defval)

    def set_one (self, name, value):
        """Set a single attribute on this object.

        Equivalent to ``setattr (self, name, value)``. Returns *self*.

        """
        self.__dict__[name] = value
        return self

    def has (self, name):
        """Return whether the named attribute has been set on this object.

        This can more naturally be expressed by writing ``name in self``.

        """
        return name in self.__dict__

    def copy (self):
        """Return a shallow copy of this object.

        """
        new = self.__class__ ()
        new.__dict__ = dict (self.__dict__)
        return new

    def to_dict (self):
        """Return a copy of this object converted to a :class:`dict`.

        """
        return self.__dict__.copy ()

    def to_pretty (self, format='str'):
        """Return a string with a prettified version of this object’s contents.

        The format is a multiline string where each line is of the form ``key
        = value``. If the *format* argument is equal to ``"str"``, each
        ``value`` is the stringification of the value; if it is ``"repr"``, it
        is its :func:`repr`.

        Calling :func:`str` on a :class:`Holder` returns a slightly different
        pretty stringification that uses a textual representation similar to a
        Python :class:`dict` literal.

        """
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
