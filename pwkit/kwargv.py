# -*- mode: python; coding: utf-8 -*-
# Copyright 2012-2015, 2018 Peter Williams <peter@newton.cx> and collaborators.
# Licensed under the MIT License.

"""The :mod:`pwkit.kwargv` module provides a framework for parsing
keyword-style arguments to command-line programs. It’s designed so that you
can easily make a routine with complex, structured configuration parameters
that can also be driven from the command line.

Keywords are defined by declaring a subclass of the
:class:`ParseKeywords` class with fields corresponding to the
support keywords::

  from pwkit.kwargv import ParseKeywords, Custom

  class MyConfig(ParseKeywords):
      foo = 1
      bar = str
      multi = [int]
      extra = Custom(float, required=True)

      @Custom(str)
      def declination(value):
          from pwkit.astutil import parsedeglat
          return parsedeglat(value)

Instantiating the subclass fills in all defaults. Calling the
:meth:`ParseKeywords.parse` method parses a list of strings (defaulting to
``sys.argv[1:]``) and updates the instance’s properties. This framework is
designed so that you can provide complex configuration to an algorithm either
programmatically, or on the command line. A typical use would be::

  from pwkit.kwargv import ParseKeywords, Custom

  class MyConfig(ParseKeywords):
      niter = 1
      input = str
      scales = [int]
      # ...

  def my_complex_algorithm(cfg):
     from pwkit.io import Path
     data = Path(cfg.input).read_fits()

     for i in range(cfg.niter):
         # ....

  def call_algorithm_in_code():
      cfg = MyConfig()
      cfg.input = 'testfile.fits'
      # ...
      my_complex_algorithm(cfg)

  if __name__ == '__main__':
      cfg = MyConfig().parse()
      my_complex_algorithm(cfg)

You could then execute the module as a program and specify arguments in the
form ``./program niter=5 input=otherfile.fits``.


Keyword Specification Format
----------------------------

Arguments are specified in the following ways:

- ``foo = 1`` defines a keyword with a default value, type inferred as
  ``int``. Likewise for ``str``, ``bool``, ``float``.

- ``bar = str`` defines an string keyword with default value of None.
  Likewise for ``int``, ``bool``, ``float``.

- ``multi = [int]`` parses as a list of integers of any length, defaulting to
  the empty list ``[]`` (I call these "flexible" lists.). List items are
  separated by commas on the command line.

- ``other = [3.0, int]`` parses as a 2-element list, defaulting to ``[3.0,
  None]``. If one value is given, the first array item is parsed, and the
  second is left as its default. (I call these "fixed" lists.)

- ``extra = Custom(float, required=True)`` parses like ``float`` and then
  customizes keyword properties. Supported properties are the attributes of
  the :class:`KeywordInfo` class.

- Use :class:`Custom` as a decorator (``@Custom``) on a function ``foo``
  defines a keyword ``foo`` that’s parsed according to the :class:`Custom`
  specification, then has its value fixed up by calling the ``foo()`` function
  after the basic parsing. That is, the final value is ``foo
  (intermediate_value)``. A common pattern is to use a fixup function for a
  fixed list where the first few values are mandatory (see
  :attr:`KeywordInfo.minvals` below) but later values can be guessed or
  defaulted.

See the :class:`KeywordInfo` documentation for specification of additional
keyword properties that may be specified. The ``Custom`` name is simply an
alias for :class:`KeywordInfo`.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = str("Custom KwargvError ParseError KeywordInfo ParseKeywords basic").split()

from . import Holder, PKError


class KwargvError(PKError):
    """Raised when invalid arguments have been provided."""


class ParseError(KwargvError):
    """Raised when the structure of the arguments appears legitimate, but a
    particular value cannot be parsed into its expected type.

    """


def basic(args=None):
    """Parse the string list *args* as a set of keyword arguments in a very
    simple-minded way, splitting on equals signs. Returns a
    :class:`pwkit.Holder` instance with attributes set to strings. The form
    ``+foo`` is mapped to setting ``foo = True`` on the :class:`pwkit.Holder`
    instance. If *args* is ``None``, ``sys.argv[1:]`` is used. Raises
    :exc:`KwargvError` on invalid arguments (i.e., ones without an equals sign
    or a leading plus sign).

    """
    if args is None:
        import sys

        args = sys.argv[1:]

    parsed = Holder()

    for arg in args:
        if arg[0] == "+":
            for kw in arg[1:].split(","):
                parsed.set_one(kw, True)
            # avoid analogous -a,b,c syntax because it gets confused with -a --help, etc.
        else:
            t = arg.split("=", 1)
            if len(t) < 2:
                raise KwargvError('don\'t know what to do with argument "%s"', arg)
            if not len(t[1]):
                raise KwargvError('empty value for keyword argument "%s"', t[0])
            parsed.set_one(t[0], t[1])

    return parsed


# The fancy, full-featured system.


class KeywordInfo(object):
    """Properties that a keyword argument may have."""

    parser = None
    """A callable used to convert the argument text to a Python value.
    This attribute is assigned automatically upon setup."""

    default = None
    """The default value for the keyword if it’s left unspecified."""

    required = False
    """Whether an error should be raised if the keyword is not seen while
    parsing."""

    sep = ","
    """The textual separator between items for list-valued keywords."""

    maxvals = None
    """The maximum number of values allowed. This only applies for flexible lists;
    fixed lists have predetermined sizes.

    """
    minvals = 0  # note: maxvals and minvals are used in different ways
    """The minimum number of values allowed in a flexible list, *if the keyword is
    specified at all*. If you want ``minvals = 1``, use ``required = True``.

    """
    scale = None
    """If not ``None``, multiply numeric values by this number after parsing."""

    repeatable = False
    """If true, the keyword value(s) will always be contained in a list. If they
    keyword is specified multiple times (i.e. ``./program kw=1 kw=2``), the
    list will have multiple items (``cfg.kw = [1, 2]``). If the keyword is
    list-valued, using this will result in a list of lists.

    """
    printexc = False
    """Print the exception as normal if there’s an exception when parsing the
    keyword value. Otherwise there’s just a message along the lines of “cannot
    parse value <val> for keyword <kw>”.

    """
    fixupfunc = None
    """If not ``None``, the final value of the keyword is set to the return value
    of ``fixupfunc(intermediate_value)``.

    """
    _attrname = None

    # This isn't used on Keyword*Info* instances, but adding a dummy here makes
    # the docs much saner:
    uiname = None
    """The name of the keyword as parsed from the command-line. For instance,
    ``some_value = Custom(int, uiname="some-value")`` will result in a
    keyword that the user sets by calling ``./program some-value=3``. This
    provides a mechanism to support keyword names that are not legal Python
    identifiers.

    """


class KeywordOptions(Holder):
    uiname = None
    subval = None

    def __init__(self, subval, **kwargs):
        self.set(**kwargs)
        self.subval = subval

    def __call__(self, fixupfunc):
        # Slightly black magic. Grayish magic. This lets us be used as
        # a decorator on "fixup" functions to modify or range-check
        # the parsed argument value.
        self.fixupfunc = fixupfunc
        return self


Custom = KeywordOptions  # sugar for users


def _parse_bool(s):
    s = s.lower()

    if s in "y yes t true on 1".split():
        return True
    if s in "n no f false off 0".split():
        return False
    raise ParseError('don\'t know how to interpret "%s" as a boolean' % s)


def _val_to_parser(v):
    if isinstance(v, bool):
        return _parse_bool
    if isinstance(v, (int, float, str)):
        return v.__class__
    raise ValueError("can't figure out how to parse %r" % v)


def _val_or_func_to_parser(v):
    if v is bool:
        return _parse_bool
    if callable(v):
        return v
    return _val_to_parser(v)


def _val_or_func_to_default(v):
    if callable(v):
        return None
    if isinstance(v, (int, float, bool, str)):
        return v
    raise ValueError

    ("can't figure out a default for %r" % v)


def _handle_flex_list(ki, ks):
    assert len(ks) == 1
    elemparser = ks[0]
    # I don't think 'foo = [0]' will be useful ...
    assert callable(elemparser)

    def flexlistparse(val):
        return [elemparser(i) for i in val.split(ki.sep)]

    return flexlistparse, []


def _handle_fixed_list(ki, ks):
    parsers = [_val_or_func_to_parser(sks) for sks in ks]
    defaults = [_val_or_func_to_default(sks) for sks in ks]
    ntot = len(parsers)

    def fixlistparse(val):
        items = val.split(ki.sep)
        ngot = len(items)

        if ngot < ki.minvals:
            if ki.minvals == ntot:
                raise ParseError(
                    "expected exactly %d values, but only got %d", ntot, ngot
                )
            raise ParseError(
                "expected between %d and %d values, but only got %d",
                ki.minvals,
                ntot,
                ngot,
            )
        if ngot > ntot:
            raise ParseError(
                "expected between %d and %d values, but got %d", ki.minvals, ntot, ngot
            )

        result = list(defaults)  # make a copy
        for i in range(ngot):
            result[i] = parsers[i](items[i])
        return result

    return fixlistparse, list(defaults)  # make a copy


class ParseKeywords(Holder):
    """The template class for defining your keyword arguments. A subclass of
    :class:`pwkit.Holder`. Declare attributes in a subclass following the
    scheme described above, then call the :meth:`ParseKeywords.parse` method.

    """

    def __init__(self):
        kwspecs = self.__class__.__dict__
        kwinfos = {}

        # Process our keywords, as specified by the class attributes, into a
        # form more friendly for parsing, and check for things we don't
        # understand. 'kw' is the keyword name exposed to the user; 'attrname'
        # is the name of the attribute to set on the resulting object.

        for kw, ks in kwspecs.items():
            if kw[0] == "_":
                continue

            ki = KeywordInfo()
            ko = None
            attrname = kw

            if isinstance(ks, KeywordOptions):
                ko = ks
                ks = ko.subval

                if ko.uiname is not None:
                    kw = ko.uiname

            if callable(ks):
                # expected to be a type (int, float, ...).
                # This branch would get taken for methods, too,
                # which sorta makes sense?
                parser = _val_or_func_to_parser(ks)
                default = _val_or_func_to_default(ks)
            elif isinstance(ks, list) and len(ks) == 1:
                parser, default = _handle_flex_list(ki, ks)
            elif isinstance(ks, list) and len(ks) > 1:
                parser, default = _handle_fixed_list(ki, ks)
            else:
                parser = _val_to_parser(ks)
                default = _val_or_func_to_default(ks)

            ki._attrname = attrname
            ki.parser = parser
            ki.default = default

            if ko is not None:  # override with user-specified options
                ki.__dict__.update(ko.__dict__)

            if ki.required:
                # makes sense, and prevents trying to call fixupfunc on
                # weird default values of fixed lists.
                ki.default = None
            elif ki.repeatable:
                ki.default = []
            elif ki.fixupfunc is not None:
                # Make sure to process the default through the fixup, if it
                # exists. This helps code use "interesting" defaults with types
                # that you might prefer to use when launching a task
                # programmatically; e.g. a default output stream that is
                # `sys.stdout`, not "-". Note, however, that the fixup will
                # always get called for the default value, so it shouldn't do
                # anything too expensive.
                ki.default = ki.fixupfunc(ki.default)

            kwinfos[kw] = ki

        # Apply defaults, save parse info, done

        for kw, ki in kwinfos.items():
            self.set_one(ki._attrname, ki.default)

        self._kwinfos = kwinfos

    def parse(self, args=None):
        """Parse textual keywords as described by this class’s attributes, and update
        this instance’s attributes with the parsed values. *args* is a list of
        strings; if ``None``, it defaults to ``sys.argv[1:]``. Returns *self*
        for convenience. Raises :exc:`KwargvError` if invalid keywords are
        encountered.

        See also :meth:`ParseKeywords.parse_or_die`.

        """
        if args is None:
            import sys

            args = sys.argv[1:]

        seen = set()

        for arg in args:
            t = arg.split("=", 1)
            if len(t) < 2:
                raise KwargvError('don\'t know what to do with argument "%s"', arg)

            kw, val = t
            ki = self._kwinfos.get(kw)

            if ki is None:
                raise KwargvError('unrecognized keyword argument "%s"', kw)

            if not len(val):
                raise KwargvError('empty value for keyword argument "%s"', kw)

            try:
                pval = ki.parser(val)
            except ParseError as e:
                raise KwargvError(
                    'cannot parse value "%s" for keyword ' 'argument "%s": %s',
                    val,
                    kw,
                    e,
                )
            except Exception as e:
                if ki.printexc:
                    raise KwargvError(
                        'cannot parse value "%s" for keyword ' 'argument "%s": %s',
                        val,
                        kw,
                        e,
                    )
                raise KwargvError(
                    'cannot parse value "%s" for keyword ' 'argument "%s"', val, kw
                )

            if ki.maxvals is not None and len(pval) > ki.maxvals:
                raise KwargvError(
                    'keyword argument "%s" may have at most %d'
                    ' values, but got %s ("%s")',
                    kw,
                    ki.maxvals,
                    len(pval),
                    val,
                )

            if ki.scale is not None:
                pval = pval * ki.scale

            if ki.fixupfunc is not None:
                pval = ki.fixupfunc(pval)

            if ki.repeatable:
                # We can't just unilaterally append to the preexisting
                # list, since if we did that starting with the default value
                # we'd mutate the default list.
                cur = self.get(ki._attrname)
                if not len(cur):
                    pval = [pval]
                else:
                    cur.append(pval)
                    pval = cur

            seen.add(kw)
            self.set_one(ki._attrname, pval)

        for kw, ki in self._kwinfos.items():
            if kw not in seen:
                if ki.required:
                    raise KwargvError(
                        'required keyword argument "%s" was not provided', kw
                    )

        return self  # convenience

    def parse_or_die(self, args=None):
        """Like :meth:`ParseKeywords.parse`, but calls :func:`pkwit.cli.die` if a
        :exc:`KwargvError` is raised, printing the exception text. Returns
        *self* for convenience.

        """
        from .cli import die

        try:
            return self.parse(args)
        except KwargvError as e:
            die(e)


if __name__ == "__main__":
    print(basic())
