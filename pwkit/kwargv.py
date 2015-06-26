# -*- mode: python; coding: utf-8 -*-
# Copyright 2012-2014 Peter Williams <peter@newton.cx> and collaborators.
# Licensed under the MIT License.

""" kwargv - keyword-style command-line arguments

Keywords are defined by declaring a subclass of ParseKeywords. Within
that:

- "foo = 1" defines a keyword with a default value, type inferred as
  int. Likewise for str, bool, float.

- "foo = int" defines an int keyword, default value of None. Likewise
  for str, bool, float.

- "foo = [int]" parses as a list of integers of any length, default []
  (I call these "flexible" lists.)

- "foo = [3.0, int]" parses as a 2-element list, default [3.0, None].
  If 1 value is given, the first array item is parsed, and the second
  is left as its default. (I call these "fixed" lists.)

- "foo = Custom(bar, a=b)" parses like "bar" and then customizes
  keyword properties as defined below.

- "@Custom(bar, a=b) \n def foo (value): ..." defines a keyword "foo"
  that parses like "bar", with custom properties as defined below, and
  has its value fixed up by calling the foo() function after the basic
  parsing.  That is, the final value is "foo (intermediate_value)". A
  common pattern is to use a fixup function for a fixed list where the
  first few values are mandatory (see 'minvals' below) but later
  values can be guessed or defaulted.

Instantiating the subclass fills in all defaults, and calling the
"parse()" method parses a list of strings (defaulting to
sys.argv[1:]). See scibin/omegamap for a somewhat complex example.

Properties for keyword customization:

parser (callable): function to parse basic textual value
default (anything): the default value if keyword is unspecified
required (bool, False): whether to raise an error if keyword is not
  seen when parsing
sep (str, ','): separator for parsing the keyword as a list
maxvals (int or None, None): maximum number of values **in flexible
  lists only**
minvals (int, 0): minimum number of values **in fixed lists only**,
  **if the keyword is specified at all**. If you want minvals=1, use
  required=True.
scale (numeric or None, None): multiply the value by this after
  parsing
printexc (bool, False): if there's an exception when parsing the
  keyword value, whether the exception message should be printed.
  (Otherwise, just prints "cannot parse value <val> for keyword <kw>".)
fixupfunc (callable or None, None): after all other parsing/transform
  steps, the final value is the return value of fixupfunc(intermediateval)
uiname (str or None, None): the name of the keyword as presented in the UI.
  I.e., "foo = Custom (0, uiname='bar')" parses keyword "bar=..." but
  sets attribute "foo" in the Python object.
repeatable (bool, False): if true, the keyword value(s) will be contained
  in a list. If the keyword is specified multiple times (ie
  "./program kw=1 kw=2") the list will have multiple items
  ("cfg.kw = [1,2]").


TODO: --help, etc.

TODO: +bool, -bool for ParseKeywords parser (but careful about allowing
--help at least)

TODO: positive, nonzero options for easy bounds-checking of numerics
"""

from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = (b'Custom KwargvError ParseError ParseKeywords basic').split ()

from . import Holder, PKError, text_type


class KwargvError (PKError):
    pass

class ParseError (KwargvError):
    pass


def basic (args=None):
    if args is None:
        import sys
        args = sys.argv[1:]

    parsed = Holder ()

    for arg in args:
        if arg[0] == '+':
            for kw in arg[1:].split (','):
                parsed.set_one (kw, True)
            # avoid analogous -a,b,c syntax because it gets confused with -a --help, etc.
        else:
            t = arg.split ('=', 1)
            if len (t) < 2:
                raise KwargvError ('don\'t know what to do with argument "%s"', arg)
            if not len (t[1]):
                raise KwargvError ('empty value for keyword argument "%s"', t[0])
            parsed.set_one (t[0], t[1])

    return parsed


# The fancy, full-featured system.

class KeywordInfo (object):
    parser = None
    default = None
    required = False
    sep = ','
    maxvals = None
    minvals = 0 # note: maxvals and minvals are used in different ways
    scale = None
    repeatable = False
    printexc = False
    fixupfunc = None
    attrname = None


class KeywordOptions (Holder):
    uiname = None
    subval = None

    def __init__ (self, subval, **kwargs):
        self.set (**kwargs)
        self.subval = subval

    def __call__ (self, fixupfunc):
        # Slightly black magic. Grayish magic. This lets us be used as
        # a decorator on "fixup" functions to modify or range-check
        # the parsed argument value.
        self.fixupfunc = fixupfunc
        return self


Custom = KeywordOptions # sugar for users


def _parse_bool (s):
    s = s.lower ()

    if s in 'y yes t true on 1'.split ():
        return True
    if s in 'n no f false off 0'.split ():
        return False
    raise ParseError ('don\'t know how to interpret "%s" as a boolean' % s)


def _val_to_parser (v):
    if isinstance (v, bool):
        return _parse_bool
    if isinstance (v, (int, float, text_type)):
        return v.__class__
    raise ValueError ('can\'t figure out how to parse %r' % v)


def _val_or_func_to_parser (v):
    if v is bool:
        return _parse_bool
    if callable (v):
        return v
    return _val_to_parser (v)


def _val_or_func_to_default (v):
    if callable (v):
        return None
    if isinstance (v, (int, float, bool, text_type)):
        return v
    raise ValueError ('can\'t figure out a default for %r' % v)


def _handle_flex_list (ki, ks):
    assert len (ks) == 1
    elemparser = ks[0]
    # I don't think 'foo = [0]' will be useful ...
    assert callable (elemparser)

    def flexlistparse (val):
        return [elemparser (i) for i in val.split (ki.sep)]

    return flexlistparse, []


def _handle_fixed_list (ki, ks):
    parsers = [_val_or_func_to_parser (sks) for sks in ks]
    defaults = [_val_or_func_to_default (sks) for sks in ks]
    ntot = len (parsers)

    def fixlistparse (val):
        items = val.split (ki.sep)
        ngot = len (items)

        if ngot < ki.minvals:
            if ki.minvals == ntot:
                raise ParseError ('expected exactly %d values, but only got %d',
                                  ntot, ngot)
            raise ParseError ('expected between %d and %d values, but only got %d',
                              ki.minvals, ntot, ngot)
        if ngot > ntot:
            raise ParseError ('expected between %d and %d values, but got %d',
                              ki.minvals, ntot, ngot)

        result = list (defaults) # make a copy
        for i in xrange (ngot):
            result[i] = parsers[i] (items[i])
        return result

    return fixlistparse, list (defaults) # make a copy


class ParseKeywords (Holder):
    def __init__ (self):
        kwspecs = self.__class__.__dict__
        kwinfos = {}

        # Process our keywords, as specified by the class attributes, into a
        # form more friendly for parsing, and check for things we don't
        # understand. 'kw' is the keyword name exposed to the user; 'attrname'
        # is the name of the attribute to set on the resulting object.

        for kw, ks in kwspecs.iteritems ():
            if kw[0] == '_':
                continue

            ki = KeywordInfo ()
            ko = None
            attrname = kw

            if isinstance (ks, KeywordOptions):
                ko = ks
                ks = ko.subval

                if ko.uiname is not None:
                    kw = ko.uiname

            if callable (ks):
                # expected to be a type (int, float, ...).
                # This branch would get taken for methods, too,
                # which sorta makes sense?
                parser = _val_or_func_to_parser (ks)
                default = _val_or_func_to_default (ks)
            elif isinstance (ks, list) and len (ks) == 1:
                parser, default = _handle_flex_list (ki, ks)
            elif isinstance (ks, list) and len (ks) > 1:
                parser, default = _handle_fixed_list (ki, ks)
            else:
                parser = _val_to_parser (ks)
                default = _val_or_func_to_default (ks)

            ki.attrname = attrname
            ki.parser = parser
            ki.default = default

            if ko is not None: # override with user-specified options
                ki.__dict__.update (ko.__dict__)

            if ki.required:
                # makes sense, and prevents trying to call fixupfunc on
                # weird default values of fixed lists.
                ki.default = None
            elif ki.repeatable:
                ki.default = []
            elif ki.fixupfunc is not None and ki.default is not None:
                # kinda gross structure here, oh well.
                ki.default = ki.fixupfunc (ki.default)

            kwinfos[kw] = ki

        # Apply defaults, save parse info, done

        for kw, ki in kwinfos.iteritems ():
            self.set_one (ki.attrname, ki.default)

        self._kwinfos = kwinfos


    def parse (self, args=None):
        if args is None:
            import sys
            args = sys.argv[1:]

        seen = set ()

        for arg in args:
            t = arg.split ('=', 1)
            if len (t) < 2:
                raise KwargvError ('don\'t know what to do with argument "%s"', arg)

            kw, val = t
            ki = self._kwinfos.get (kw)

            if ki is None:
                raise KwargvError ('unrecognized keyword argument "%s"', kw)

            if not len (val):
                raise KwargvError ('empty value for keyword argument "%s"', kw)

            try:
                pval = ki.parser (val)
            except ParseError as e :
                raise KwargvError ('cannot parse value "%s" for keyword '
                                   'argument "%s": %s', val, kw, e)
            except Exception as e:
                if ki.printexc:
                    raise KwargvError ('cannot parse value "%s" for keyword '
                                       'argument "%s": %s', val, kw, e)
                raise KwargvError ('cannot parse value "%s" for keyword '
                                   'argument "%s"', val, kw)

            if ki.maxvals is not None and len (pval) > ki.maxvals:
                raise KwargvError ('keyword argument "%s" may have at most %d'
                                   ' values, but got %s ("%s")', kw,
                                   ki.maxvals, len (pval), val)

            if ki.scale is not None:
                pval = pval * ki.scale

            if ki.fixupfunc is not None:
                pval = ki.fixupfunc (pval)

            if ki.repeatable:
                # We can't just unilaterally append to the preexisting
                # list, since if we did that starting with the default value
                # we'd mutate the default list.
                cur = self.get (ki.attrname)
                if not len (cur):
                    pval = [pval]
                else:
                    cur.append (pval)
                    pval = cur

            seen.add (kw)
            self.set_one (ki.attrname, pval)

        for kw, ki in self._kwinfos.iteritems ():
            if ki.required and kw not in seen:
                raise KwargvError ('required keyword argument "%s" was not '
                                   'provided', kw)

        return self # convenience


    def parse_or_die (self, args=None):
        from .cli import die

        try:
            return self.parse (args)
        except KwargvError as e:
            die (e)


if __name__ == '__main__':
    print (basic ())
