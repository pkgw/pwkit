# -*- mode: python; coding: utf-8 -*-
# Copyright 2013-2014 Peter Williams <peter@newton.cx> and collaborators.
# Licensed under the MIT License.

"""pwkit.tinifile - Dealing with typed ini-format files full of measurements.

Functions:

read
  Generate :class:`pwkit.Holder` instances of measurements from an ini-format file.
write
  Write :class:`pwkit.Holder` instances of measurements to an ini-format file.
read_stream
  Lower-level version; only operates on streams, not path names.
write_stream
  Lower-level version; only operates on streams, not path names.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = str ('read_stream read write_stream write').split ()

import six

from . import Holder, inifile, msmt


def _parse_one (old):
    new = {}

    for name, value in six.iteritems (old.__dict__):
        if name == 'section':
            new[name] = value
            continue

        a = name.rsplit (':', 1)
        if len (a) == 1:
            a.append ('s')
        shname, typetag = a
        new[shname] = msmt.parsers[typetag] (value)

    return Holder (**new)


def read_stream (stream, **kwargs):
    for unparsed in inifile.read_stream (stream, **kwargs):
        yield _parse_one (unparsed)


def read (stream_or_path, **kwargs):
    for unparsed in inifile.read (stream_or_path, **kwargs):
        yield _parse_one (unparsed)


def _format_many (holders, defaultsection, extrapos, digest):
    # We need to handle defaultsection here, and not just leave it to inifile,
    # so that we can get consistent digest computation.

    for old in holders:
        s = old.get ('section', defaultsection)
        if s is None:
            raise ValueError ('cannot determine section name for item <%s>' % old)
        new = {'section': s}

        if digest is not None:
            digest.update ('s')
            digest.update (s)

        for name in sorted (x for x in six.iterkeys (old.__dict__) if x != 'section'):
            value = old.get (name)
            if value is None:
                continue

            typetag, ftext, is_imprecise = msmt.fmtinfo (value)
            lname = name
            if len (typetag):
                if is_imprecise and name in extrapos and typetag in ('u', 'f'):
                    typetag = 'P' + typetag
                lname += ':' + typetag
            itext = ' # imprecise' if is_imprecise else ''
            new[lname] = ftext + itext

            if digest is not None:
                digest.update ('k')
                digest.update (name)
                digest.update (typetag)
                digest.update ('v')
                if is_imprecise:
                    digest.update ('<impreciseval>')
                else:
                    digest.update (ftext)

        yield Holder (**new)


def write_stream (stream, holders, defaultsection=None, extrapos=(), sha1sum=False, **kwargs):
    """`extrapos` is basically a hack for multi-step processing. We have some flux
    measurements that are computed from luminosities and distances. The flux
    value is therefore an unwrapped Uval, which doesn't retain memory of any
    positivity constraint it may have had. Therefore, if we write out such a
    value using this routine, we may get something like `fx:u = 1pm1`, and the
    next time it's read in we'll get negative fluxes. Fields listed in
    `extrapos` will have a "P" constraint added if they are imprecise and
    their typetag is just "f" or "u".

    """
    if sha1sum:
        import hashlib
        sha1 = hashlib.sha1 ()
    else:
        sha1 = None

    inifile.write_stream (stream,
                          _format_many (holders, defaultsection, extrapos, sha1),
                          defaultsection=defaultsection,
                          **kwargs)

    if sha1sum:
        return sha1.digest ()


def write (stream_or_path, holders, defaultsection=None, extrapos=(),
           sha1sum=False, **kwargs):
    if sha1sum:
        import hashlib
        sha1 = hashlib.sha1 ()
    else:
        sha1 = None

    inifile.write (stream_or_path,
                   _format_many (holders, defaultsection, extrapos, sha1),
                   defaultsection=defaultsection,
                   **kwargs)

    if sha1sum:
        return sha1.digest ()
