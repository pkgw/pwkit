# -*- mode: python; coding: utf-8 -*-
# Copyright 2013-2014 Peter Williams <peter@newton.cx> and collaborators.
# Licensed under the MIT License.

"""pwkit.tabfile - I/O with typed tables of uncertain measurements.

Functions:

read    - Read a typed table file.
vizread - Read a headerless table file, with columns specified separately
write   - Write a typed table file.

The table format is line-oriented text. Hashes denote comments. Initial lines
of the form "colname = value" set a column name that gets the same value for
every item in the table. The header line is prefixed with an @ sign.
Subsequent lines are data rows.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = str ('read vizread write').split ()

import six
from . import Holder, PKError, io, msmt, reraise_context


def _getparser (lname):
    a = lname.rsplit (':', 1)
    if len (a) == 1:
        a.append ('s')
    return a[0], msmt.parsers[a[1]]


def _trimmedlines (path, **kwargs):
    for line in io.pathlines (path, **kwargs):
        line = line[:-1] # trailing newline
        line = line.split ('#', 1)[0]
        if not len (line):
            continue
        if line.isspace ():
            continue
        yield line


def read (path, tabwidth=8, **kwargs):
    """Read a typed tabular text file into a stream of Holders.

    Arguments:

    path
      The path of the file to read.
    tabwidth=8
      The tab width to assume. Please don't monkey with it.
    mode='rt'
      The file open mode (passed to io.open()).
    noexistok=False
      If True and the file is missing, treat it as empty.
    ``**kwargs``
      Passed to io.open ().

    Returns a generator for a stream of `pwkit.Holder`s, each of which will
    contain ints, strings, or some kind of measurement (cf `pwkit.msmt`).

    """
    datamode = False
    fixedcols = {}

    for text in _trimmedlines (path, **kwargs):
        text = text.expandtabs (tabwidth)

        if datamode:
            # table row
            h = Holder ()
            h.set (**fixedcols)
            for name, cslice, parser in info:
                try:
                    v = parser (text[cslice].strip ())
                except:
                    reraise_context ('while parsing "%s"', text[cslice].strip ())
                h.set_one (name, v)
            yield h
        elif text[0] != '@':
            # fixed column
            padnamekind, padval = text.split ('=', 1)
            name, parser = _getparser (padnamekind.strip ())
            fixedcols[name] = parser (padval.strip ())
        else:
            # column specification
            n = len (text)
            assert n > 1
            start = 0
            info = []

            while start < n:
                end = start + 1
                while end < n and (not text[end].isspace ()):
                    end += 1

                if start == 0:
                    namekind = text[start+1:end] # eat leading @
                else:
                    namekind = text[start:end]

                while end < n and text[end].isspace ():
                    end += 1

                name, parser = _getparser (namekind)
                if parser is None: # allow columns to be ignored
                    skippedlast = True
                else:
                    skippedlast = False
                    info.append ((name, slice (start, end), parser))
                start = end

            datamode = True

            if not skippedlast:
                # make our last column go as long as the line goes
                # (e.g. for "comments" columns)
                # but if the real last column is ":x"-type, then info[-1]
                # doesn't run up to the end of the line, so do nothing in that case.
                lname, lslice, lparser = info[-1]
                info[-1] = lname, slice (lslice.start, None), lparser


def _tabpad (text, width, tabwidth=8):
    # note: assumes we're starting tab-aligned
    l = len (text)
    assert l <= width

    if l == width:
        return text

    n = width - l
    ntab = n // tabwidth
    nsp = n - ntab * tabwidth
    return ''.join ((text, ' ' * nsp, '\t' * ntab))


def write (stream, items, fieldnames, tabwidth=8):
    """Write a typed tabular text file to the specified stream.

    Arguments:

    stream
      The destination stream.
    items
      An iterable of items to write. Two passes have to
      be made over the items (to discover the needed column widths),
      so this will be saved into a list.
    fieldnames
      Either a list of field name strings, or a single string.
      If the latter, it will be split into a list with .split().
    tabwidth=8
      The tab width to use. Please don't monkey with it.

    Returns nothing.

    """
    if isinstance (fieldnames, six.string_types):
        fieldnames = fieldnames.split ()

    maxlens = [0] * len (fieldnames)

    # We have to make two passes, so listify:
    items = list (items)

    # pass 1: get types and maximum lengths for each record. Pad by 1 to
    # ensure there's at least one space between all columns.

    coltypes = [None] * len (fieldnames)

    for i in items:
        for idx, fn in enumerate (fieldnames):
            val = i.get (fn)
            if val is None:
                continue

            typetag, text, inexact = msmt.fmtinfo (val)
            maxlens[idx] = max (maxlens[idx], len (text) + 1)

            if coltypes[idx] is None:
                coltypes[idx] = typetag
                continue

            if coltypes[idx] == typetag:
                continue

            if coltypes[idx][-1] == 'f' and typetag[-1] == 'u':
                # Can upcast floats to uvals
                if coltypes[idx][:-1] == typetag[:-1]:
                    coltypes[idx] = coltypes[idx][:-1] + 'u'
                    continue

            if coltypes[idx][-1] == 'u' and typetag[-1] == 'f':
                if coltypes[idx][:-1] == typetag[:-1]:
                    continue

            raise PKError ('irreconcilable column types: %s and %s', coltypes[idx], typetag)

    # Compute column headers and their widths

    headers = list (fieldnames)
    headers[0] = '@' + headers[0]

    for idx, fn in enumerate (fieldnames):
        if coltypes[idx] != '':
            headers[idx] += ':' + coltypes[idx]

        maxlens[idx] = max (maxlens[idx], len (headers[idx]))

    widths = [tabwidth * ((k + tabwidth - 1) // tabwidth) for k in maxlens]

    # pass 2: write out

    print (''.join (_tabpad (h, widths[idx], tabwidth)
                    for (idx, h) in enumerate (headers)), file=stream)

    def ustr (i, f):
        v = i.get (f)
        if v is None:
            return ''
        return msmt.fmtinfo (v)[1]

    for i in items:
        print (''.join (_tabpad (ustr (i, fn), widths[idx], tabwidth)
                        for (idx, fn) in enumerate (fieldnames)), file=stream)


def vizread (descpath, descsection, tabpath, tabwidth=8, **kwargs):
    """Read a headerless tabular text file into a stream of Holders.

    Arguments:

    descpath
      The path of the table description ini file.
    descsection
      The section in the description file to use.
    tabpath
      The path to the actual table data.
    tabwidth=8
      The tab width to assume. Please don't monkey with it.
    mode='rt'
      The table file open mode (passed to io.open()).
    noexistok=False
      If True and the file is missing, treat it as empty.
    ``**kwargs``
      Passed to io.open ().

    Returns a generator of a stream of `pwkit.Holder`s, each of which will
    contain ints, strings, or some kind of measurement (cf `pwkit.msmt`). In
    this version, the table file does not contain a header, as seen in Vizier
    data files. The corresponding section in the description ini file has keys
    of the form "colname = <start> <end> [type]", where <start> and <end> are
    the **1-based** character numbers defining the column, and [type] is an
    optional specified of the measurement type of the column (one of the usual
    b, i, f, u, Lu, Pu).

    """
    from .inifile import read as iniread

    cols = []

    for i in iniread (descpath):
        if i.section != descsection:
            continue

        for field, desc in six.iteritems (i.__dict__):
            if field == 'section':
                continue

            a = desc.split ()
            idx0 = int (a[0]) - 1

            if len (a) == 1:
                cols.append ((field, slice (idx0, idx0 + 1), msmt.parsers['s']))
                continue

            if len (a) == 2:
                parser = msmt.parsers['s']
            else:
                parser = msmt.parsers[a[2]]

            cols.append ((field, slice (idx0, int (a[1])), parser))

    for text in _trimmedlines (tabpath, **kwargs):
        text = text.expandtabs (tabwidth)

        h = Holder ()
        for name, cslice, parser in cols:
            try:
                v = parser (text[cslice].strip ())
            except:
                reraise_context ('while parsing "%s"', text[cslice].strip ())
            h.set_one (name, v)

        yield h
