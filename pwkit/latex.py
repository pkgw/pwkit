# -*- mode: python; coding: utf-8 -*-
# Copyright 2013-2014 Peter Williams <peter@newton.cx> and collaborators.
# Licensed under the MIT License.

"""pwkit.latex - various helpers for the LaTeX typesetting system.

Classes
-------

Referencer
  Accumulate a numbered list of bibtex references, then output them.
TableBuilder
  Create awesome deluxetables programmatically.

Functions
---------

latexify_l3col
  Format value in LaTeX, suitable for tables of limit values.
latexify_n2col
  Format a number in LaTeX in 2-column decimal-aligned formed.
latexify_u3col
  Format value in LaTeX, suitable for tables of uncertain values.
latexify
  Format a value in LaTeX appropriately.

Helpers for TableBuilder
------------------------

AlignedNumberFormatter
  Format numbers, aligning them at the decimal point.
BasicFormatter
  Base class for formatters.
BoolFormatter
  Format a boolean; default is True -> bullet, False -> nothing.
LimitFormatter
  Format measurements for a table of limits.
MaybeNumberFormatter
  Format numbers with a fixed number of decimal places, or
  objects with __pk_latex__().
UncertFormatter
  Format measurements for a table of detailed uncertainties.
WideHeader
  Helper for multi-column headers.


XXX: Barely tested!

"""
from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = str ('''AlignedNumberFormatter BasicFormatter BoolFormatter LimitFormatter
                  MaybeNumberFormatter Referencer TableBuilder UncertFormatter
                  WideHeader latexify_l3col latexify_n2col latexify_u3col
                  latexify''').split ()

import six
from six.moves import range
from . import Holder, PKError, binary_type, msmt, reraise_context, text_type


def _reftext (key):
    if key[0] == '*':
        return key[1:].encode ('ascii')
    return b'\\citet{%s}' % key.encode ('ascii')


class Referencer (object):
    """Accumulate a numbered list of bibtex references. Methods:

    refkey (bibkey)
      Return a string that should be used to give
      a numbered reference to the given bibtex
      key. "thiswork" is handled specially.
    dump ()
      Return a string with \citet{} commands identifing
      all of the numbered references.

    Attributes:

    thisworktext
      text referring to "this work"; defaults to that.
    thisworkmarker
      special symbol used to denote "this work"; defaults to star.

    Bibtex keys beginning with asterisks have the rest of their value used for
    the citation text, rather than "\citet{<key>}".

    """
    thisworktext = b'this work'
    thisworkmarker = b'$\\star$'

    def __init__ (self):
        self.bibkeys = []
        self.seenthiswork = False


    def refkey (self, bibkey):
        if bibkey is None:
            return ''

        if bibkey == 'thiswork':
            self.seenthiswork = True
            return self.thisworkmarker

        try:
            idx = self.bibkeys.index (bibkey)
        except ValueError:
            idx = len (self.bibkeys)
            self.bibkeys.append (bibkey)

        return text_type (idx + 1).encode ('ascii')


    def dump (self):
        s = b', '.join (b'[%d] %s' % (i + 1, _reftext (self.bibkeys[i]))
                       for i in range (len (self.bibkeys)))

        if self.seenthiswork:
            s = (b'[%s]: %s, ' % (self.thisworkmarker, self.thisworktext)) + s

        return s


# Generic infrastructure for converting Python objects to LaTeX.
#
# Note that it's important that these functions all accept miscellaneous
# kwargs arguments, so that TableBuilder invocations can pass along special
# control keywords that are only specific to certain cells, without causing
# crashes elsewhere.

def latexify (obj, **kwargs):
    """Render an object in LaTeX appropriately.

    """
    if hasattr (obj, '__pk_latex__'):
        return obj.__pk_latex__ (**kwargs)

    if isinstance (obj, text_type):
        from unicode_to_latex import unicode_to_latex
        return unicode_to_latex (obj)

    if isinstance (obj, bool):
        # isinstance (True, int) = True, so gotta handle this first.
        raise ValueError ('no well-defined LaTeXification of bool %r' % obj)

    if isinstance (obj, float):
        nplaces = kwargs.get ('nplaces')
        if nplaces is None:
            return b'$%f$' % obj
        return b'$%.*f$' % (nplaces, obj)

    if isinstance (obj, int):
        return b'$%d$' % obj

    if isinstance (obj, binary_type):
        raise ValueError ('no safe LaTeXification of binary string %r' % obj)

    raise ValueError ('can\'t LaTeXify %r' % obj)


def latexify_n2col (x, nplaces=None, **kwargs):
    """Render a number into LaTeX in a 2-column format, where the columns split
    immediately to the left of the decimal point. This gives nice alignment of
    numbers in a table.

    """
    if nplaces is not None:
        t = b'%.*f' % (nplaces, x)
    else:
        t = b'%f' % x

    if b'.' not in t:
        return b'$%s$ &' % t

    left, right = t.split (b'.')
    return b'$%s$ & $.%s$' % (left, right)


def latexify_u3col (obj, **kwargs):
    """Convert an object to special LaTeX for uncertainty tables.

    This conversion is meant for uncertain values in a table. The return value
    should span three columns. The first column ends just before the decimal
    point in the main number value, if it has one. It has no separation from
    the second column. The second column goes from the decimal point until
    just before the "plus-or-minus" indicator. The third column goes from the
    "plus-or-minus" until the end. If the item being formatted does not fit this
    schema, it can be wrapped in something like '\multicolumn{3}{c}{...}'.

    """
    if hasattr (obj, '__pk_latex_u3col__'):
        return obj.__pk_latex_u3col__ (**kwargs)

    # TODO: there are reasonable ways to format many basic types, but I'm not
    # going to implement them until I need to.

    raise ValueError ('can\'t LaTeXify %r in 3-column uncertain format' % obj)


def latexify_l3col (obj, **kwargs):
    """Convert an object to special LaTeX for limit tables.

    This conversion is meant for limit values in a table. The return value
    should span three columns. The first column is the limit indicator: <, >,
    ~, etc. The second column is the whole part of the value, up until just
    before the decimal point. The third column is the decimal point and the
    fractional part of the value, if present. If the item being formatted does
    not fit this schema, it can be wrapped in something like
    '\multicolumn{3}{c}{...}'.

    """
    if hasattr (obj, '__pk_latex_l3col__'):
        return obj.__pk_latex_l3col__ (**kwargs)

    if isinstance (obj, bool):
        # isinstance (True, int) = True, so gotta handle this first.
        raise ValueError ('no well-defined l3col LaTeXification of bool %r' % obj)

    if isinstance (obj, float):
        return b'&' + latexify_n2col (obj, **kwargs)

    if isinstance (obj, int):
        return b'& $%d$ &' % obj

    raise ValueError ('can\'t LaTeXify %r in 3-column limit format' % obj)



# Building nice deluxetables.

class WideHeader (object):
    """Information needed for constructing wide table headers.

    nlogcols - Number of logical columns consumed by this header.
    content  - The LaTeX to insert for this header's content.
    align    - The alignment of this header; default 'c'.

    Rendered as \multicolumn{nlatex}{align}{content}, where `nlatex` is the
    number of LaTeX columns spanned by this header -- which may be larger than
    `nlogcols` if certain logical columns span multiple LaTeX columns.

    """
    def __init__ (self, nlogcols, content, align=b'c'):
        self.nlogcols = nlogcols
        self.align = align
        self.content = content


class TableBuilder (object):
    """Build and then emit a nice deluxetable.

    Methods:

    addcol (headings, datafunc, formatter=None, colspec=None, numbering='(%d)')
       Define a logical column.
    addnote (key, text)
       Define a table note that can appear in cells.
    addhcline (headerrowix, logcolidx, latexdeltastart, latexdeltaend)
       Add a horizontal line between columns.
    notemark (key)
       Return a \\tablenotemark{} command for the specified note key.
    emit (stream, items)
       Write the table, with one row for each thing in `items`, to the stream.

    If an item has an attribute `tb_row_preamble`, that text is written verbatim
    before that corresponding row is output.

    Attributes:

    environment
      The name of the latex environment to use, default "deluxetable".
      You may want to specify "deluxetable*", or "mydeluxetable" if
      using a hacked package.
    label
      The latex reference label of the table. Mandatory.
    note
      A note at the table footer ("\\tablecomments{}" in LaTeX).
    preamble
      Commands for table preamble. See below.
    refs
      Contents of the table References section.
    title
      Table title. Default "Untitled table".
    widthspec
      Passed to \\tablewidth{}; default "0em" = auto-widen.
    numbercols
      If True, number each column. This can be disabled on a
      col-by-col basis by calling `addcol` with `numbering` set to
      False.
    final_double_backslash
      If True, end the final table row with a ''\\''. AAStex6 requires this,
      giving an error about a misplaced '\omit' if you don't provide one.
      On the other hand, classic TeX tables look worse if you do provide this.

    Legal preamble commands are::

        \\rotate
        \\tablenum{<manual table identifier>}
        \\tabletypesize{<font size command>}

    The commands \\tablecaption, \\tablecolumns, \\tablehead, and \\tablewidth
    are handled specially.

    If \\tablewidth{} is not provided, the table is set at full width, not its
    natural width, which is a lame default. The default `widthspec` lets us
    auto-widen while providing a clear avenue to customizing the width.

    """
    environment = b'deluxetable'
    label = None
    note = b''
    preamble = b''
    refs = b''
    title = b'Untitled table'
    widthspec = b'0em'
    numbercols = True
    final_double_backslash = False

    def __init__ (self, label):
        self._colinfo = []
        self._hclines = []
        self._notes = {}
        self._notecounter = 0
        self.label = label


    def addcol (self, headings, datafunc, formatter=None, colspec=None, numbering='(%d)'):
        """Define a logical column. Arguments:

        headings
          A string, or list of strings and WideHeaders. The headings are stacked
          vertically in the table header section.
        datafunc
          Return LaTeX for this cell. Call spec should be
          (item, [formatter, [tablebuilder]]).
        formatter
          The formatter to use; defaults to a new BasicFormatter.
        colspec
          The LaTeX column specification letters to use; defaults to 'c's.
        numbering
          If non-False, a format for writing this column's number; if False,
          no number is written.

        """
        if formatter is None:
            formatter = BasicFormatter ()

        if isinstance (headings, six.string_types):
            headings = (headings, )

        if hasattr (datafunc, 'func_code'):
            nargs = datafunc.func_code.co_argcount
        elif hasattr (datafunc, '__call__'):
            # This is pretty hacky ...
            nargs = datafunc.__call__.func_code.co_argcount - 1
        else:
            raise ValueError ('datafunc must have a "func_code" field')

        if nargs == 3:
            wrapped = datafunc # (item, formatter, builder)
        elif nargs == 2:
            wrapped = lambda i, f, b: datafunc (i, f)
        elif nargs == 1:
            wrapped = lambda i, f, b: datafunc (i)
        elif nargs == 0: # why not
            wrapped = lambda i, f, b: datafunc ()
        else:
            raise ValueError ('datafunc must accept between 0 and 3 args; it takes %d' % nargs)

        ci = Holder (headings=headings, formatter=formatter,
                     wdatafunc=wrapped, colspec=colspec, numbering=numbering)
        self._colinfo.append (ci)
        return self


    def addnote (self, key, text):
        self._notes[key] = [None, text]
        return self


    def addhcline (self, headerrowidx, logcolidx, latexdeltastart, latexdeltaend):
        """Adds a horizontal line below a limited range of columns in the header section.
        Arguments:

        headerrowidx    - The 0-based row number *below* which the line will be
                          drawn; i.e. 0 means that the line will be drawn below
                          the first row of header cells.
        logcolidx       - The 0-based 'logical' column number relative to which
                          the line will be placed; i.e. 1 means that the line
                          placement will be relative to the second column
                          defined in an addcol() call.
        latexdeltastart - The relative position at which to start drawing the
                          line relative to that logical column, in LaTeX
                          columns; typically going to be zero.
        latexdeltaend   - The relative position at which to finish drawing the
                          line, in the standard Python noninclusive sense. I.e.,
                          if you want to underline two LaTeX columns,
                          latexdeltaend = latexdeltastart + 2.

        """
        self._hclines.append ((headerrowidx, logcolidx, latexdeltastart, latexdeltaend))
        return self


    def notemark (self, key):
        noteinfo = self._notes.get (key)
        if noteinfo is None:
            raise ValueError ('unrecognized note key "%s"' % key)

        if noteinfo[0] is None:
            if self._notecounter > 25:
                raise PKError ('maximum number of table notes exceeded')

            noteinfo[0] = self._notecounter
            self._notecounter += 1

        return b'\\tablenotemark{%c}' % chr (ord (b'a') + noteinfo[0]).encode ('ascii')


    def emit (self, stream, items):
        from six import itervalues
        write = stream.write
        colinfo = self._colinfo

        colspec = b''
        ncols = 0
        nheadrows = 0
        curlatexcol = 1

        for ci in colinfo:
            ci.nlcol, colspecpart, ci.headprefix = ci.formatter.colinfo (self)
            ci.latexcol = curlatexcol

            if ci.colspec is not None:
                # This is more about convenience for columns that don't have
                # fancy alignment requirements, rather than about allowing
                # overriding.
                colspecpart = ci.colspec

            if colspecpart is None:
                colspecpart = b'c' * ci.nlcol

            ncols += ci.nlcol
            colspec += colspecpart
            nheadrows = max (nheadrows, len (ci.headings))
            curlatexcol += ci.nlcol

        write (b'% TableBuilder table\n')
        write (br'\begin{')
        write (self.environment)
        write (b'}{')
        write (colspec)
        write (b'}\n%custom preamble\n')
        write (self.preamble)
        write (b'\n%hardcoded preamble\n\\tablecolumns{')
        write (text_type (ncols).encode ('ascii'))
        write (b'}\n\\tablewidth{')
        write (self.widthspec)
        write (b'}\n\\tablecaption{')
        write (self.title)
        write (b'\\label{')
        write (self.label)
        write (b'}}\n\\tablehead{\n')

        cr = b''

        for i in range (nheadrows):
            write (cr)

            for hidx, cidx, lds, lde in self._hclines:
                # Note super inefficiency. Who cares?
                if hidx == i - 1:
                    latexcolbase = colinfo[cidx].latexcol
                    write (b' \\cline{')
                    write (text_type (latexcolbase + lds).encode ('ascii'))
                    write (b'-')
                    write (text_type (latexcolbase + lde - 1).encode ('ascii'))
                    write (b'} ')

            sep = b''
            nlefttoskip = 0

            for cidx, ci in enumerate (colinfo):
                write (sep)

                if nlefttoskip < 1:
                    if len (ci.headings) <= i:
                        write (b' & ' * (ci.nlcol - 1))
                    else:
                        h = ci.headings[i]

                        if isinstance (h, WideHeader):
                            nlefttoskip = h.nlogcols

                            nlatex = 0
                            for j in range (h.nlogcols):
                                nlatex += colinfo[cidx + j].nlcol

                            write (b'\\multicolumn{')
                            write (text_type (nlatex).encode ('ascii'))
                            write (b'}{')
                            write (h.align)
                            write (b'}{')
                            write (h.content)
                            write (b'}')
                        else:
                            write (ci.headprefix)
                            write (b'{')
                            write (h)
                            write (b'}')

                nlefttoskip -= 1

                if nlefttoskip > 0:
                    sep = b' '
                else:
                    sep = b' & '

            cr = b' \\\\\n'

        if self.numbercols:
            colnum = 1
            sep = b''
            write (b' \\\\ \\\\\n')

            for ci in colinfo:
                write (sep)
                write (b'\\multicolumn{')
                write (text_type (ci.nlcol).encode ('ascii'))
                write (b'}{c}{')
                if ci.numbering is False:
                    pass
                elif b'%d' in ci.numbering:
                    write (ci.numbering % colnum)
                    colnum += 1
                else:
                    write (ci.numbering)
                write (b'}')
                sep = b' & '

        write (b'\n}\n\\startdata\n')

        cr = b''

        for item in items:
            write (cr)
            sep = b''

            rp = getattr (item, 'tb_row_preamble', None)
            if rp is not None:
                write (rp)

            for ci in colinfo:
                write (sep)
                formatted = ci.wdatafunc (item, ci.formatter, self)
                try:
                    write (formatted)
                except Exception:
                    reraise_context ('while writing %r (from %r with %r)',
                                     formatted, item, ci.formatter)
                sep = b' & '

            cr = b' \\\\\n'

        if self.final_double_backslash:
            write (b' \\\\')
        write (b'\n\\enddata\n')

        if self.note is not None and len (self.note):
            write (b'\\tablecomments{')
            write (self.note)
            write (b'}\n')

        if self.refs is not None and len (self.refs):
            write (b'\\tablerefs{')
            write (self.refs)
            write (b'}\n')

        for noteinfo in sorted ((ni for ni in itervalues (self._notes)
                                 if ni[0] is not None), key=lambda ni: ni[0]):
            write (b'\\tablenotetext{')
            write (chr (ord ('a') + noteinfo[0]).encode ('ascii'))
            write (b'}{')
            write (noteinfo[1])
            write (b'}\n')

        write (br'\end{')
        write (self.environment)
        write (b'}\n% end TableBuilder table\n')


class BasicFormatter (object):
    """Base class for formatting table cells in a TableBuilder.

    Generally a formatter will also provide methods for turning input data
    into fancified LaTeX output that can be used by the column's "data
    function".

    """
    def colinfo (self, builder):
        """Return (nlcol, colspec, headprefix), where:

        nlcol      - The number of LaTeX columns encompassed by this logical
                     column.
        colspec    - Its LaTeX column specification (None to force user to
                     specify).
        headprefix - Prefix applied before heading items in {} (e.g.,
                     "\\colhead").

        """
        return 1, None, b'\\colhead'



class BoolFormatter (BasicFormatter):
    """Format booleans. Attributes `truetext` and `falsetext` set what shows up
    for true and false values, respectively.

    """
    truetext = b'$\\bullet$'
    falsetext = b''

    def colinfo (self, builder):
        return 1, b'c', b'\\colhead'

    def format (self, value):
        if value:
            return self.truetext
        return self.falsetext


class MaybeNumberFormatter (BasicFormatter):
    """Format Python objects. If it's a number, format it as such, without any
    fancy column alignment, but with a specifiable number of decimal places.
    Otherwise, call latexify() on it.

    """
    def __init__ (self, nplaces=1, align=b'c'):
        self.nplaces = nplaces
        self.align = align

    def colinfo (self, builder):
        return 1, self.align, b'\\colhead'

    def format (self, datum, nplaces=None):
        if datum is None:
            return b''

        try:
            v = float (datum)
        except TypeError:
            return latexify (datum)
        else:
            if nplaces is None:
                nplaces = self.nplaces
            return b'$%.*f$' % (nplaces, v)


class AlignedNumberFormatter (BasicFormatter):
    """Format numbers. Allows the number of decimal places to be specified, and
    aligns the numbers at the decimal point.

    """
    def __init__ (self, nplaces=1):
        self.nplaces = nplaces

    def colinfo (self, builder):
        return 2, b'r@{}l', b'\\multicolumn{2}{c}'

    def format (self, datum, nplaces=None):
        if datum is None:
            return b' & '
        if nplaces is None:
            nplaces = self.nplaces

        return latexify_n2col (float (datum), nplaces=nplaces)


class UncertFormatter (BasicFormatter):
    """Format measurements (cf. pwkit.msmt) with detailed uncertainty information,
    possibly including asymmetric uncertainties. Because of the latter
    possibility, table rows have to be made extra-high to maintain evenness.

    """
    strut = br'\rule{0pt}{3ex}'

    def colinfo (self, builder):
        return 3, b'r@{}l@{\,}l', b'\\multicolumn{3}{c}'

    def format (self, datum, **kwargs):
        if datum is None:
            return b' & & ' + self.strut
        return latexify_u3col (datum, **kwargs) + self.strut


class LimitFormatter (BasicFormatter):
    """Format measurements (cf pwkit.msmt) with nice-looking limit information.
    Specific uncertainty information is discarded. The default formats do not
    involve fancy subscripts or superscripts, so row struts are not needed ...
    by default.

    """
    strut = br''

    def colinfo (self, builder):
        return 3, br'r@{\,}r@{}l', br'\multicolumn{3}{c}'

    def format (self, datum, **kwargs):
        if datum is None:
            return b' & & ' + self.strut
        return latexify_l3col (datum, **kwargs) + self.strut
