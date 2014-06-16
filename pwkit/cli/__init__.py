# -*- mode: python; coding: utf-8 -*-
# Copyright 2012-2014 Peter Williams <peter@newton.cx> and collaborators.
# Licensed under the MIT License.

"""pwkit.cli - miscellaneous utilities for command-line programs."""

from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = (b'check_usage die pop_option show_usage unicode_stdio warn '
           b'wrong_usage').split ()

import codecs, sys
from .. import text_type


def unicode_stdio ():
    """Make sure that the standard I/O streams accept Unicode.

    The standard I/O streams accept bytes, not Unicode characters. This means
    that in principle every Unicode string that we want to output should be
    encoded to utf-8 before print()ing. But Python 2.X has a hack where, if
    the output is a terminal, it will automatically encode your strings, using
    UTF-8 in most cases.

    BUT this hack doesn't kick in if you pipe your program's output to another
    program. So it's easy to write a tool that works fine in most cases but then
    blows up when you log its output to a file.

    The proper solution is just to do the encoding right. This function sets
    things up to do this in the most sensible way I can devise. This approach
    sets up compatibility with Python 3, which has the stdio streams be in
    text mode rather than bytes mode to begin with.

    Basically, every command-line Python program should call this right at
    startup. I'm tempted to just invoke this code whenever this module is
    imported since I foresee many accidentally omissions of the call.

    """
    enc = sys.stdin.encoding or 'utf-8'
    sys.stdin = codecs.getreader (enc) (sys.stdin)
    enc = sys.stdout.encoding or enc
    sys.stdout = codecs.getwriter (enc) (sys.stdout)
    enc = sys.stderr.encoding or enc
    sys.stderr = codecs.getwriter (enc) (sys.stderr)


def die (fmt, *args):
    """Raise a :exc:`SystemExit` exception with a formatted error message.

    :arg str fmt: a format string
    :arg args: arguments to the format string

    If *args* is empty, a :exc:`SystemExit` exception is raised with the
    argument ``'error: ' + str (fmt)``. Otherwise, the string component is
    ``fmt % args``. If uncaught, the interpreter exits with an error code and
    prints the exception argument.

    Example::

       if ndim != 3:
          die ('require exactly 3 dimensions, not %d', ndim)

    """
    if not len (args):
        raise SystemExit ('error: ' + text_type (fmt))
    raise SystemExit ('error: ' + (fmt % args))


def warn (fmt, *args):
    if not len (args):
        s = text_type (fmt)
    else:
        s = fmt % args

    print ('warning:', s, file=sys.stderr)


# Simple-minded argument handling -- see also kwargv.

def pop_option (ident, argv=None):
    """A lame routine for grabbing command-line arguments. Returns a boolean
    indicating whether the option was present. If it was, it's removed from
    the argument string. Because of the lame behavior, options can't be
    combined, and non-boolean options aren't supported. Operates on sys.argv
    by default.

    Note that this will proceed merrily if argv[0] matches your option.

    """
    if argv is None:
        from sys import argv

    if len (ident) == 1:
        ident = '-' + ident
    else:
        ident = '--' + ident

    found = ident in argv
    if found:
        argv.remove (ident)

    return found


def show_usage (docstring, short, stream, exitcode):
    """Print program usage information and exit.

    :arg str docstring: the program help text

    This function just prints *docstring* and exits. In most cases, the
    function :func:`check_usage` should be used: it automatically checks
    :data:`sys.argv` for a sole "-h" or "--help" argument and invokes this
    function.

    This function is provided in case there are instances where the user
    should get a friendly usage message that :func:`check_usage` doesn't catch.
    It can be contrasted with :func:`wrong_usage`, which prints a terser usage
    message and exits with an error code.

    """
    if stream is None:
        from sys import stdout as stream

    if not short:
        print ('Usage:', docstring.strip (), file=stream)
    else:
        intext = False
        for l in docstring.splitlines ():
            if intext:
                if not len (l):
                    break
                print (l, file=stream)
            elif len (l):
                intext = True
                print ('Usage:', l, file=stream)

        print ('\nRun with a sole argument --help for more detailed '
               'usage information.', file=stream)

    raise SystemExit (exitcode)


def check_usage (docstring, argv=None, usageifnoargs=False):
    """Check if the program has been run with a --help argument; if so,
    print usage information and exit.

    :arg str docstring: the program help text
    :arg argv: the program arguments; taken as :data:`sys.argv` if
        given as :const:`None` (the default). (Note that this implies
        ``argv[0]`` should be the program name and not the first option.)
    :arg bool usageifnoargs: if :const:`True`, usage information will be
        printed and the program will exit if no command-line arguments are
        passed. If "long", print long usasge. Default is :const:`False`.

    This function is intended for small programs launched from the command
    line. The intention is for the program help information to be written in
    its docstring, and then for the preamble to contain something like::

        \"\"\"myprogram - this is all the usage help you get\"\"\"
        import sys
        ... # other setup
        check_usage (__doc__)
        ... # go on with business

    If it is determined that usage information should be shown,
    :func:`show_usage` is called and the program exits.

    See also :func:`wrong_usage`.

    """
    if argv is None:
        from sys import argv

    if len (argv) == 1 and usageifnoargs:
        show_usage (docstring, (usageifnoargs != 'long'), None, 0)
    if len (argv) == 2 and argv[1] in ('-h', '--help'):
        show_usage (docstring, False, None, 0)


def wrong_usage (docstring, *rest):
    """Print a message indicating invalid command-line arguments and exit with an
    error code.

    :arg str docstring: the program help text
    :arg rest: an optional specific error message

    This function is intended for small programs launched from the command
    line. The intention is for the program help information to be written in
    its docstring, and then for argument checking to look something like
    this::

        \"\"\"mytask <input> <output>

        Do something to the input to create the output.
        \"\"\"
        ...
        import sys
        ... # other setup
        check_usage (__doc__)
        ... # more setup
        if len (sys.argv) != 3:
           wrong_usage (__doc__, "expect exactly 2 arguments, not %d",
                        len (sys.argv))

    When called, an error message is printed along with the *first stanza* of
    *docstring*. The program then exits with an error code and a suggestion to
    run the program with a --help argument to see more detailed usage
    information. The "first stanza" of *docstring* is defined as everything up
    until the first blank line, ignoring any leading blank lines.

    The optional message in *rest* is treated as follows. If *rest* is empty,
    the error message "invalid command-line arguments" is printed. If it is a
    single item, the stringification of that item is printed. If it is more
    than one item, the first item is treated as a format string, and it is
    percent-formatted with the remaining values. See the above example.

    See also :func:`check_usage` and :func:`show_usage`.

    """
    intext = False

    if len (rest) == 0:
        detail = 'invalid command-line arguments'
    elif len (rest) == 1:
        detail = rest[0]
    else:
        detail = rest[0] % tuple (rest[1:])

    print ('error:', detail, '\n', file=sys.stderr) # extra NL
    show_usage (docstring, True, sys.stderr, 1)
