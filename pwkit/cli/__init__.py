# -*- mode: python; coding: utf-8 -*-
# Copyright 2012-2015 Peter Williams <peter@newton.cx> and collaborators.
# Licensed under the MIT License.

"""pwkit.cli - miscellaneous utilities for command-line programs.

Functions:

backtrace_on_usr1 - Make it so that a Python backtrace is printed on SIGUSR1.
check_usage       - Print usage and exit if --help is in argv.
die               - Print an error and exit.
fork_detached_process - Fork a detached process.
pop_option        - Check for a single command-line option.
propagate_sigint  - Ensure that calling shells know when we die from SIGINT.
show_usage        - Print a usage message.
unicode_stdio     - Ensure that sys.std{in,out,err} accept unicode strings.
warn              - Print a warning.
wrong_usage       - Print an error about wrong usage and the usage help.

Context managers:

print_tracebacks  - Catch exceptions and print tracebacks without reraising them.

Submodules:

multitool - Framework for command-line programs with sub-commands.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = str(
    """check_usage die fork_detached_process pop_option print_tracebacks
                  propagate_sigint show_usage unicode_stdio warn
                  wrong_usage"""
).split()

import os, signal, sys


def unicode_stdio():
    """Make sure that the standard I/O streams accept Unicode.

    This function does nothing, because we have dropped support for Python 2. In
    Python 3 it is not necessary.

    In Python 2, the standard I/O streams accept bytes, not Unicode characters.
    This means that in principle every Unicode string that we want to output
    should be encoded to utf-8 before print()ing. But Python 2.X has a hack
    where, if the output is a terminal, it will automatically encode your
    strings, using UTF-8 in most cases.

    BUT this hack doesn't kick in if you pipe your program's output to another
    program. So it's easy to write a tool that works fine in most cases but then
    blows up when you log its output to a file.

    The proper solution is just to do the encoding right. This function sets
    things up to do this in the most sensible way I can devise, if we're running
    on Python 2. This approach sets up compatibility with Python 3, which has
    the stdio streams be in text mode rather than bytes mode to begin with.

    Basically, every command-line Python program should call this right at
    startup. I'm tempted to just invoke this code whenever this module is
    imported since I foresee many accidentally omissions of the call.

    """
    return


class _InterruptSignalPropagator(object):
    """Ensure that calling shells know when we die from SIGINT.

    Imagine that a shell script is running a long-running subprogram and the
    user hits control-C to interrupt the program. What happens is that both
    the shell and the subprogram are sent SIGINT, which usually causes the
    subprogram to die immediately. However, the shell's behavior is more
    complicated. Certain subprograms might handle the SIGINT and *not* die
    immediately, and the shell needs to be prepared to handle that situation.
    Therefore the shell notes the SIGINT and sees what happens next. If the
    subprogram dies from the SIGINT, then the shell dies too. If not, the
    shell continues. The shell can determine this by using the POSIX-defined C
    macros WIFSIGNALED() and WTERMSIG() to see how the subprogram exited.

    A problem comes in to view. Python programs trap SIGINT and turn it into
    a KeyboardInterrupt. Uncaught KeyboardInterrupts cause the program to exit,
    but *not* through the death-by-signal route. Therefore, interrupting
    Python programs will not cause parent shells to exit as desired. This can
    be seen by control-C'ing the following shell script:

       for x in 1 2 3 4 5 ; do
         echo $x
         python -c "import time; time.sleep (5)"
       done

    This function fixes this behavior by causing uncaught KeyboardInterrupts
    to trigger death-by-SIGINT. Importantly, you can't fool the shell by
    exiting with the right code; you have to kill yourself with an
    honest-to-God uncaught SIGINT.

    This is all accomplished by placing in a shim sys.excepthook() handler for
    KeyboardInterrupt exceptions. The previous excepthook can be accessed as
    `pwkit.cli.propagate_sigint.inner_excepthook`.

    """

    inner_excepthook = None

    def __call__(self):
        """Set sys.excepthook to our special version.

        It chains to the previous excepthook, of course. This value can be
        accessed as `pwkit.cli.propagate_sigint.inner_excepthook`.

        """
        if self.inner_excepthook is None:
            self.inner_excepthook = sys.excepthook
            sys.excepthook = self.excepthook

    def excepthook(self, etype, evalue, etb):
        """Handle an uncaught exception. We always forward the exception on to
        whatever `sys.excepthook` was present upon setup. However, if the
        exception is a KeyboardInterrupt, we additionally kill ourselves with
        an uncaught SIGINT, so that invoking programs know what happened.

        """
        self.inner_excepthook(etype, evalue, etb)

        if issubclass(etype, KeyboardInterrupt):
            # Don't try this at home, kids. On some systems os.kill (0, ...)
            # signals our entire progress group, which is not what we want,
            # so we use os.getpid ().
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            os.kill(os.getpid(), signal.SIGINT)


propagate_sigint = _InterruptSignalPropagator()


def _print_backtrace_signal_handler(signum, frame):
    try:
        import traceback

        print(
            "*** Printing traceback due to receipt of signal #%d" % signum,
            file=sys.stderr,
        )
        for fn, line, func, text in traceback.extract_stack(frame):
            print(
                "***   %s (%s:%d): %s" % (fn, func, line, text or "??"), file=sys.stderr
            )
        print("*** End of traceback (innermost call is last)", file=sys.stderr)
        assert False
    except Exception as e:
        print(
            "*** Failed to print traceback on receipt of signal #%d: %s (%s)"
            % (signum, e, e.__class__.__name__),
            file=sys.stderr,
        )


def backtrace_on_usr1():
    """Install a signal handler such that this program prints a Python traceback
    upon receipt of SIGUSR1. This could be useful for checking that
    long-running programs are behaving properly, or for discovering where an
    infinite loop is occurring.

    Note, however, that the Python interpreter does not invoke Python signal
    handlers exactly when the process is signaled. For instance, a signal
    delivered in the midst of a time.sleep() call will only be seen by Python
    code after that call completes. This means that this feature may not be as
    helpful as one might like for debugging certain kinds of problems.

    """
    import signal

    try:
        signal.signal(signal.SIGUSR1, _print_backtrace_signal_handler)
    except Exception as e:
        warn("failed to set up Python backtraces on SIGUSR1: %s", e)


def die(fmt, *args):
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
    if not len(args):
        raise SystemExit("error: " + str(fmt))
    raise SystemExit("error: " + (fmt % args))


def warn(fmt, *args):
    if not len(args):
        s = str(fmt)
    else:
        s = fmt % args

    print("warning:", s, file=sys.stderr)


class print_tracebacks(object):
    """Context manager that catches exceptions and prints their tracebacks without
    reraising them. Intended for robust programs that want to continue
    execution even if something bad happens; this provides the infrastructure
    to swallow exceptions while still preserving exception information for
    later debugging.

    You can specify which exception classes to catch with the `types` keyword
    argument to the constructor. The `header` keyword will be printed if
    specified; this could be used to add contextual information. The `file`
    keyword specifies the destination for the printed output; default is
    sys.stderr.

    Instances preserve the exception information in the fields 'etype',
    'evalue', and 'etb' if your program in fact wants to do something with the
    information. One basic use would be checking whether an exception did, in
    fact, occur.

    """

    header = "Swallowed exception:"

    def __init__(self, types=(Exception,), header=None, file=None):
        self.types = types
        self.file = file

        if header is not None:
            self.header = header

    def __enter__(self):
        self.etype = self.evalue = self.etb = None
        return self

    def __exit__(self, etype, evalue, etb):
        if etype is None:
            return False  # all good, woohoo

        if not isinstance(evalue, self.types):
            # Exception happened but not something of the kind we expect. Reraise.
            return False

        # Exception happened and we should do our thing.
        self.etype = etype
        self.evalue = evalue
        self.etb = etb

        if self.header is not None:
            print(self.header, file=self.file or sys.stderr)

        from traceback import print_exception

        print_exception(etype, evalue, etb, file=self.file)
        return True  # swallow this exception


def fork_detached_process():
    """Fork this process, creating a subprocess detached from the current context.

    Returns a :class:`pwkit.Holder` instance with information about what
    happened. Its fields are:

    whoami
      A string, either "original" or "forked" depending on which process we are.
    pipe
      An open binary file descriptor. It is readable by the original process
      and writable by the forked one. This can be used to pass information
      from the forked process to the one that launched it.
    forkedpid
      The PID of the forked process. Note that this process is *not* a child of
      the original one, so waitpid() and friends may not be used on it.

    Example::

      from pwkit import cli

      info = cli.fork_detached_process ()
      if info.whoami == 'original':
          message = info.pipe.readline ().decode ('utf-8')
          if not len (message):
              cli.die ('forked process (PID %d) appears to have died', info.forkedpid)
          info.pipe.close ()
          print ('forked process said:', message)
      else:
          info.pipe.write ('hello world'.encode ('utf-8'))
          info.pipe.close ()

    As always, the *vital* thing to understand is that immediately after a
    call to this function, you have **two** nearly-identical but **entirely
    independent** programs that are now both running simultaneously. Until you
    execute some kind of ``if`` statement, the only difference between the two
    processes is the value of the ``info.whoami`` field and whether
    ``info.pipe`` is readable or writeable.

    This function uses :func:`os.fork` twice and also calls :func:`os.setsid`
    in between the two invocations, which creates new session and process
    groups for the forked subprocess. It does *not* perform other operations
    that you might want, such as changing the current directory, dropping
    privileges, closing file descriptors, and so on. For more discussion of
    best practices when it comes to “daemonizing” processes, see (stalled)
    `PEP 3143`_.

    .. _PEP 3143: https://www.python.org/dev/peps/pep-3143/

    """
    import os, struct
    from .. import Holder

    payload = struct.Struct("L")

    info = Holder()
    readfd, writefd = os.pipe()

    pid1 = os.fork()
    if pid1 > 0:
        info.whoami = "original"
        info.pipe = os.fdopen(readfd, "rb")
        os.close(writefd)

        retcode = os.waitpid(pid1, 0)[1]
        if retcode:
            raise Exception("child process exited with error code %d" % retcode)

        (info.forkedpid,) = payload.unpack(info.pipe.read(payload.size))
    else:
        # We're the intermediate child process. Start new session and process
        # groups, detaching us from TTY signals and whatnot.
        os.setsid()

        pid2 = os.fork()
        if pid2 > 0:
            # We're the intermediate process; we're all done
            os._exit(0)

        # If we get here, we're the detached child process.
        info.whoami = "forked"
        info.pipe = os.fdopen(writefd, "wb")
        os.close(readfd)
        info.forkedpid = os.getpid()
        info.pipe.write(payload.pack(info.forkedpid))

    return info


# Simple-minded argument handling -- see also kwargv.


def pop_option(ident, argv=None):
    """A lame routine for grabbing command-line arguments. Returns a boolean
    indicating whether the option was present. If it was, it's removed from
    the argument string. Because of the lame behavior, options can't be
    combined, and non-boolean options aren't supported. Operates on sys.argv
    by default.

    Note that this will proceed merrily if argv[0] matches your option.

    """
    if argv is None:
        from sys import argv

    if len(ident) == 1:
        ident = "-" + ident
    else:
        ident = "--" + ident

    found = ident in argv
    if found:
        argv.remove(ident)

    return found


def show_usage(docstring, short, stream, exitcode):
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
        print("Usage:", docstring.strip(), file=stream)
    else:
        intext = False
        for l in docstring.splitlines():
            if intext:
                if not len(l):
                    break
                print(l, file=stream)
            elif len(l):
                intext = True
                print("Usage:", l, file=stream)

        print(
            "\nRun with a sole argument --help for more detailed " "usage information.",
            file=stream,
        )

    raise SystemExit(exitcode)


def check_usage(docstring, argv=None, usageifnoargs=False):
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

    if len(argv) == 1 and usageifnoargs:
        show_usage(docstring, (usageifnoargs != "long"), None, 0)
    if len(argv) == 2 and argv[1] in ("-h", "--help"):
        show_usage(docstring, False, None, 0)


def wrong_usage(docstring, *rest):
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

    if len(rest) == 0:
        detail = "invalid command-line arguments"
    elif len(rest) == 1:
        detail = rest[0]
    else:
        detail = rest[0] % tuple(rest[1:])

    print("error:", detail, "\n", file=sys.stderr)  # extra NL
    show_usage(docstring, True, sys.stderr, 1)
