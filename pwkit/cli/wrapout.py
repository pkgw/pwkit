# -*- mode: python; coding: utf-8 -*-
# Copyright 2014-2015 Peter Williams <peter@newton.cx> and collaborators
# Licensed under the MIT License.

"""pwkit.cli.wrapout - the 'wrapout' program."""

from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = str('commandline').split()

import os, signal, sys, time

from . import die, propagate_sigint

usage = """usage: wrapout [-ces] [-a name] <command> [command args...]

Runs *command*, merging its standard output ("stdout") and standard error
("stderr") into a single stream that is printed to this program's stdout.
Output lines are timestamped and prefixed so that errors can be easily
identified.

Options:

-c      -- always colorize output, even if not connected to a TTY
-a name -- use *name* as the argv[0] for *command*
-e      -- echo the subcommand's standard error to our own
           (helpful when logging output to a file).
-s      -- do not propagate signals to or from the child program

Examples:

  $ wrapout echo hello
  $ wrapout false
  $ wrapout complicated_chatty_program foo=bar verbose=1 >log.txt

If our stdout is a TTY, errors are highlighted in red and informational
messages are made bold.

Several messages are printed before and after running the command, including
timestamps, the exit code, and the precise command being run. Each output line
is prefixed with a timestamp (in terms of wall-clock seconds elapsed since the
program was started) and a marker. "II" indicates an informational message,
"--" a line printed to stdout, and "EE" a line printed to stderr.

This program obviously assumes that *command* outputs line-oriented text. It
processes the output line-by-line, so extremely long lines and the like may
cause problems.

If "-s" is not specified, wrapout will attempt to propagate signals to and
from the child program. If this process receives a signal such as SIGTERM, it
will kill the child with the same signal. If the child dies from a signal,
wrapout will kill itself with the same signal.

"""

rfc3339_fmt = '%Y-%m-%dT%H:%M:%S%z'

ansi_red = b'\033[31m'
ansi_cyan = b'\033[36m'
ansi_bold = b'\033[1m'
ansi_reset = b'\033[m'

OUTKIND_STDOUT, OUTKIND_STDERR, OUTKIND_EXTRA = 0, 1, 2


from ..io import get_stdout_bytes, get_stderr_bytes
binary_stdout = get_stdout_bytes()
binary_stderr = get_stderr_bytes()


class Wrapper(object):
    # I like !! for errors and ** for info, but those are nigh-un-grep-able.
    markers = [b' -- ', b' EE ', b' II ']
    use_colors = False
    echo_stderr = False
    propagate_signals = False
    destination = None
    slurp_factory = None

    _red = b''
    _cyan = b''
    _bold = b''
    _reset = b''
    _kind_prefixes = [b'', b'', b'']

    def __init__(self, destination=None):
        if destination is None:
            self.destination = binary_stdout
        else:
            self.destination = destination


    def output(self, kind, line):
        "*line* should be bytes"
        self.destination.write(b''.join([
            self._cyan,
            b't=%07d' % (time.time() - self._t0),
            self._reset,
            self._kind_prefixes[kind],
            self.markers[kind],
            line,
            self._reset,
        ]))
        self.destination.flush()


    def output_stderr(self, text):
        "*text* should be bytes"
        binary_stderr.write(b''.join([
            self._red,
            b't=%07d' % (time.time() - self._t0),
            self._reset,
            b' ',
            text,
        ]))
        binary_stderr.flush()


    def outpar(self, name, value):
        line = ('%s = %s\n' % (name, value)).encode('utf-8')
        self.output(OUTKIND_EXTRA, line)


    def launch(self, cmd, argv, env=None, cwd=None):
        slurp_factory = self.slurp_factory
        if slurp_factory is None:
            from ..slurp import Slurper as slurp_factory

        if self.use_colors:
            self._red = ansi_red
            self._cyan = ansi_cyan
            self._bold = ansi_bold
            self._reset = ansi_reset
            self._kind_prefixes = [b'', self._red, self._bold]

        self._t0 = time.time()
        self.outpar('start_time', time.strftime(rfc3339_fmt))
        self.outpar('exec', cmd)
        self.outpar('argv', ' '.join(repr(s) for s in argv))

        midline_kind = None
        stderr_midline = False

        with slurp_factory(argv=argv, executable=cmd, env=env, cwd=cwd,
                           propagate_signals=self.propagate_signals) as slurp:
            # Here's where we get tricky since we want to output partially
            # complete lines, but we may have to switch between different
            # types of output or informational messages.

            for etype, data in slurp:
                if etype == 'forwarded-signal':
                    if midline_kind is not None:
                        self.destination.write(b'\n')
                        midline_kind = None
                    self.output(OUTKIND_EXTRA, b'forwarded signal %d to child\n' % data)
                elif etype in ('stdout', 'stderr'):
                    if not len(data):
                        continue # EOF, nothing for us to do.

                    kind = OUTKIND_STDERR if etype == 'stderr' else OUTKIND_STDOUT

                    if midline_kind is not None and midline_kind != kind:
                        self.destination.write(b'\n')
                        midline_kind = None

                    lines = data.split(b'\n')

                    if midline_kind is None:
                        self.output(kind, lines[0]) # start a new line
                    else:
                        # If we're here, we must be continuing a line
                        self.destination.write(lines[0])

                    for line in lines[1:-1]:
                        # mid-lines are straightforward
                        self.destination.write(b'\n')
                        self.output(kind, line)

                    if len(lines) == 1:
                        self.destination.flush()
                        midline_kind = kind
                    elif not len(lines[-1]):
                        # We ended right on a newline, which is convenient.
                        self.destination.write(b'\n')
                        self.destination.flush()
                        midline_kind = None
                    else:
                        # We ended with a partial line.
                        self.destination.write(b'\n')
                        self.output(kind, lines[-1])
                        midline_kind = kind

                    if self.echo_stderr and kind == OUTKIND_STDERR:
                        # We use a different format since the intended usage
                        # is that the main output is being logged elsewhere;
                        # this should be terser and distinguishable from the
                        # stdout output.
                        if stderr_midline:
                            binary_stderr.write(lines[0])
                        else:
                            self.output_stderr(lines[0])

                        for line in lines[1:-1]:
                            binary_stderr.write(b'\n')
                            self.output_stderr(line)

                        if len(lines) == 1:
                            binary_stderr.flush()
                            stderr_midline = True
                        elif not len(lines[-1]):
                            binary_stderr.write(b'\n')
                            binary_stderr.flush()
                            stderr_midline = False
                        else:
                            binary_stderr.write(b'\n')
                            self.output_stderr(lines[-1])
                            stderr_midline = True

        self.outpar('finish_time', time.strftime(rfc3339_fmt))
        self.outpar('elapsed_seconds', int(round(time.time() - self._t0)))
        self.outpar('exitcode', slurp.proc.returncode)

        # note: subprocess pre-processes exit codes, so shouldn't use
        # os.WIFSIGNALED, os.WTERMSIG, etc.

        if slurp.proc.returncode < 0:
            signum = -slurp.proc.returncode
            self.output(OUTKIND_STDERR,
                        b'process killed by signal %d\n' % signum)

            if self.propagate_signals:
                signal.signal(signum, signal.SIG_DFL)
                os.kill(os.getpid(), signum) # sayonara
        elif slurp.proc.returncode != 0:
            self.output(OUTKIND_STDERR, b'process exited with error code\n')

        return slurp.proc.returncode


def commandline(argv=None):
    if argv is None:
        argv = sys.argv

    # NOTE: we do NOT initialize stdout and stderr to be Unicode streams
    # since we're actually intentionally writing raw bytes to them.
    propagate_sigint()

    args = list(argv[1:])
    use_colors = None
    echo_stderr = False
    propagate_signals = True
    argv0 = None

    while len(args):
        if args[0] == '-c':
            use_colors = True
            args = args[1:]
        elif args[0] == '-e':
            echo_stderr = True
            args = args[1:]
        elif args[0] == '-s':
            propagate_signals = False
            args = args[1:]
        elif args[0] == '-a':
            if len(args) < 2:
                die('another argument must come after the "-a" option')
            argv0 = args[1]
            args = args[2:]
        elif args[0] == '--':
            args = args[1:]
            break
        elif args[0][0] == '-':
            die('unrecognized option "%s"', args[0])
        else:
            # End of option arguments.
            break

    if len(args) < 1:
        print(usage.strip(), file=sys.stderr)
        sys.exit(0)

    subcommand = args[0]
    subargv = args
    if argv0 is not None:
        subargv[0] = argv0

    if use_colors is None:
        use_colors = binary_stdout.isatty()

    wrapper = Wrapper()
    wrapper.use_colors = use_colors
    wrapper.echo_stderr = echo_stderr
    wrapper.propagate_signals = propagate_signals
    sys.exit(wrapper.launch(subcommand, subargv))


if __name__ == '__main__':
    # Note that the standard wrapper created by setup.py does not actually
    # follow this code path! It invokes commandline() directly.
    commandline()
