# -*- mode: python; coding: utf-8 -*-
# Copyright 2014 Peter Williams <peter@newton.cx> and collaborators
# Licensed under the MIT License.

"""pwkit.cli.wrapout - the 'wrapout' program."""

from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = [b'commandline']

import os, signal, subprocess, sys, threading, time, Queue

from . import die, propagate_sigint


usage = """usage: wrapout [-c] [-e] [-a name] <command> [command args...]

Runs *command*, merging its standard output ("stdout") and standard error
("stderr") into a single stream that is printed to this program's stdout.
Output lines are timestamped and prefixed so that errors can be easily
identified.

Options:

-c      -- always colorize output, even if not connected to a TTY
-a name -- use *name* as the argv[0] for *command*
-e      -- echo the subcommand's standard error to our own
           (helpful when logging output to a file).

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

"""

rfc3339_fmt = '%Y-%m-%dT%H:%M:%S%z'

ansi_red = '\033[31m'
ansi_cyan = '\033[36m'
ansi_bold = '\033[1m'
ansi_reset = '\033[m'

OUTKIND_STDOUT, OUTKIND_STDERR, OUTKIND_EXTRA = 0, 1, 2


class Wrapper (object):
    # I like !! for errors and ** for info, but those are nigh-un-grep-able.
    markers = [' -- ', ' EE ', ' II ']
    use_colors = False
    echo_stderr = False
    poll_timeout = 0.2
    destination = None

    _red = ''
    _cyan = ''
    _bold = ''
    _reset = ''
    _kind_prefixes = ['', '', '']

    def __init__ (self, destination=None):
        # Python print output isn't threadsafe (!) so we have to communicate
        # lines from the readers back to the main thread for things to come
        # out correctly.
        self._lines = Queue.Queue ()

        if destination is None:
            self.destination = sys.stdout
        else:
            self.destination = destination


    def monitor (self, fd, outkind):
        while True:
            # NOTE: 'for line in fd' queues up a lot of lines before yielding anything.
            line = fd.readline ()
            if not len (line):
                break
            self._lines.put ((outkind, line))


    def output (self, kind, line):
        print (self._cyan,
               't=%07d' % (time.time () - self._t0),
               self._reset,
               self._kind_prefixes[kind],
               self.markers[kind],
               line,
               self._reset,
               sep='', end='', file=self.destination)
        self.destination.flush ()


    def outpar (self, name, value):
        line = ('%s = %s\n' % (name, value)).encode ('utf-8')
        self.output (OUTKIND_EXTRA, line)


    def launch (self, cmd, argv, env=None, cwd=None):
        if self.use_colors:
            self._red = ansi_red
            self._cyan = ansi_cyan
            self._bold = ansi_bold
            self._reset = ansi_reset
            self._kind_prefixes = ['', self._red, self._bold]

        self._t0 = time.time ()
        self.outpar ('start_time', time.strftime (rfc3339_fmt))
        self.outpar ('exec', cmd)
        self.outpar ('argv', ' '.join (repr (s) for s in argv))

        proc = subprocess.Popen (argv,
                                 executable=cmd,
                                 env=env,
                                 cwd=cwd,
                                 stdin=open (os.devnull, 'r'),
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE,
                                 shell=False)

        tout = threading.Thread (target=self.monitor,
                                 name='stdout-monitor',
                                 args=(proc.stdout, OUTKIND_STDOUT))
        tout.daemon = True
        tout.start ()

        terr = threading.Thread (target=self.monitor,
                                 name='stderr-monitor',
                                 args=(proc.stderr, OUTKIND_STDERR))
        terr.daemon = True
        terr.start ()

        while True:
            keepgoing = proc.poll () is None
            keepgoing = keepgoing or tout.is_alive ()
            keepgoing = keepgoing or terr.is_alive ()
            keepgoing = keepgoing or not self._lines.empty ()
            if not keepgoing:
                break

            try:
                kind, line = self._lines.get (timeout=self.poll_timeout)
            except Queue.Empty:
                continue
            except KeyboardInterrupt:
                self.output (OUTKIND_STDERR, 'interrupted\n')
                proc.send_signal (2) # SIGINT
                continue

            self.output (kind, line)

            if self.echo_stderr and kind == OUTKIND_STDERR:
                # We use a different format since the intended usage is that
                # the main output is being logged elsewhere; this should be
                # terser and distinguishable from the stdout output.
                print (self._red,
                       't=%07d' % (time.time () - self._t0),
                       self._reset,
                       ' ',
                       line,
                       sep='', end='', file=sys.stderr)
                sys.stderr.flush ()

        self.outpar ('finish_time', time.strftime (rfc3339_fmt))
        self.outpar ('elapsed_seconds', int (round (time.time () - self._t0)))
        self.outpar ('exitcode', proc.returncode)

        # note: subprocess pre-processes exit codes, so shouldn't use
        # os.WIFSIGNALED, os.WTERMSIG, etc.

        if proc.returncode < 0:
            self.output (OUTKIND_STDERR,
                         'process killed by signal %d\n' % -proc.returncode)
            if proc.returncode == -signal.SIGINT:
                raise KeyboardInterrupt () # make sure to propagate death-by-SIGINT
        elif proc.returncode != 0:
            self.output (OUTKIND_STDERR, 'process exited with error code\n')

        return proc.returncode


def commandline (argv=None):
    if argv is None:
        argv = sys.argv

    # NOTE: we do NOT initialize stdout and stderr to be Unicode streams
    # since we're actually intentionally writing raw bytes to them.
    propagate_sigint ()

    args = list (argv[1:])
    use_colors = None
    echo_stderr = False
    argv0 = None

    while len (args):
        if args[0] == '-c':
            use_colors = True
            args = args[1:]
        elif args[0] == '-e':
            echo_stderr = True
            args = args[1:]
        elif args[0] == '-a':
            if len (args) < 2:
                die ('another argument must come after the "-a" option')
            argv0 = args[1]
            args = args[2:]
        elif args[0] == '--':
            args = args[1:]
            break
        elif args[0][0] == '-':
            die ('unrecognized option "%s"', args[0])
        else:
            # End of option arguments.
            break

    if len (args) < 1:
        print (usage.strip (), file=sys.stderr)
        sys.exit (0)

    subcommand = args[0]
    subargv = args
    if argv0 is not None:
        subargv[0] = argv0

    if use_colors is None:
        use_colors = sys.stdout.isatty ()

    wrapper = Wrapper ()
    wrapper.use_colors = use_colors
    wrapper.echo_stderr = echo_stderr
    sys.exit (wrapper.launch (subcommand, subargv))


if __name__ == '__main__':
    # Note that the standard wrapper created by setup.py does not actually
    # follow this code path! It invokes commandline() directly.
    commandline ()
