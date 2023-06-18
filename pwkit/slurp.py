# -*- mode: python; coding: utf-8 -*-
# Copyright 2015 Peter Williams <peter@newton.cx> and collaborators
# Licensed under the MIT License.

"""pwkit.slurp - run a program and capture its output."""

from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = str("Event Redirection Slurper").split()

import fcntl, os, signal, subprocess
from select import select, error as selecterror

from . import Holder

try:
    from subprocess import DEVNULL as _DEVNULL
except ImportError:
    _DEVNULL = subprocess.STDOUT - 1


@Holder
class Event(object):
    Stdout = "stdout"
    Stderr = "stderr"
    ForwardedSignal = "forwarded-signal"
    Timeout = "timeout"


@Holder
class Redirection(object):
    Pipe = subprocess.PIPE
    Stdout = subprocess.STDOUT
    DevNull = _DEVNULL


signals_for_child = [
    signal.SIGHUP,
    signal.SIGINT,
    signal.SIGQUIT,
    signal.SIGTERM,
    signal.SIGUSR1,
    signal.SIGUSR2,
]


class SlurperIterator(object):
    def __init__(self, parent):
        self.parent = parent

    def __iter__(self):
        return self

    def __next__(self):  # Python 3
        if not len(self.parent._files):
            raise StopIteration()
        return self.parent._next_lowlevel()

    next = __next__  # Python 2


def _decode_streams(event_source, which_events, encoding):
    from codecs import getincrementaldecoder

    decoders = {}

    for etype, edata in event_source:
        if etype not in which_events:
            yield etype, edata
            continue

        dec = decoders.get(etype)
        if dec is None:
            dec = decoders[etype] = getincrementaldecoder(encoding)()

        final = not len(edata)
        result = dec.decode(edata, final)
        if len(result):
            yield etype, result  # no false EOF indicators

        if final:
            yield etype, edata  # make sure we have an EOF signal


def _linebreak_streams(event_source, which_events):
    partials = {}

    for etype, edata in event_source:
        if etype not in which_events:
            yield etype, edata
            continue

        if not len(edata):
            # EOF on this stream.
            trailer = partials.get(etype, edata)
            if len(trailer):
                yield etype, trailer
            yield etype, edata
            continue

        lines = (partials.get(etype, edata * 0) + edata).split(edata.__class__(b"\n"))
        for line in lines[:-1]:
            yield etype, line
        partials[etype] = lines[-1]


class Slurper(object):
    _chunksize = 1024

    def __init__(
        self,
        argv=None,
        env=None,
        cwd=None,
        propagate_signals=True,
        timeout=10,
        linebreak=False,
        encoding=None,
        stdin=Redirection.DevNull,
        stdout=Redirection.Pipe,
        stderr=Redirection.Pipe,
        executable=None,
        subproc_factory=None,
    ):
        if subproc_factory is None:
            subproc_factory = subprocess.Popen

        self.subproc_factory = subproc_factory
        self.proc = None
        self.argv = argv
        self.env = env
        self.cwd = cwd
        self.propagate_signals = propagate_signals
        self.timeout = timeout
        self.linebreak = linebreak
        self.encoding = encoding
        self.stdin = stdin
        self.stdout = stdout
        self.stderr = stderr
        self.executable = executable

    def __enter__(self):
        self._prev_handlers = {}
        self._other_events = []
        self._file_event_types = {}
        self._files = []

        stdin = self.stdin
        if stdin == Redirection.DevNull:
            stdin = open(os.devnull, "r")

        stdout = self.stdout
        if stdout == Redirection.DevNull:
            stdout = open(os.devnull, "w")

        stderr = self.stderr
        if stderr == Redirection.DevNull:
            stderr = open(os.devnull, "w")

        self.proc = self.subproc_factory(
            self.argv,
            env=self.env,
            executable=self.executable,
            cwd=self.cwd,
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            shell=False,
        )

        if self.propagate_signals:

            def handle(signum, frame):
                self.proc.send_signal(signum)
                self._other_events.insert(0, (Event.ForwardedSignal, signum))

            for signum in signals_for_child:
                self._prev_handlers[signum] = signal.signal(signum, handle)

        if stdout == Redirection.Pipe:
            self._file_event_types[self.proc.stdout.fileno()] = Event.Stdout
            self._files.append(self.proc.stdout)

        if stderr == Redirection.Pipe:
            self._file_event_types[self.proc.stderr.fileno()] = Event.Stderr
            self._files.append(self.proc.stderr)

        for fd in self._files:
            fl = fcntl.fcntl(fd.fileno(), fcntl.F_GETFL)
            fcntl.fcntl(fd.fileno(), fcntl.F_SETFL, fl | os.O_NONBLOCK)

        return self

    def _next_lowlevel(self):
        if len(self._other_events):
            return self._other_events.pop()

        while True:
            try:
                rd, wr, er = select(self._files, [], [], self.timeout)
                break
            except selecterror as e:
                # if EINTR or EAGAIN, try again; we won't get EINTR unless
                # we're forwarding signals, since otherwise it'll show up as a
                # KeyboardInterrupt. "e.args[0]" is the only way to get errno.
                if e.args[0] not in (4, 11):
                    raise

        for fd in rd:
            chunk = fd.read(self._chunksize)
            if not len(chunk):
                self._files.remove(fd)
            return (self._file_event_types[fd.fileno()], chunk)

        return (Event.Timeout, None)

    def __iter__(self):
        result = SlurperIterator(self)

        if self.encoding is not None:
            which = frozenset((Event.Stdout, Event.Stderr))
            result = _decode_streams(result, which, self.encoding)

        if self.linebreak:
            which = frozenset((Event.Stdout, Event.Stderr))
            result = _linebreak_streams(result, which)

        return result

    def __exit__(self, etype, evalue, etb):
        self.proc.wait()

        for signum, prev_handler in self._prev_handlers.items():
            signal.signal(signum, prev_handler)

        return False
