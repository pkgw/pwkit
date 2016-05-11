# -*- mode: python; coding: utf-8 -*-
# Copyright 2015 Peter Williams <peter@newton.cx> and collaborators.
# Licensed under the MIT License.

"""pwkit.environments.casa.scripting - scripted invocation of casapy.

The "casapy" program is extremely resistant to encapsulated scripting -- it
pops up GUI windows and child processes, leaves log files around, provides a
non-vanilla Python environment, and so on. However, sometimes scripting CASA
is what we need to do. This tool enables that.

We provide a single-purpose CLI tool for this functionality, so that you can
write standalone scripts with a hashbang line of "#! /usr/bin/env
pkcasascript" -- hashbang lines support only one extra command-line
argument, so if we're using "env" we can't take a multitool approach.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = str ('CasapyScript commandline').split ()

import os.path, shutil, signal, six, sys, tempfile
from ... import PKError, cli, reraise_context
from . import CasaEnvironment


casapy_argv = ['casa', '--log2term', '--nogui', '-c']

signals_for_child = [
    signal.SIGHUP,
    signal.SIGINT,
    signal.SIGQUIT,
    signal.SIGTERM,
    signal.SIGUSR1,
    signal.SIGUSR2,
]


class CasapyScript (object):
    """Context manager for launching a script in the casapy environment. This
    involves creating a temporary wrapper and then using the CasaEnvironment
    to run it in a temporary directory.

    When this context manager is entered, the script is launched and the
    calling process waits until it finishes. This object is returned. The
    `with` statement body is then executed so that information can be
    extracted from the results of the casapy invocation. When the context
    manager is exited, the casapy files are (usually) cleaned up.

    Attributes:

    args
      the arguments to passed to the script.
    env
      the CasaEnvironment used to launch the casapy process.
    exitcode
      the exit code of the casapy process. 0 is success. 127 indicates
      an intentional error exit by the script; additional diagnostics
      don't need printing and the work directory doesn't need
      preservation. Negative values indicate death from a signal.
    proc
      the `subprocess.Popen` instance of casapy; inside the context
      manager body it's already exited.
    rmtree
      boolean; whether to delete the working tree upon context manager
      exit.
    script
      the path to the script to be invoked.
    workdir
      the working directory in which casapy was started.
    wrapped
      the path to the wrapper script run inside casapy.

    There is a very large overhead to running casapy scripts. The outer Python
    code sleeps for at least 5 seconds to allow various cleanups to happen.

    """
    def __init__ (self, script, raise_on_error=True, **kwargs):
        self.script = script
        self.kwargs = kwargs
        self.raise_on_error = raise_on_error


    def __enter__ (self):
        # We read in the entire script and save it in the wrapper. That way we
        # don't have to worry about dealing with file-not-found errors inside
        # casapy, where exception handling is annoying and the startup time is
        # significant.

        try:
            with open (self.script) as f:
                text = f.read ()
        except Exception:
            reraise_context ('while trying to read %r', self.script)

        self.workdir = tempfile.mkdtemp (prefix='casascript', dir='.')
        self.wrapped = os.path.join (self.workdir, 'wrapped.py')

        with open (self.wrapped, 'wb') as wrapper:
            print ('_pkcs_script = ' + repr (self.script), file=wrapper)
            print ('_pkcs_text = ' + repr (text), file=wrapper)
            print ('_pkcs_kwargs = ' + repr (self.kwargs), file=wrapper)
            print ('_pkcs_origcwd = ' + repr (os.getcwd ()), file=wrapper)

            driver = __file__.replace ('.pyc', '.py').replace ('.py', '_driver.py')
            with open (driver) as driver:
                for line in driver:
                    print (line, end='', file=wrapper)

        def preexec ():
            # Start new session and process groups so that the module can kill all
            # CASA-related processes as best we can.
            os.setsid ()

            # We want to direct casapy's stdout and stderr to separate files since
            # they're full of chatter, while still giving the script access to
            # intentional output on the wrapper's stdout and stderr. At this
            # point, FD's 1 and 2 are the latter. We want to move them to FD's 3
            # and 4, while changing 1 and 2 to the temp files. The close_fds logic
            # of subprocess runs after this function, so we have to set close_fds
            # to False.

            os.dup2 (1, 3) # dup2 closes target fd if needed.
            os.dup2 (2, 4)

            with open ('casa_stdout', 'wb') as stdout:
                os.dup2 (stdout.fileno (), 1)

            with open ('casa_stderr', 'wb') as stderr:
                os.dup2 (stderr.fileno (), 2)

        self.env = CasaEnvironment ()
        self.proc = self.env.launch (casapy_argv + ['wrapped.py'],
                                     cwd=self.workdir,
                                     stdin=open (os.devnull, 'rb'),
                                     preexec_fn=preexec,
                                     close_fds=False)

        # Set up signal handlers to propagate to the child process. Copied from
        # wrapout.py.
        prev_handlers = {}

        def handle (signum, frame):
            self.proc.send_signal (signum)

        for signum in signals_for_child:
            prev_handlers[signum] = signal.signal (signum, handle)

        self.exitcode = self.proc.wait ()

        for signum, prev_handler in six.iteritems (prev_handlers):
            signal.signal (signum, prev_handler)

        # default: delte workdir on success or intentional script abort
        self.rmtree = (self.exitcode == 0 or self.exitcode == 127)

        if self.raise_on_error and self.exitcode != 0:
            # Note that we have to raise the exception here to prevent the
            # `with` statement body from executing. In that case __exit__
            # isn't called so we need to do that too.
            self._cleanup ()

            if self.exitcode < 0:
                raise PKError ('casapy was killed by signal %d', -self.exitcode)
            elif self.exitcode == 127:
                raise PKError ('the casapy script signaled an internal error')
            else:
                raise PKError ('casapy exited with error code %d', self.exitcode)

        return self


    def __exit__ (self, etype, evalue, etb):
        if etype is not None:
            self.rmtree = False

        self._cleanup ()
        return False # propagate exceptions


    def _cleanup (self):
        # Ugh, I hate having a hardcoded sleep, but it seems to be necessary
        # to let the watchdog clean everything up. Or something. The casapy
        # process tree is a mess several process groups being created, and I
        # think the only way we can really contain it is with cgroups, which
        # would be difficult and make us Linux-specific. Grrr.
        import time
        time.sleep (4)

        # If I'm interpreting things correctly, this bit is needed to kill
        # the "watchdog" process.

        try:
            os.killpg (self.proc.pid, signal.SIGTERM)
        except Exception as e:
            pass

        time.sleep (1)

        try:
            os.killpg (self.proc.pid, signal.SIGKILL)
        except Exception as e:
            pass

        # OK, blow away the directory.

        if not self.rmtree:
            cli.warn ('preserving directory tree %r since script %r failed',
                      self.workdir, self.script)
        else:
            shutil.rmtree (self.workdir, ignore_errors=True)


cli_usage = """pkcasascript <scriptfile> [more args...]

Run a specially-designed script inside a CASA environment. This program is not
meant for regular users. See the documentation of the module
`pwkit.environments.casa.scripting` for more information."""

def commandline (argv=None):
    if argv is None:
        argv = sys.argv
        cli.propagate_sigint ()
        cli.unicode_stdio ()
        cli.backtrace_on_usr1 ()

    cli.check_usage (cli_usage, argv, usageifnoargs='long')
    script = argv[1]
    args = argv[2:]

    try:
        with CasapyScript (script, cli_args=args) as cs:
            pass
    except Exception:
        reraise_context ('when running casapy script %r', script)

    if cs.exitcode < 0:
        signum = -cs.exitcode
        print ('casascript error: casapy died with signal %d' % signum)
        signal.signal (signum, signal.SIG_DFL)
        os.kill (os.getpid (), signum)
    elif cs.exitcode:
        if cs.exitcode != 127:
            print ('casascript error: casapy died with exit code %d' % cs.exitcode)
        sys.exit (cs.exitcode)
