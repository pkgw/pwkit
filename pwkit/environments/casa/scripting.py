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

__all__ = (b'launch_script commandline').split ()

import os.path, shutil, signal, sys, tempfile
from ... import cli, reraise_context
from . import CasaEnvironment


casapy_argv = ['casapy', '--log2term', '--nogui', '-c']

signals_for_child = [
    signal.SIGHUP,
    signal.SIGINT,
    signal.SIGQUIT,
    signal.SIGTERM,
    signal.SIGUSR1,
    signal.SIGUSR2,
]

def launch_script (script, args):
    """Launch a script in the casapy environment. This involves creating a
    temporary wrapper and then using the CasaEnvironment to run it in a
    temporary directory.

    """
    # We read in the entire script and save it in the wrapper. That way we
    # don't have to worry about dealing with file-not-found errors inside
    # casapy, where exception handling is annoying and the startup time is
    # significant.

    try:
        with open (script) as f:
            text = f.read ()
    except Exception:
        reraise_context ('while trying to read %r', script)

    workdir = tempfile.mkdtemp (prefix='casascript', dir='.')
    wrapped = os.path.join (workdir, 'wrapped.py')

    with open (wrapped, 'wb') as wrapper:
        print ('_pkcs_script = ' + repr (script), file=wrapper)
        print ('_pkcs_text = ' + repr (text), file=wrapper)
        print ('_pkcs_args = ' + repr (args), file=wrapper)
        print ('_pkcs_origcwd = ' + repr (os.getcwd ()), file=wrapper)

        driver = __file__.replace ('.pyc', '.py').replace ('.py', '_driver.py')
        with open (driver) as driver:
            for line in driver:
                print (line, end='', file=wrapper)

    def preexec ():
        # Start new session and process groups so that the module can kill all
        # CASA-related processes.
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

        with open ('pgid', 'wb') as pgid:
            print (os.getpid (), file=pgid)
            print (os.getpgrp (), file=pgid)

    env = CasaEnvironment ()
    proc = env.launch (casapy_argv + ['wrapped.py'],
                       cwd=workdir,
                       stdin=open (os.devnull, 'rb'),
                       preexec_fn=preexec,
                       close_fds=False)

    # Set up signal handlers to propagate to the child process. Copied from
    # wrapout.py.
    prev_handlers = {}

    def handle (signum, frame):
        proc.send_signal (signum)

    for signum in signals_for_child:
        prev_handlers[signum] = signal.signal (signum, handle)

    # Let's see what happened ...
    exitcode = proc.wait ()

    for signum, prev_handler in prev_handlers.iteritems ():
        signal.signal (signum, prev_handler)

    rmtree = (exitcode == 0 or exitcode == 127) # success or intentional script abort

    if not rmtree:
        cli.warn ('preserving directory tree %r since script %r failed',
                  workdir, script)
    else:
        shutil.rmtree (workdir, ignore_errors=True)

    # Make sure all child processes are cleaned up
    for sig in signal.SIGTERM, signal.SIGKILL:
        try:
            os.kill (-proc.pid, sig)
        except:
            pass

    return exitcode


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
    scriptargs = argv[2:]

    try:
        exitcode = launch_script (script, scriptargs)
    except Exception as e:
        cli.die ('couldn\'t launch %r: %s', script, e)

    if exitcode < 0:
        signum = -exitcode
        print ('error: casapy died with signal %d' % signum)
        signal.signal (signum, signal.SIG_DFL)
        os.kill (os.getpid (), signum)
    elif exitcode:
        print ('error: casapy died with exit code %d' % exitcode)
        sys.exit (exitcode)
