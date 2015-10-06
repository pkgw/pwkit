# -*- mode: python; coding: utf-8 -*-
# Copyright 2015 Peter Williams <peter@newton.cx> and collaborators.
# Licensed under the MIT License.

"""Do not use this file.

This file is installed like a module in pwkit, but it's not meant to be used
by pwkit users! Rather, it's executed inside casapy by the pkcasascript
system. It's thus running inside CASA's special IPython instance that is
built on a weird, different Python interpreter and has a bunch of special
globals and such.

When we're invoked, the following variables are set (by inserting their
reprs before the beginning of this file).

  _pkcs_script  - the path the script file to be run
  _pkcs_text    - the contents of that file -- allows existence-checking to
                  happen before casapy is started
  _pkcs_kwargs  - keyword arguments to the script. If it was launched from
                  the CLI, the command-line arguments are available as a
                  list "cli_args", not including argv[0].
  _pkcs_origcwd - the original working directory of the script; we chdir()
                  on startup to hide the log files.
"""

running_in_casapy = 'casalog' in locals ()

# We're not allowed to put "from __future__ import print_function" at the top
# of this file since other lines will be prepended, so we gain access to
# print() this way:
import __builtin__
printfn = getattr (__builtin__, 'print')

if running_in_casapy:
    import os
    script_stdout = os.fdopen (3, 'wb')
    script_stderr = os.fdopen (4, 'wb')
    __rethrow_casa_exceptions = True # Amateurs.

class CasaNamespace (object):
    def __init__ (self):
        self.__dict__ = globals ()

class CasapyScriptHelper (object):
    """A helper object passed to the script being run within casapy.

    Methods:

    die  (fmt, *args)  - print an error and exit
    log  (fmt, *args)  - print to standard output
    warn (fmt, *args)  - print a warning
    temppath (*pieces) - generate a path within `tempdir`

    Attributes:

    casans      - The populated casapy namespace, with items accessible
                  as (e.g.) `helper.casans.clean()`.
    script_path - The original path of the script being run.
    tempdir     - The path of the temporary directory.

    """
    def __init__ (self, script_path, tempdir):
        self.script_path = script_path
        self.tempdir = tempdir
        self.casans = CasaNamespace ()

    def log (self, fmt, *args, **kwargs):
        if len (args):
            text = fmt % args
        else:
            text = str (fmt)
        printfn (text, file=script_stdout, **kwargs)

    def warn (self, fmt, *args, **kwargs):
        if len (args):
            text = fmt % args
        else:
            text = str (fmt)
        printfn ('warning:', text, file=script_stderr, **kwargs)

    def die (self, fmt, *args):
        if len (args):
            text = fmt % args
        else:
            text = str (fmt)

        printfn ('error:', text, file=script_stderr)
        os._exit (127) # indicates internal script abort, not problem running script

    def temppath (self, *args):
        return os.path.join (self.tempdir, *args)


def _pkcs_inner ():
    # Get back to the directory where we were invoked, rather than the
    # temporary directory where we hide CASA's logfiles:
    import os
    tempdir = os.getcwd ()
    os.chdir (_pkcs_origcwd)

    helper = CasapyScriptHelper (_pkcs_script, tempdir)

    try:
        code = compile (_pkcs_text, _pkcs_script, 'exec')
    except Exception as e:
        helper.die ('cannot compile script %r: %s', _pkcs_script, e)

    ns = {'__file__': _pkcs_script,
          '__name__': '__casascript__'}

    try:
        exec (code, ns)
    except Exception as e:
        raise

    in_casapy_func = ns.get ('in_casapy')
    if in_casapy_func is None:
        helper.die ('script %r does not contain a function called "in_casapy"',
                    _pkcs_script)
    if not callable (in_casapy_func):
        helper.die ('script %r provides something called "in_casapy", but '
                    'it\'s not a function', _pkcs_script)

    in_casapy_func (helper, **_pkcs_kwargs)


if running_in_casapy:
    try:
        _pkcs_inner ()
    except:
        import os, sys, traceback
        class Prefixer (object):
            def __init__ (self, inner):
                self.inner = inner
            def write (self, text):
                self.inner.write ('casascript: ')
                self.inner.write (text)
        printfn ('casascript: unhandled exception within casapy:', file=script_stderr)
        traceback.print_exception (*sys.exc_info (), file=Prefixer (script_stderr))
        os._exit (1)
