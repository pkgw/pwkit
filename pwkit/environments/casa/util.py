# -*- mode: python; coding: utf-8 -*-
# Copyright 2015-2017 Peter Williams <peter@newton.cx> and collaborators.
# Licensed under the MIT License.

"""This module provides low-level tools and utilities for interacting with the
``casac`` module provided by CASA.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = str(
    """INVERSE_C_MS INVERSE_C_MNS pol_names pol_to_miriad msselect_keys
datadir logger forkandlog sanitize_unicode tools"""
).split()

# Some constants that can be useful.

INVERSE_C_MS = 3.3356409519815204e-09  # inverse speed of light in m/s
INVERSE_C_MNS = 3.3356409519815204  # inverse speed of light in m/ns

pol_names = {
    0: "?",
    1: "I",
    2: "Q",
    3: "U",
    4: "V",
    5: "RR",
    6: "RL",
    7: "LR",
    8: "LL",
    9: "XX",
    10: "XY",
    11: "YX",
    12: "YY",
    13: "RX",
    14: "RY",
    15: "LX",
    16: "LY",
    17: "XR",
    18: "XL",
    19: "YR",
    20: "YL",
    21: "PP",
    22: "PQ",
    23: "QP",
    24: "QQ",
    25: "RCirc",
    26: "Lcirc",
    27: "Lin",
    28: "Ptot",
    29: "Plin",
    30: "PFtot",
    31: "PFlin",
    32: "Pang",
}

pol_to_miriad = {
    # see mirtask.util for the MIRIAD magic numbers.
    1: 1,
    2: 2,
    3: 3,
    4: 4,  # IQUV
    5: -1,
    6: -3,
    7: -4,
    8: -2,  # R/L
    9: -5,
    10: -7,
    11: -8,
    12: -6,  # X/Y
    # rest are inexpressible
}

pol_is_intensity = {
    0: False,
    1: True,
    2: False,
    3: False,
    4: False,  # IQUV
    5: True,
    6: False,
    7: False,
    8: True,  # RR RL LR LL
    9: True,
    10: False,
    11: False,
    12: True,  # XX XY YX YY
    13: False,
    14: False,
    15: False,
    16: False,  # RX RY LX LY
    17: False,
    18: False,
    19: False,
    20: False,  # XR XL YR YL
    21: True,
    22: False,
    23: False,
    24: True,  # PP PQ QP QQ
    25: False,
    26: False,
    27: False,
    28: False,
    29: False,
    30: False,
    31: False,
    32: False,
}

# "polarization" is technically valid as an MS selection, but it pretty much
# doesn't do what you'd want since records generally contain multiple pols.
# ms.selectpolarization() should be used instead. Maybe ditto for spw?

msselect_keys = frozenset(
    "array baseline field observation " "scan scaninent spw taql time uvdist".split()
)


def sanitize_unicode(item):
    """Safely pass string values to the CASA tools.

    item
      A value to be passed to a CASA tool.

    In Python 2, the bindings to CASA tasks expect to receive all string values
    as binary data (:class:`str`) and not Unicode. But :mod:`pwkit` often uses
    the ``from __future__ import unicode_literals`` statement to prepare for
    Python 3 compatibility, and other Python modules are getting better about
    using Unicode consistently, so more and more module code ends up using
    Unicode strings in cases where they might get exposed to CASA. Doing so
    will lead to errors.

    This helper function converts Unicode into UTF-8 encoded bytes for
    arguments that you might pass to a CASA tool. It will leave non-strings
    unchanged and recursively transform collections, so you can safely use it
    just about anywhere.

    I usually import this as just ``b`` and write ``tool.method(b(arg))``, in
    analogy with the ``b''`` byte string syntax. This leads to code such as::

      from pwkit.environments.casa.util import tools, sanitize_unicode as b

      tb = tools.table()
      path = u'data.ms'
      tb.open(path) # => raises exception
      tb.open(b(path)) # => works

    """
    if isinstance(item, str):
        return item.encode("utf8")
    if isinstance(item, dict):
        return dict((sanitize_unicode(k), sanitize_unicode(v)) for k, v in item.items())
    if isinstance(item, (list, tuple)):
        return item.__class__(sanitize_unicode(x) for x in item)

    from ...io import Path

    if isinstance(item, Path):
        return str(item)

    return item


# Finding the data directory


def datadir(*subdirs):
    """Get a path within the CASA data directory.

    subdirs
      Extra elements to append to the returned path.

    This function locates the directory where CASA resource data files (tables
    of time offsets, calibrator models, etc.) are stored. If called with no
    arguments, it simply returns that path. If arguments are provided, they are
    appended to the returned path using :func:`os.path.join`, making it easy to
    construct the names of specific data files. For instance::

      from pwkit.environments.casa import util

      cal_image_path = util.datadir('nrao', 'VLA', 'CalModels', '3C286_C.im')
      tb = util.tools.image()
      tb.open(cal_image_path)

    """
    import os.path

    data = None

    if "CASAPATH" in os.environ:
        data = os.path.join(os.environ["CASAPATH"].split()[0], "data")

    if data is None:
        # The Conda CASA directory layout:
        try:
            import casadef
        except ImportError:
            pass
        else:
            data = os.path.join(os.path.dirname(casadef.task_directory), "data")
            if not os.path.isdir(data):
                # Sigh, hack for CASA 4.7 + Conda; should be straightened out:
                dn = os.path.dirname
                data = os.path.join(
                    dn(dn(dn(casadef.task_directory))), "lib", "casa", "data"
                )
                if not os.path.isdir(data):
                    data = None

    if data is None:
        import casac

        prevp = None
        p = os.path.dirname(casac.__file__)
        while len(p) and p != prevp:
            data = os.path.join(p, "data")
            if os.path.isdir(data):
                break
            prevp = p
            p = os.path.dirname(p)

    if not os.path.isdir(data):
        raise RuntimeError("cannot identify CASA data directory")

    return os.path.join(data, *subdirs)


# Trying to use the logging facility in a sane way.
#
# As soon as you create a logsink, it creates a file called casapy.log.
# So we do some junk to not leave turds all around the filesystem.


def _rmtree_error(func, path, excinfo):
    from ...cli import warn

    warn("couldn't delete temporary file %s: %s (%s)", path, excinfo[0], func)


def logger(filter="WARN"):
    """Set up CASA to write log messages to standard output.

    filter
      The log level filter: less urgent messages will not be shown. Valid values
      are strings: "DEBUG1", "INFO5", ... "INFO1", "INFO", "WARN", "SEVERE".

    This function creates and returns a CASA ”log sink” object that is
    configured to write to standard output. The default CASA implementation
    would *always* create a file named ``casapy.log`` in the current
    directory; this function safely prevents such a file from being left
    around. This is particularly important if you don’t have write permissions
    to the current directory.

    """
    import os, shutil, tempfile

    cwd = os.getcwd()
    tempdir = None

    try:
        tempdir = tempfile.mkdtemp(prefix="casautil")

        try:
            os.chdir(tempdir)
            sink = tools.logsink()
            sink.setlogfile(sanitize_unicode(os.devnull))
            try:
                os.unlink("casapy.log")
            except OSError as e:
                if e.errno != 2:
                    raise
                # otherwise, it's a ENOENT, in which case, no worries.
        finally:
            os.chdir(cwd)
    finally:
        if tempdir is not None:
            shutil.rmtree(tempdir, onerror=_rmtree_error)

    sink.showconsole(True)
    sink.setglobal(True)
    sink.filter(sanitize_unicode(filter.upper()))
    return sink


def forkandlog(function, filter="INFO5", debug=False):
    """Fork a child process and read its CASA log output.

    function
      A function to run in the child process
    filter
      The CASA log level filter to apply in the child process: less urgent
      messages will not be shown. Valid values are strings: "DEBUG1", "INFO5",
      ... "INFO1", "INFO", "WARN", "SEVERE".
    debug
      If true, the standard output and error of the child process are *not*
      redirected to /dev/null.

    Some CASA tools produce important results that are *only* provided via log
    messages. This is a problem for automation, since there’s no way for
    Python code to intercept those log messages and extract the results of
    interest. This function provides a framework for working around this
    limitation: by forking a child process and sending its log output to a
    pipe, the parent process can capture the log messages.

    This function is a generator. It yields lines from the child process’ CASA
    log output.

    Because the child process is a fork of the parent, it inherits a complete
    clone of the parent’s state at the time of forking. That means that the
    *function* argument you pass it can do just about anything you’d do in a
    regular program.

    The child process’ standard output and error streams are redirected to
    ``/dev/null`` unless the *debug* argument is true. Note that the CASA log
    output is redirected to a pipe that is neither of these streams. So, if
    the function raises an unhandled Python exception, the Python traceback
    will not pollute the CASA log output. But, by the same token, the calling
    program will not be able to detect that the exception occurred except by
    its impact on the expected log output.

    """
    import sys, os

    readfd, writefd = os.pipe()
    pid = os.fork()

    if pid == 0:
        # Child process. We never leave this branch.
        #
        # Log messages of priority >WARN are sent to stderr regardless of the
        # status of log.showconsole(). The idea is for this subprocess to be
        # something super lightweight and constrained, so it seems best to
        # nullify stderr, and stdout, to not pollute the output of the calling
        # process.
        #
        # I thought of using the default logger() setup and dup2'ing stderr to
        # the pipe fd, but then if anything else gets printed to stderr (e.g.
        # Python exception info), it'll get sent along the pipe too. The
        # caller would have to be much more complex to be able to detect and
        # handle such output.

        os.close(readfd)

        if not debug:
            f = open(os.devnull, "w")
            os.dup2(f.fileno(), 1)
            os.dup2(f.fileno(), 2)

        sink = logger(filter=filter)
        sink.setlogfile(b"/dev/fd/%d" % writefd)
        function(sink)
        sys.exit(0)

    # Original process.

    os.close(writefd)

    with os.fdopen(readfd) as readhandle:
        for line in readhandle:
            yield line

    info = os.waitpid(pid, 0)

    if info[1]:
        # Because we're a generator, this is the only way for us to signal if
        # the process died. We could be rewritten as a context manager.
        e = RuntimeError(
            "logging child process PID %d exited " "with error code %d" % tuple(info)
        )
        e.pid, e.exitcode = info
        raise e


# Tool factories.


class _Tools(object):
    """This class is structured so that it supports useful tab-completion
    interactively, but also so that new tools can be constructed if the
    underlying library provides them.

    """

    _builtinNames = """agentflagger atmosphere calanalysis calibrater calplot
                     componentlist coordsys deconvolver fitter flagger
                     functional image imagepol imager logsink measures
                     msmetadata ms msplot mstransformer plotms regionmanager
                     simulator spectralline quanta table tableplot utils
                     vlafiller vpmanager""".split()

    def __getattribute__(self, n):
        """Returns factories, not instances."""
        # We need to make this __getattribute__, not __getattr__, only because
        # we set the builtin names in the class __dict__ to enable tab-completion.
        import casac

        if hasattr(casac, "casac"):  # casapy >= 4.0?
            t = getattr(casac.casac, n, None)
            if t is None:
                raise AttributeError('tool "%s" not present' % n)
            return t
        else:
            try:
                return casac.homefinder.find_home_by_name(n + "Home").create
            except Exception:
                # raised exception is class 'homefinder.error'; it appears unavailable
                # on the Python layer
                raise AttributeError('tool "%s" not present' % n)


for n in _Tools._builtinNames:
    setattr(_Tools, n, None)  # ease autocompletion

tools = _Tools()
