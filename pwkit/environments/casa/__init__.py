# -*- mode: python; coding: utf-8 -*-
# Copyright 2015-2017 Peter Williams <peter@newton.cx> and collaborators.
# Licensed under the MIT License.

"""The :mod:`pwkit.environments.casa` package provides convenient interfaces to
the `CASA`_ package for analysis of radio interferometric data. In particular,
it makes it much easier to build scripts and modules for automated data
analysis.

.. _CASA: https://casa.nrao.edu/

This module does *not* require a full CASA installation, but it does depend on
the availability of the ``casac`` Python module, which provides Python access
to the C++ code that drives most of CASA’s low-level functionality. By far the
easiest way to obtain this module is to use an installation of `Anaconda or
Miniconda Python`_ and install the `casa-python`_ package provided by Peter
Williams, which builds on the infrastructure provided by the `conda-forge`_
project.

.. _Anaconda or Miniconda Python: http://conda.pydata.org/miniconda.html
.. _casa-python: https://anaconda.org/pkgw-forge/casa-python
.. _conda-forge: https://conda-forge.github.io/

Alternatively, you can try to install CASA and extract the ``casac`` module
from its files `as described here`_. Or you can try to install *this module*
inside the Python environment bundled with CASA. Or you can compile and
underlying CASA C++ code yourself. But, using the pre-built packages is going
to be by far the simplest approach and is **strongly** recommended.

.. _as described here: https://newton.cx/~peter/2014/02/casa-in-python-without-casapy/


Outline of functionality
------------------------

This package provides several kinds of functionality.

- The :mod:`pwkit.environments.casa.tasks` module provides straightforward
  programmatic access to a wide selection of commonly-used CASA takes like
  ``gaincal`` and ``setjy``.
- ``pwkit`` installs a command-line program, ``casatask``, which provides
  command-line access to the tasks implemented in the
  :mod:`~pwkit.environments.casa.tasks` module, much as MIRIAD tasks can be
  driven straight from the command line.
- The :mod:`pwkit.environments.casa.util` module provides the lowest-level access
  to the “tool” structures defined in the C++ code.
- Several modules like :mod:`pwkit.environments.casa.dftphotom` provide
  original analysis features; :mod:`~pwkit.environments.casa.dftphotom`
  extracts light curves of point sources from calibrated visibility data.
- If you do have a full CASA installation available on your compuer, the
  :mod:`pwkit.environments.casa.scripting` module allows you to drive it from
  Python code in a way that allows you to analyze its output, check for
  error conditions, and so on. This is useful for certain features that are not
  currently available in the :mod:`~pwkit.environments.casa.tasks` module.

"""
from __future__ import absolute_import, division, print_function

__all__ = "CasaEnvironment CasaTool commandline".split()

import glob, io, os.path

from ... import PKError, cli
from ...cli import multitool
from .. import Environment, prepend_environ_path, user_data_path


class CasaEnvironment(Environment):
    _rootdir = None

    def __init__(self, rootdir=None):
        if rootdir is None:
            rootdir = self._default_rootdir()

        self._rootdir = os.path.abspath(rootdir)

    def _default_rootdir(self):
        d = os.environ.get("PWKIT_CASA")
        if d is None:
            raise PKError(
                "CASA installation directory must be specified "
                "in the $PWKIT_CASA environment variable"
            )
        return d

    def modify_environment(self, env):
        """Maintaining compatibility with different CASA versions is a pain."""

        # Ugh. I don't see any way out of special-casing the RPM-based
        # installations ... which only exist on NRAO computers, AFAICT.
        # Hardcoding 64-bitness, hopefully that won't come back to bite me.
        is_rpm_install = self._rootdir.startswith("/usr/lib64/casapy/release/")

        def path(*args):
            return os.path.join(self._rootdir, *args)

        env["CASAROOT"] = path()
        env["CASAPATH"] = " ".join(
            [path(), os.uname()[0].lower(), "local", os.uname()[1]]
        )

        if is_rpm_install:
            env["CASA_INSTALLATION_TYPE"] = "rpm-installation"
            prepend_environ_path(env, "PATH", "/usr/lib64/casa/01/bin")
            prepend_environ_path(env, "PATH", path("bin"))
        else:
            env["CASA_INSTALLATION_TYPE"] = "tar-installation"

            lib = "lib64" if os.path.isdir(path("lib64")) else "lib"
            # 4.3.1 comes with both python2.6 and python2.7???
            pydir = sorted(glob.glob(path(lib, "python2*")))[-1]

            tcldir = path("share", "tcl")
            if os.path.isdir(tcldir):
                env["TCL_LIBRARY"] = tcldir
            else:
                tcl_versioned_dirs = glob.glob(path("share", "tcl*"))
                if len(tcl_versioned_dirs):
                    env["TCL_LIBRARY"] = tcl_versioned_dirs[-1]

            bindir = path(lib, "casa", "bin")
            if not os.path.isdir(bindir):
                bindir = path(lib, "casapy", "bin")
            prepend_environ_path(env, "PATH", bindir)

            env["CASA_INSTALLATION_DIRECTORY"] = env["CASAROOT"]
            env["__CASAPY_PYTHONDIR"] = pydir
            env["MATPLOTLIBRC"] = path("share", "matplotlib")
            env["PYTHONHOME"] = env["CASAROOT"]
            env["TK_LIBRARY"] = path("share", "tk")
            env["QT_PLUGIN_PATH"] = path(lib, "qt4", "plugins")

            prepend_environ_path(env, "LD_LIBRARY_PATH", path(lib))
            # should we overwite PYTHONPATH instead?
            prepend_environ_path(
                env, "PYTHONPATH", os.path.join(pydir, "site-packages")
            )
            prepend_environ_path(env, "PYTHONPATH", os.path.join(pydir, "heuristics"))
            prepend_environ_path(env, "PYTHONPATH", pydir)

        return env


# Command-line interface

from .. import DefaultExecCommand, DefaultShellCommand


class CasaTool(multitool.Multitool):
    cli_name = "pkenvtool casa"
    summary = "Run programs in the CASA environment."

    def invoke_command(self, cmd, args, **kwargs):
        return super(CasaTool, self).invoke_command(
            cmd,
            args,
            envname="casa",
            envclass=CasaEnvironment,
            module=__package__,
            **kwargs
        )


def commandline(argv):
    tool = CasaTool()
    tool.populate(globals().values())
    tool.commandline(argv)
