# -*- mode: python; coding: utf-8 -*-
# Copyright 2016 Peter Williams <peter@newton.cx> and collaborators.
# Licensed under the MIT License.

"""ciao - running software in the CIAO environment

To use, export an environment variable $PWKIT_CIAO pointing to the CIAO
installation root.


Unpacking data sets
==========================

Data sets are provided as tar files. They unpack to a directory named by the
“obsid” which contains an ``oif.fits`` file and ``primary`` and ``secondary``
subdirectories.


CIAO installation notes
==========================

Download installer script from http://cxc.harvard.edu/ciao/download/. Select
some kind of parent directory like ``/soft/ciao`` for both downloading
tarballs and installing CIAO itself. This may also download and install the
large “caldb” data set. All of the files will end up in a subdirectory such as
``/soft/ciao/ciao-4.8``.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = str("CiaoEnvironment CiaoTool").split()

from ... import PKError
from ...cli import multitool
from ...io import Path
from .. import Environment, prepend_environ_path, user_data_path


class CiaoEnvironment(Environment):
    _installpath = None
    _parampath = None

    def __init__(self, installdir=None, paramdir=None):
        if installdir is None:
            installdir = self._default_installdir()
        if paramdir is None:
            paramdir = user_data_path("cxcds_param")

        self._installpath = Path(installdir).absolute()
        self._parampath = Path(paramdir).absolute()

    def _default_installdir(self):
        import os

        d = os.environ.get("PWKIT_CIAO")
        if d is None:
            raise PKError(
                "CIAO installation directory must be specified "
                "in the $PWKIT_CIAO environment variable"
            )
        return d

    def modify_environment(self, env):
        p = self._installpath

        env["ASCDS_INSTALL"] = str(p)
        env["ASCDS_CONTRIB"] = str(p / "contrib")
        env["ASCDS_BIN"] = str(p / "bin")
        env["ASCDS_LIB"] = str(p / "lib")
        env["ASCDS_IMAGER_PATH"] = str(p / "ots" / "bin")
        env["ASCDS_WORK_PATH"] = str("/tmp")  # needed by at least specextract
        env["CIAO_XPA"] = b"CIAO"
        env["CIAO_PYTHON"] = b"CIAO"
        env["CIAO_APP_PYTHON"] = b"CIAO"
        env["CIAO_IPYTHON"] = b"CIAO"
        env["CIAO_APP_IPYTHON"] = b"CIAO"
        env["CIAO_PYTHON_EXE"] = str(p / "ots" / "bin" / "python")
        env["CIAO_SCRIPT_LANG"] = b"python"
        env["XPA_METHOD"] = b"local"
        env["CALDB"] = str(p / "CALDB")
        env["CALDBCONFIG"] = str(p / "CALDB" / "software" / "tools" / "caldb.config")
        env["CALDBALIAS"] = str(
            p / "CALDB" / "software" / "tools" / "alias_config.fits"
        )
        env["ASCDS_CALIB"] = str(p / "data")
        env["ASCDS_CIAO"] = b"ciao"

        # Obsvis:
        env["OBSVIS_PKG_PATH"] = str(p / "lib" / "tcltk" / "packages" / "obsvis")

        # Sherpa:
        env["CIAO_HEADAS"] = str(p / "ots" / "spectral")
        env["XSPEC_HELP_FILE"] = str(p / "doc" / "xspec.hlp")

        # Proposal tools:
        env["DATA_ROOT"] = str(p / "config")
        env["JCMLIBDATA"] = str(p / "config" / "jcm_data")
        env["ASCDS_PROP_NHBASE"] = env["JCMLIBDATA"]
        env["JCMPATH"] = env["JCMLIBDATA"]
        env["ASCDS_PROP_DATE_DATA"] = env["JCMLIBDATA"]
        env["ASCDS_PROP_PREC_DATA"] = env["JCMLIBDATA"]

        env["PFILES"] = "%s;%s:%s" % (
            self._parampath,
            p / "contrib" / "param",
            p / "param",
        )

        prepend_environ_path(env, "PATH", str(p / "contrib" / "bin"))
        prepend_environ_path(env, "PATH", str(p / "ots" / "bin"))
        prepend_environ_path(env, "PATH", str(p / "bin"))

        return env

    def _preexec(self, env, **kwargs):
        self._parampath.ensure_dir(parents=True)


# Command-line interface

from .. import DefaultExecCommand, DefaultShellCommand


class BgbandCommand(multitool.Command):
    name = "bgband"
    argspec = "<evt> <srcreg> <bkgreg> <elo1> <ehi1> [... <eloN> <ehiN>]"
    summary = "Compute basic background statistics for a source in energy bands."

    def invoke(self, args, envclass=None, **kwargs):
        if len(args) < 5:
            raise multitool.UsageError("bgband takes at least two arguments")
        if len(args) % 2 == 0:
            raise multitool.UsageError("bgband expects an odd number of arguments")

        evt = args[0]
        srcreg = args[1]
        bkgreg = args[2]
        ebins = [(float(args[i]), float(args[i + 1])) for i in range(3, len(args), 2)]

        env = envclass()
        from .analysis import compute_bgband

        df = compute_bgband(evt, srcreg, bkgreg, ebins, env)
        print(df.to_string(index=False, justify="left"))


class CiaoTool(multitool.Multitool):
    cli_name = "pkenvtool ciao"
    summary = "Run tools in the CIAO environment."

    def invoke_command(self, cmd, args, **kwargs):
        return super(CiaoTool, self).invoke_command(
            cmd,
            args,
            envname="ciao",
            envclass=CiaoEnvironment,
            module=__package__,
            **kwargs
        )


def commandline(argv):
    tool = CiaoTool()
    tool.populate(globals().values())
    tool.commandline(argv)
