# -*- mode: python; coding: utf-8 -*-
# Copyright 2015-2017 Peter Williams <peter@newton.cx> and collaborators.
# Licensed under the MIT License.

"""This module provides an encapsulated scheme for running HEASoft tools
within the :mod:`pwkit.environments` framework.

This module sets things up such that parameter files for HEASoft tasks
(“pfiles”) land in the directory ``~/.local/share/hea-pfiles/``.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = str ('').split ()

import os.path

from .. import PKError, io
from . import Environment, prepend_environ_path, user_data_path


class HeasoftEnvironment (Environment):
    _installdir = None
    _platform = None

    def __init__ (self, platformdir=None):
        if platformdir is None:
            platformdir = self._default_platformdir ()

        self._installdir = os.path.abspath (os.path.dirname (platformdir))
        self._platform = os.path.basename (platformdir)


    def _default_platformdir (self):
        d = os.environ.get ('PWKIT_HEASOFT')
        if d is None:
            raise PKError ('HEAsoft installation directory must be specified '
                           'in the $PWKIT_HEASOFT environment variable')
        return d


    def modify_environment (self, env):
        """The headas-init.sh script generates its variables in a bit of a funky way
        -- it runs a script that generates a list of settings. These are their
        transcriptions.

        """
        plat = self._platform

        def path (*args):
            return os.path.join (self._installdir, *args)

        env['CALDB'] = b'http://heasarc.gsfc.nasa.gov/FTP/caldb'
        env['CALDBCONFIG'] = path ('caldb.config')
        env['CALDBALIAS'] = path ('alias_config.fits')

        env['HEADAS'] = path (plat)
        env['LHEASOFT'] = env['HEADAS']
        env['FTOOLS'] = env['HEADAS']

        prepend_environ_path (env, 'PATH', path (plat, 'bin'))
        prepend_environ_path (env, 'LD_LIBRARY_PATH', path (plat, 'lib'))
        prepend_environ_path (env, 'PERLLIB', path (plat, 'lib', 'perl'))
        prepend_environ_path (env, 'PERL5LIB', path (plat, 'lib', 'perl'))
        prepend_environ_path (env, 'PYTHONPATH', path (plat, 'lib'))
        prepend_environ_path (env, 'PYTHONPATH', path (plat, 'lib', 'python'))

        userpfiles = user_data_path ('hea-pfiles')
        io.ensure_dir (userpfiles, parents=True)
        env['PFILES'] = ';'.join ([userpfiles,
                                   path (plat, 'syspfiles')])

        env['LHEA_DATA'] = path (plat, 'refdata')
        env['LHEA_HELP'] = path (plat, 'help')
        env['PGPLOT_DIR'] = path (plat, 'lib')
        env['PGPLOT_FONT'] = path (plat, 'lib', 'grfont.dat')
        env['PGPLOT_RGB'] = path (plat, 'lib', 'rgb.txt')
        env['POW_LIBRARY'] = path (plat, 'lib', 'pow')
        env['TCLRL_LIBDIR'] = path (plat, 'lib')
        env['XANADU'] = path ()
        env['XANBIN'] = path (plat)
        env['XRDEFAULTS'] = path (plat, 'xrdefaults')

        env['EXT'] = b'lnx' # XXX portability probably ...
        env['LHEAPERL'] = b'/usr/bin/perl' # what could go wrong?
        env['PFCLOBBER'] = b'1'
        env['FTOOLSINPUT'] = b'stdin'
        env['FTOOLSOUTPUT'] = b'stdout'
        return env
