# -*- mode: python; coding: utf-8 -*-
# Copyright 2015 Peter Williams <peter@newton.cx> and collaborators.
# Licensed under the MIT License.

"""heasoft - running software in the HEAsoft/CALDB environment

To use, export an environment variable $PWKIT_HEASOFT pointing to the HEAsoft
platform-specific directory, usually known as $HEADAS. E.g.,
$PWKIT_HEASOFT/bin, $PWKIT_HEASOFT/BUILD_DIR, and
$PWKIT_HEASOFT/headas-init.sh should exist. CALDB also needs to be set up as
described below.

"pfiles" are set up to land in ~/.local/share/hea-pfiles/.


HEAsoft installation notes
==========================

(All examples assume version 6.16 for convenience, substitute as needed of
course.)

Installation from source strongly recommended. Download from something like
http://heasarc.gsfc.nasa.gov/FTP/software/lheasoft/release/heasoft-6.16src.tar.gz
- the website lets you customize the tarball, but it's probably easiest just
to do the full install every time. Tarball unpacks into `heasoft-6.16/...` so
you can safely curl|tar in ~/sw/.

$ cd heasoft-6.16/BUILD_DIR
$ ./configure --prefix=/a/heasoft/6.16
$ make # note: not parallel-friendly
$ make install

The CALDB setup is so lightweight that it's not worth separating it out:

$ cd /a/heasoft/6.16
$ wget http://heasarc.gsfc.nasa.gov/FTP/caldb/software/tools/caldb.config
$ wget http://heasarc.gsfc.nasa.gov/FTP/caldb/software/tools/alias_config.fits

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

        env[b'CALDB'] = b'http://heasarc.gsfc.nasa.gov/FTP/caldb'
        env[b'CALDBCONFIG'] = path ('caldb.config')
        env[b'CALDBALIAS'] = path ('alias_config.fits')

        env[b'HEADAS'] = path (plat)
        env[b'LHEASOFT'] = env[b'HEADAS']
        env[b'FTOOLS'] = env[b'HEADAS']

        prepend_environ_path (env, b'PATH', path (plat, 'bin'))
        prepend_environ_path (env, b'LD_LIBRARY_PATH', path (plat, 'lib'))
        prepend_environ_path (env, b'PERLLIB', path (plat, 'lib', 'perl'))
        prepend_environ_path (env, b'PERL5LIB', path (plat, 'lib', 'perl'))
        prepend_environ_path (env, b'PYTHONPATH', path (plat, 'lib'))
        prepend_environ_path (env, b'PYTHONPATH', path (plat, 'lib', 'python'))

        userpfiles = user_data_path ('hea-pfiles')
        io.ensure_dir (userpfiles, parents=True)
        env[b'PFILES'] = ';'.join ([userpfiles,
                                    path (plat, 'syspfiles')])

        env[b'LHEA_DATA'] = path (plat, 'refdata')
        env[b'LHEA_HELP'] = path (plat, 'help')
        env[b'PGPLOT_DIR'] = path (plat, 'lib')
        env[b'PGPLOT_FONT'] = path (plat, 'lib', 'grfont.dat')
        env[b'PGPLOT_RGB'] = path (plat, 'lib', 'rgb.txt')
        env[b'POW_LIBRARY'] = path (plat, 'lib', 'pow')
        env[b'TCLRL_LIBDIR'] = path (plat, 'lib')
        env[b'XANADU'] = path ()
        env[b'XANBIN'] = path (plat)
        env[b'XRDEFAULTS'] = path (plat, 'xrdefaults')

        env[b'EXT'] = b'lnx' # XXX portability probably ...
        env[b'LHEAPERL'] = b'/usr/bin/perl' # what could go wrong?
        env[b'PFCLOBBER'] = b'1'
        env[b'FTOOLSINPUT'] = b'stdin'
        env[b'FTOOLSOUTPUT'] = b'stdout'
        return env
