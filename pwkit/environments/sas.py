# -*- mode: python; coding: utf-8 -*-
# Copyright 2015 Peter Williams <peter@newton.cx> and collaborators.
# Licensed under the MIT License.

"""sas - running software in the SAS environment

To use, export an environment variable $PWKIT_SAS pointing to the SAS
installation root. The files $PWKIT_SAS/RELEASE and $PWKIT_SAS/setsas.sh
should exist. The "current calibration files" (CCF) should be accessible as
$PWKIT_SAS/ccf/; a symlink may make sense if multiple SAS versions are going
to be used.

SAS is unusual because you need to set up some magic environment variables
specific to the dataset that you're working with. There is also default
preparation to be run on each dataset before anything useful can be done.


Unpacking data sets
==========================

Data sets are downloaded as tar.gz files. Those unpack to a few files in '.'
including a .TAR file, which should be unpacked too. That unpacks to a bunch
of data files in '.' as well.


SAS installation notes
==========================

Download tarball from, e.g.,

ftp://legacy.gsfc.nasa.gov/xmm/software/sas/14.0.0/64/Linux/Fedora20/

Tarball unpacks installation script and data into '.', and the installation
script sets up a SAS install in a versioned subdirectory of '.', so curl|tar
should be run from something like /a/sas.

$ ./install.sh

The CCF are like CALDB and need to be rsynced -- see the update-ccf
subcommand.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = (b'').split ()

import io, os.path

from .. import PKError, cli
from . import Environment, prepend_environ_path, user_data_path


class SasEnvironment (Environment):
    _odfdir = None
    _revnum = None
    _obsid = None
    _sumsas = None
    _installdir = None
    _heaenv = None

    def __init__ (self, manifest, installdir=None, heaenv=None):
        if installdir is None:
            installdir = self._default_installdir ()
        if heaenv is None:
            from . import heasoft
            heaenv = heasoft.HeasoftEnvironment ()

        self._installdir = os.path.abspath (installdir)
        self._heaenv = heaenv

        with io.open (manifest, 'rt') as f:
            for line in f:
                if not line.startswith ('File '):
                    continue

                bits = line.split ()[1].split ('_')
                if len (bits) < 3:
                    continue

                self._revnum = bits[0] # note: kept as a string; not an int
                self._obsid = bits[1]
                break

        self._odfdir = os.path.abspath (os.path.dirname (manifest))
        self._sumsas = os.path.join (self._odfdir,
                                     '%s_%s_SCX00000SUM.SAS' % (self._revnum,
                                                                self._obsid))


    def _default_installdir (self):
        d = os.environ.get ('PWKIT_SAS')
        if d is None:
            raise PKError ('SAS installation directory must be specified '
                           'in the $PWKIT_SAS environment variable')
        return d


    def modify_environment (self, env):
        self._heaenv.modify_environment (env)

        def path (*args):
            return os.path.join (self._installdir, *args)

        env[b'SAS_DIR'] = path ()
        env[b'SAS_PATH'] = env[b'SAS_DIR']
        env[b'SAS_CCFPATH'] = path ('ccf')
        env[b'SAS_ODF'] = self._sumsas # but see _preexec
        env[b'SAS_CCF'] = os.path.join (self._odfdir, 'ccf.cif')

        prepend_environ_path (env, b'PATH', path ('bin'))
        prepend_environ_path (env, b'LD_LIBRARY_PATH', path ('libextra'))
        prepend_environ_path (env, b'LD_LIBRARY_PATH', path ('lib'))
        prepend_environ_path (env, b'PERL5LIB', path ('lib', 'perl5'))

        env[b'SAS_BROWSER'] = b'firefox' # yay hardcoding
        env[b'SAS_IMAGEVIEWER'] = b'ds9'
        env[b'SAS_SUPPRESS_WARNING'] = b'1'
        env[b'SAS_VERBOSITY'] = b'4'

        # These can be helpful:
        env[b'PWKIT_SAS_REVNUM'] = self._revnum
        env[b'PWKIT_SAS_OBSID'] = self._obsid

        return env


    def _preexec (self, env, printbuilds=True):
        from pwkit.cli import wrapout

        # Need to compile the CCF info?

        cif = env[b'SAS_CCF']
        if not os.path.exists (cif):
            if printbuilds:
                print ('[building %s]' % cif)

            log = os.path.join (self._odfdir, 'cifbuild.log')
            env['SAS_ODF'] = self._odfdir

            with open (log, 'wb') as f:
                w = wrapout.Wrapper (f)
                w.use_colors = True
                if w.launch ('cifbuild', ['cifbuild'], env=env, cwd=self._odfdir):
                    raise PKError ('failed to build CIF; see %s', log)

            if not os.path.exists (cif):
                # cifbuild can exit with status 0 whilst still having failed
                raise PKError ('failed to build CIF; see %s', log)

            env['SAS_ODF'] = self._sumsas

        # Need to generate SUM.SAS file?

        if not os.path.exists (self._sumsas):
            if printbuilds:
                print ('[building %s]' % self._sumsas)

            log = os.path.join (self._odfdir, 'odfingest.log')
            env['SAS_ODF'] = self._odfdir

            with open (log, 'wb') as f:
                w = wrapout.Wrapper (f)
                w.use_colors = True
                if w.launch ('odfingest', ['odfingest'], env=env, cwd=self._odfdir):
                    raise PKError ('failed to build CIF; see %s', log)

            env['SAS_ODF'] = self._sumsas


def commandline (argv):
    # TODO: convert to multitool if other subcommands are needed

    if len (argv) < 4 or argv[1] in ('-h', '--help'):
        print ('''usage: %s <manifestpath> exec <command> [args...]

Execute a program in the SAS environment. Due to the way SAS works, the path
to a MANIFEST.nnnnn file in an ODF directory must be specified, and all
operations work on the specified data set.''' % argv[0])

    manifest = argv[1]
    env = SasEnvironment (manifest)
    cmd = argv[2]

    if cmd != 'exec':
        cli.die ('usage: %s <manifestpath> exec <command> [args...]' % argv[0])

    progargv = argv[3:]
    env.execvpe (progargv)
