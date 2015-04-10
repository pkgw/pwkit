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
from ..cli import multitool
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


# Command-line interface

class Exec (multitool.Command):
    name = 'exec'
    argspec = '<manifest> <command> [args...]'
    summary = 'Run a program in SAS.'
    more_help = '''Due to the way SAS works, the path to a MANIFEST.nnnnn file in an ODF
directory must be specified, and all operations work on the specified data
set.'''

    def invoke (self, args, **kwargs):
        if len (args) < 2:
            raise multitool.UsageError ('exec requires at least 2 arguments')

        manifest = args[0]
        progargv = args[1:]

        env = SasEnvironment (manifest)
        env.execvpe (progargv)


class MakeOMAliases (multitool.Command):
    name = 'make-om-aliases'
    argspec = '<srcdir> <destdir>'
    summary = 'Generate user-friendly aliases to XMM-Newton OM data files.'
    more_help = 'destdir should already not exist and will be created.'

    PROD_TYPE = slice (0, 1) # 'P': final product; 'F': intermediate
    OBSID = slice (1, 11)
    EXPFLAG = slice (11, 12) # 'S': sched, 'U': unsched; 'X': N/A
    EXPNO = slice (14, 17) # (12-14 is the string 'OM')
    DTYPE = slice (17, 23)
    WINNUM = slice (23, 24)
    SRCNUM = slice (24, 27)
    EXTENSION = slice (28, None)

    extmap = {
        'ASC': 'txt',
        'FIT': 'fits',
        'PDF': 'pdf',
        'PS': 'ps',
    }

    dtypemap = {
        'image_': 'image_ccd',
        'simage': 'image_sky',
        'swsrli': 'source_list',
        'timesr': 'lightcurve',
        'tshplt': 'tracking_plot',
        'tstrts': 'tracking_stars',
    }

    def invoke (self, args, **kwargs):
        if len (args) != 2:
            raise multitool.UsageError ('make-om-aliases requires exactly 2 arguments')

        from fnmatch import fnmatch
        srcdir, destdir = args

        srcfiles = [x for x in os.listdir (srcdir)
                    if x[0] == 'P' and len (x) > 28]

        # Sorted list of exposure numbers.

        expnos = set ()

        for f in srcfiles:
            if not fnmatch (f, 'P*IMAGE_*.FIT'):
                continue
            expnos.add (f[self.EXPNO])

        expseqs = dict ((n, i) for i, n in enumerate (sorted (expnos)))

        # Do it.

        idents = set ()
        os.mkdir (destdir) # intentionally crash if exists; easiest approach

        for f in srcfiles:
            ptype = f[self.PROD_TYPE]
            obsid = f[self.OBSID]
            eflag = f[self.EXPFLAG]
            expno = f[self.EXPNO]
            dtype = f[self.DTYPE]
            winnum = f[self.WINNUM]
            srcnum = f[self.SRCNUM]
            ext = f[self.EXTENSION]

            seq = expseqs[expno]
            dtype = self.dtypemap[dtype.lower ()]
            ext = self.extmap[ext]

            # There's only one clash, and it's easy:
            if dtype == 'lightcurve' and ext == 'pdf':
                continue

            ident = (seq, dtype)
            if ident in idents:
                cli.die ('short identifier clash: %r', ident)
            idents.add (ident)

            oldpath = os.path.join (srcdir, f)
            newpath = os.path.join (destdir, '%s.%02d.%s' % (dtype, seq, ext))
            os.symlink (os.path.relpath (oldpath, destdir), newpath)


class Shell (multitool.Command):
    # XXX we hardcode bash! and we copy/paste from environments/__init__.py
    name = 'shell'
    argspec = '<manifest>'
    summary = 'Start an interactive shell in the SAS environment.'
    help_if_no_args = False
    more_help = '''Due to the way SAS works, the path to a MANIFEST.nnnnn file in an ODF
directory must be specified, and all operations work on the specified data
set.'''

    def invoke (self, args, **kwargs):
        if len (args) != 1:
            raise multitool.UsageError ('shell expects exactly 1 argument')

        env = SasEnvironment (args[0])

        from tempfile import NamedTemporaryFile
        with NamedTemporaryFile (delete=False) as f:
            print ('''[ -e ~/.bashrc ] && source ~/.bashrc
PS1="SAS(%s) $PS1"
rm %s''' % (env._obsid, f.name), file=f)

        env.execvpe (['bash', '--rcfile', f.name, '-i'])


class UpdateCcf (multitool.Command):
    name = 'update-ccf'
    argspec = ''
    summary = 'Update the SAS "current calibration files".'
    more_help = 'This executes an rsync command to make sure the files are up-to-date.'
    help_if_no_args = False

    def invoke (self, args, **kwargs):
        if len (args):
            raise multitool.UsageError ('update-ccf expects no arguments')

        sasdir = os.environ.get ('PWKIT_SAS')
        if sasdir is None:
            cli.die ('environment variable $PWKIT_SAS must be set')

        os.chdir (os.path.join (sasdir, 'ccf'))
        os.execvp ('rsync', ['rsync',
                             '-av',
                             '--delete',
                             '--delete-after',
                             '--force',
                             '--include=*.CCF',
                             '--exclude=*/',
                             'xmm.esac.esa.int::XMM_RED_CCF',
                             '.'])


class SasTool (multitool.Multitool):
    cli_name = 'pkenvtool sas'
    summary = 'Run tools in the SAS environment.'


def commandline (argv):
    tool = SasTool ()
    tool.populate (globals ().itervalues ())
    tool.commandline (argv)
