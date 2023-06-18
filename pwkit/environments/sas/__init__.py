# -*- mode: python; coding: utf-8 -*-
# Copyright 2015-2018 Peter Williams <peter@newton.cx> and collaborators.
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
should be run from something like /a/sas::

  $ ./install.sh

The CCF are like CALDB and need to be rsynced -- see the update-ccf
subcommand.


ODF data format notes
=========================

ODF files all have names in the format RRRR_NNNNNNNNNN_IIUEEECCMMM.ZZZ where:

RRRR
  revolution (orbit) number
NNNNNNNNNN
  obs ID
II
  The instrument:

  OM
    optical monitor
  R1
    RGS (reflection grating spectrometer) unit 1
  R2
    RGS 2
  M1
    EPIC (imaging camera) MOS 1 detector
  M2
    EPIC (imaging camera) MOS 2 detector
  PN
    EPIC (imaging camera) PN detector
  RM
    EPIC radiation monitor
  SC
    spacecraft
U
  Scheduling status of exposure:

  S
    scheduled
  U
    unscheduled
  X
    N/A
EEE
  exposure number
CC
  CCD/OM-window ID
MMM
  data type of file (many; not listed here)
ZZZ
  file extension

See the ``make-*-aliases`` commands for tools that generate symlinks with saner
names.

"""
from __future__ import absolute_import, division, print_function

__all__ = "".split()

import os.path

from ... import PKError, cli
from ...cli import multitool
from ...io import Path
from .. import Environment, prepend_environ_path


class SasEnvironment(Environment):
    _odfdir = None
    _revnum = None
    _obsid = None
    _sumsas = None
    _installdir = None
    _heaenv = None

    def __init__(self, manifest, installdir=None, heaenv=None):
        if installdir is None:
            installdir = self._default_installdir()
        if heaenv is None:
            from .. import heasoft

            heaenv = heasoft.HeasoftEnvironment()

        self._installdir = os.path.abspath(installdir)
        self._heaenv = heaenv

        # TODO: I used to read the manifest file to infer both the revolution
        # number and obsid, but in the case of 0673000145, the obsid mentioned
        # in the manifest is different! (But close: 0673000101.) So now I glob
        # the containing directory for that.

        manifest = Path(manifest)

        for line in manifest.read_lines():
            if not line.startswith("File "):
                continue

            bits = line.split()[1].split("_")
            if len(bits) < 3:
                continue

            self._revnum = bits[0]  # note: kept as a string; not an int
            break

        self._odfdir = Path(manifest).resolve().parent

        for p in self._odfdir.glob("%s_*_*.FIT" % self._revnum):
            bits = p.name.split("_")
            self._obsid = bits[1]
            break

        self._sumsas = self._odfdir / (
            "%s_%s_SCX00000SUM.SAS" % (self._revnum, self._obsid)
        )

    def _default_installdir(self):
        d = os.environ.get("PWKIT_SAS")
        if d is None:
            raise PKError(
                "SAS installation directory must be specified "
                "in the $PWKIT_SAS environment variable"
            )
        return d

    def modify_environment(self, env):
        self._heaenv.modify_environment(env)

        def path(*args):
            return os.path.join(self._installdir, *args)

        env["SAS_DIR"] = path()
        env["SAS_PATH"] = env["SAS_DIR"]
        env["SAS_CCFPATH"] = path("ccf")
        env["SAS_ODF"] = str(self._sumsas)  # but see _preexec
        env["SAS_CCF"] = str(self._odfdir / "ccf.cif")

        prepend_environ_path(env, "PATH", path("bin"))
        prepend_environ_path(env, "LD_LIBRARY_PATH", path("libextra"))
        prepend_environ_path(env, "LD_LIBRARY_PATH", path("lib"))
        prepend_environ_path(env, "PERL5LIB", path("lib", "perl5"))

        env["SAS_BROWSER"] = "firefox"  # yay hardcoding
        env["SAS_IMAGEVIEWER"] = "ds9"
        env["SAS_SUPPRESS_WARNING"] = "1"
        env["SAS_VERBOSITY"] = "4"

        # These can be helpful:
        env["PWKIT_SAS_REVNUM"] = self._revnum
        env["PWKIT_SAS_OBSID"] = self._obsid

        return env

    def _preexec(self, env, printbuilds=True):
        from ...cli import wrapout

        # Need to compile the CCF info?

        cif = env["SAS_CCF"]
        if not os.path.exists(cif):
            if printbuilds:
                print("[building %s]" % cif)

            env["SAS_ODF"] = str(self._odfdir)
            log = self._odfdir / "cifbuild.log"

            with log.open("wb") as f:
                w = wrapout.Wrapper(f)
                w.use_colors = True
                if w.launch("cifbuild", ["cifbuild"], env=env, cwd=str(self._odfdir)):
                    raise PKError("failed to build CIF; see %s", log)

            if not os.path.exists(cif):
                # cifbuild can exit with status 0 whilst still having failed
                raise PKError("failed to build CIF; see %s", log)

            env["SAS_ODF"] = str(self._sumsas)

        # Need to generate SUM.SAS file?

        if not self._sumsas.exists():
            if printbuilds:
                print("[building %s]" % self._sumsas)

            env["SAS_ODF"] = str(self._odfdir)
            log = self._odfdir / "odfingest.log"

            with log.open("wb") as f:
                w = wrapout.Wrapper(f)
                w.use_colors = True
                if w.launch("odfingest", ["odfingest"], env=env, cwd=str(self._odfdir)):
                    raise PKError("failed to build CIF; see %s", log)

            env["SAS_ODF"] = str(self._sumsas)


# Command-line interface


class Exec(multitool.Command):
    name = "exec"
    argspec = "<manifest> <command> [args...]"
    summary = "Run a program in SAS."
    more_help = """Due to the way SAS works, the path to a MANIFEST.nnnnn file in an ODF
directory must be specified, and all operations work on the specified data
set."""

    def invoke(self, args, **kwargs):
        if len(args) < 2:
            raise multitool.UsageError("exec requires at least 2 arguments")

        manifest = args[0]
        progargv = args[1:]

        env = SasEnvironment(manifest)
        env.execvpe(progargv)


class MakeEPICAliases(multitool.Command):
    name = "make-epic-aliases"
    argspec = "<srcdir> <destdir>"
    summary = "Generate user-friendly aliases to XMM-Newton EPIC data files."
    more_help = """destdir should already not exist and will be created. <srcdir> should
be the ODF directory, containing a file named MANIFEST.<numbers> and many others."""

    INSTRUMENT = slice(16, 18)
    EXPFLAG = slice(18, 19)  # 'S': sched, 'U': unsched; 'X': N/A
    EXPNO = slice(19, 22)
    CCDNO = slice(22, 24)
    DTYPE = slice(24, 27)
    EXTENSION = slice(28, None)

    instrmap = {
        "M1": "mos1",
        "M2": "mos2",
        "PN": "pn",
        "RM": "radmon",
    }

    extmap = {
        "FIT": "fits",
    }

    dtypemap = {
        "aux": "aux",
        "bue": "burst",
        "ccx": "counting_cycle",
        "cte": "compressed_timing",
        "dii": "diagnostic",
        "dli": "discarded_lines",
        "ecx": "hk_extraheating_config",  # or radiation mon count rate
        "esx": "spectra",  # radiation monitor spectra, that is
        "hbh": "hk_hbr_buffer",
        "hch": "hk_hbr_config",
        "hdi": "high_rate_offset_data",
        "hth": "hk_hbr_threshold",
        "ime": "imaging",
        "noi": "noise",
        "odi": "offset_data",
        "ove": "offset_variance",
        "pah": "hk_additional",
        "peh": "hk_periodic",
        "pmh": "hk_main",
        "pth": "hk_bright_pixels",
        "rie": "reduced_imaging",
        "tmh": "hk_thermal_limits",
        "tie": "timing",
    }

    def invoke(self, args, **kwargs):
        if len(args) != 2:
            raise multitool.UsageError("make-epic-aliases requires exactly 2 arguments")

        srcdir = Path(args[0])
        destdir = Path(args[1])

        srcpaths = [x for x in srcdir.iterdir() if len(x.name) > 28]

        # Sorted list of exposure numbers.

        expnos = dict((i, set()) for i in self.instrmap.keys())

        for p in srcpaths:
            instr = p.name[self.INSTRUMENT]
            if instr not in self.instrmap:
                continue

            expno = int(p.name[self.EXPNO])
            dtype = p.name[self.DTYPE]

            if expno > 0 and dtype not in ("DLI", "ODI"):
                expnos[instr].add(expno)

        expseqs = {}

        for k, v in expnos.items():
            expseqs[self.instrmap[k]] = dict((n, i) for i, n in enumerate(sorted(v)))

        # Do it.

        stems = set()
        destdir.mkdir()  # intentionally crash if exists; easiest approach

        for p in srcpaths:
            instr = p.name[self.INSTRUMENT]
            if instr not in self.instrmap:
                continue

            eflag = p.name[self.EXPFLAG]
            expno = p.name[self.EXPNO]
            ccdno = p.name[self.CCDNO]
            dtype = p.name[self.DTYPE]
            ext = p.name[self.EXTENSION]

            instr = self.instrmap[instr]
            expno = int(expno)
            dtype = self.dtypemap[dtype.lower()]
            ext = self.extmap[ext]

            if expno > 0 and dtype not in ("discarded_lines", "offset_data"):
                expno = expseqs[instr][expno]

            if instr == "radmon" and dtype == "hk_extraheating_config":
                dtype = "rates"

            if instr == "radmon" or dtype == "aux":
                stem = "%s_e%03d_%s.%s" % (instr, expno, dtype, ext)
            elif ccdno == "00":
                stem = "%s_%s.%s" % (instr, dtype, ext)
            elif dtype in ("discarded_lines", "offset_data"):
                stem = "%s_%s_e%03d_c%s.%s" % (instr, dtype, expno, ccdno, ext)
            else:
                stem = "%s_e%03d_c%s_%s.%s" % (instr, expno, ccdno, dtype, ext)

            if stem in stems:
                cli.die("short identifier clash: %r", stem)
            stems.add(stem)

            (destdir / stem).rellink_to(p)


class MakeOMAliases(multitool.Command):
    name = "make-om-aliases"
    argspec = "<srcdir> <destdir>"
    summary = "Generate user-friendly aliases to XMM-Newton OM data files."
    more_help = "destdir should already not exist and will be created."

    PROD_TYPE = slice(0, 1)  # 'P': final product; 'F': intermediate
    OBSID = slice(1, 11)
    EXPFLAG = slice(11, 12)  # 'S': sched, 'U': unsched; 'X': N/A
    EXPNO = slice(14, 17)  # (12-14 is the string 'OM')
    DTYPE = slice(17, 23)
    WINNUM = slice(23, 24)
    SRCNUM = slice(24, 27)
    EXTENSION = slice(28, None)

    extmap = {
        "ASC": "txt",
        "FIT": "fits",
        "PDF": "pdf",
        "PS": "ps",
    }

    dtypemap = {
        "image_": "image_ccd",
        "simage": "image_sky",
        "swsrli": "source_list",
        "timesr": "lightcurve",
        "tshplt": "tracking_plot",
        "tstrts": "tracking_stars",
    }

    def invoke(self, args, **kwargs):
        if len(args) != 2:
            raise multitool.UsageError("make-om-aliases requires exactly 2 arguments")

        from fnmatch import fnmatch

        srcdir, destdir = args

        srcfiles = [x for x in os.listdir(srcdir) if x[0] == "P" and len(x) > 28]

        # Sorted list of exposure numbers.

        expnos = set()

        for f in srcfiles:
            if not fnmatch(f, "P*IMAGE_*.FIT"):
                continue
            expnos.add(f[self.EXPNO])

        expseqs = dict((n, i) for i, n in enumerate(sorted(expnos)))

        # Do it.

        idents = set()
        os.mkdir(destdir)  # intentionally crash if exists; easiest approach

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
            dtype = self.dtypemap[dtype.lower()]
            ext = self.extmap[ext]

            # There's only one clash, and it's easy:
            if dtype == "lightcurve" and ext == "pdf":
                continue

            ident = (seq, dtype)
            if ident in idents:
                cli.die("short identifier clash: %r", ident)
            idents.add(ident)

            oldpath = os.path.join(srcdir, f)
            newpath = os.path.join(destdir, "%s.%02d.%s" % (dtype, seq, ext))
            os.symlink(os.path.relpath(oldpath, destdir), newpath)


class MakeRGSAliases(multitool.Command):
    name = "make-rgs-aliases"
    argspec = "<srcdir> <destdir>"
    summary = "Generate user-friendly aliases to XMM-Newton RGS data files."
    more_help = """destdir should already not exist and will be created. <srcdir> should
be the ODF directory, containing a file named MANIFEST.<numbers> and many others."""

    INSTRUMENT = slice(16, 18)
    EXPFLAG = slice(18, 19)  # 'S': sched, 'U': unsched; 'X': N/A
    EXPNO = slice(19, 22)
    CCDNO = slice(22, 24)
    DTYPE = slice(24, 27)
    EXTENSION = slice(28, None)

    instrmap = {
        "R1": "rgs1",
        "R2": "rgs2",
    }

    extmap = {
        "FIT": "fits",
    }

    dtypemap = {
        "aux": "aux",
        "d1h": "hk_dpp1",
        "d2h": "hk_dpp2",
        "dii": "diagnostic",
        "hte": "high_time_res",
        "ofx": "offset",
        "pch": "hk_ccd_temp",
        "pfh": "hk_periodic",
        "spe": "spectra",
    }

    def invoke(self, args, **kwargs):
        if len(args) != 2:
            raise multitool.UsageError("make-rgs-aliases requires exactly 2 arguments")

        srcdir = Path(args[0])
        destdir = Path(args[1])
        srcpaths = [x for x in srcdir.iterdir() if len(x.name) > 28]

        # Sorted list of exposure numbers.

        expnos = dict((i, set()) for i in self.instrmap.keys())

        for p in srcpaths:
            instr = p.name[self.INSTRUMENT]
            if instr not in self.instrmap:
                continue

            expno = int(p.name[self.EXPNO])
            if expno > 0 and expno < 900:
                expnos[instr].add(expno)

        expseqs = {}

        for k, v in expnos.items():
            expseqs[self.instrmap[k]] = dict((n, i) for i, n in enumerate(sorted(v)))

        # Do it.

        stems = set()
        destdir.mkdir()  # intentionally crash if exists; easiest approach

        for p in srcpaths:
            instr = p.name[self.INSTRUMENT]
            if instr not in self.instrmap:
                continue

            eflag = p.name[self.EXPFLAG]
            expno = p.name[self.EXPNO]
            ccdno = p.name[self.CCDNO]
            dtype = p.name[self.DTYPE]
            ext = p.name[self.EXTENSION]

            instr = self.instrmap[instr]
            expno = int(expno)
            dtype = self.dtypemap[dtype.lower()]
            ext = self.extmap[ext]

            if expno > 0 and expno < 900:
                expno = expseqs[instr][expno]

            if ccdno == "00" and dtype != "aux":
                stem = "%s_%s.%s" % (instr, dtype, ext)
            elif dtype == "aux":
                stem = "%s_e%03d_%s.%s" % (instr, expno, dtype, ext)
            elif dtype == "diagnostic":
                stem = "%s_%s_e%03d_c%s.%s" % (instr, dtype, expno, ccdno, ext)
            else:
                stem = "%s_e%03d_c%s_%s.%s" % (instr, expno, ccdno, dtype, ext)

            if stem in stems:
                cli.die("short identifier clash: %r", stem)
            stems.add(stem)

            (destdir / stem).rellink_to(p)


class MakeSCAliases(multitool.Command):
    name = "make-sc-aliases"
    argspec = "<srcdir> <destdir>"
    summary = "Generate user-friendly aliases to XMM-Newton spacecraft (SC) data files."
    more_help = """destdir should already not exist and will be created. <srcdir> should
be the ODF directory, containing a file named MANIFEST.<numbers> and many others."""

    INSTRUMENT = slice(16, 18)
    EXPFLAG = slice(18, 19)  # 'S': sched, 'U': unsched; 'X': N/A
    EXPNO = slice(19, 22)
    CCDNO = slice(22, 24)
    DTYPE = slice(24, 27)
    EXTENSION = slice(28, None)

    extmap = {
        "ASC": "txt",
        "FIT": "fits",
        "SAS": "txt",
    }

    dtypemap = {
        "ats": "attitude",
        "das": "dummy_attitude",
        "pos": "pred_orbit",
        "p1s": "phk_hk1",
        "p2s": "phk_hk2",
        "p3s": "phk_att1",
        "p4s": "phk_att2",
        "p5s": "phk_sid0",
        "p6s": "phk_sid1",
        "p7s": "phk_sid4",
        "p8s": "phk_sid5",
        "p9s": "phk_sid6",
        "ras": "raw_attitude",
        "ros": "recon_orbit",
        "sum": "summary",
        "tcs": "raw_time_corr",
        "tcx": "recon_time_corr",
    }

    def invoke(self, args, **kwargs):
        if len(args) != 2:
            raise multitool.UsageError("make-sc-aliases requires exactly 2 arguments")

        srcdir = Path(args[0])
        destdir = Path(args[1])

        srcfiles = [x for x in srcdir.iterdir() if len(x.name) > 28]

        # Do it.

        idents = set()
        destdir.mkdir()  # intentionally crash if exists; easiest approach

        for p in srcfiles:
            instr = p.name[self.INSTRUMENT]
            if instr != "SC":
                continue

            # none of these are actually useful for SC files:
            # eflag = p.name[self.EXPFLAG]
            # expno = p.name[self.EXPNO]
            # ccdno = p.name[self.CCDNO]
            dtype = p.name[self.DTYPE]
            ext = p.name[self.EXTENSION]

            # One conflict, easy to resolve
            if dtype == "SUM" and ext == "ASC":
                continue

            dtype = self.dtypemap[dtype.lower()]
            ext = self.extmap[ext]

            ident = dtype
            if ident in idents:
                cli.die("short identifier clash: %r", ident)
            idents.add(ident)

            (destdir / (dtype + "." + ext)).rellink_to(p)


class Shell(multitool.Command):
    # XXX we hardcode bash! and we copy/paste from environments/__init__.py
    name = "shell"
    argspec = "<manifest>"
    summary = "Start an interactive shell in the SAS environment."
    help_if_no_args = False
    more_help = """Due to the way SAS works, the path to a MANIFEST.nnnnn file in an ODF
directory must be specified, and all operations work on the specified data
set."""

    def invoke(self, args, **kwargs):
        if len(args) != 1:
            raise multitool.UsageError("shell expects exactly 1 argument")

        env = SasEnvironment(args[0])

        from tempfile import NamedTemporaryFile

        with NamedTemporaryFile(delete=False, mode="wt") as f:
            print(
                """[ -e ~/.bashrc ] && source ~/.bashrc
PS1="SAS(%s) $PS1"
rm %s"""
                % (env._obsid, f.name),
                file=f,
            )

        env.execvpe(["bash", "--rcfile", f.name, "-i"])


class UpdateCcf(multitool.Command):
    name = "update-ccf"
    argspec = ""
    summary = 'Update the SAS "current calibration files".'
    more_help = "This executes an rsync command to make sure the files are up-to-date."
    help_if_no_args = False

    def invoke(self, args, **kwargs):
        if len(args):
            raise multitool.UsageError("update-ccf expects no arguments")

        sasdir = os.environ.get("PWKIT_SAS")
        if sasdir is None:
            cli.die("environment variable $PWKIT_SAS must be set")

        os.chdir(os.path.join(sasdir, "ccf"))
        os.execvp(
            "rsync",
            [
                "rsync",
                "-av",
                "--delete",
                "--delete-after",
                "--force",
                "--include=*.CCF",
                "--exclude=*/",
                "xmm.esac.esa.int::XMM_VALID_CCF",
                ".",
            ],
        )


class SasTool(multitool.Multitool):
    cli_name = "pkenvtool sas"
    summary = "Run tools in the SAS environment."


def commandline(argv):
    tool = SasTool()
    tool.populate(globals().values())
    tool.commandline(argv)
