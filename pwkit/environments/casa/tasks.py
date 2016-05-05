# -*- mode: python; coding: utf-8 -*-
# Copyright 2015 Peter Williams <peter@newton.cx> and collaborators.
# Licensed under the MIT License.

"""pwkit.environments.casa.tasks - library of clones of CASA tasks

The way that the casapy code is written it's basically impossible to import
its tasks into a straight-Python environment (trust me, I've tried), so we're
more-or-less duplicating lots of CASA code. I try to provide saner semantics,
APIs, etc.

The goal is to make task-like functionality as a real Python library with no
side effects, so that we can actually script data processing. While we're at
it, we make them available on the command line.

"""

from __future__ import absolute_import, division, print_function, unicode_literals

import os.path, six, sys
from six.moves import range
import numpy as np

from . import util
from ... import binary_type, reraise_context, text_type, PKError
from ...cli import check_usage, wrong_usage, warn, die
from ...kwargv import ParseKeywords, Custom

# Keep the tasks alphabetized!

__all__ = str ('''
applycal applycal_cli ApplycalConfig
bpplot bpplot_cli BpplotConfig
clearcal clearcal_cli
concat concat_cli
delcal delcal_cli
delmod_cli
dftdynspec_cli
dftphotom_cli
dftspect_cli
extractbpflags extractbpflags_cli
flagmanager_cli
flaglist flaglist_cli FlaglistConfig
flagzeros flagzeros_cli FlagzerosConfig
fluxscale fluxscale_cli FluxscaleConfig
ft ft_cli FtConfig
gaincal gaincal_cli GaincalConfig
gencal gencal_cli GencalConfig
getopacities getopacities_cli
gpdetrend gpdetrend_cli GpdetrendConfig
gpdiagnostics_cli
gpplot gpplot_cli GpplotConfig
image2fits image2fits_cli
importevla importevla_cli
listobs listobs_cli
mfsclean mfsclean_cli MfscleanConfig
mjd2date_cli
mstransform mstransform_cli MstransformConfig
plotants plotants_cli
plotcal plotcal_cli PlotcalConfig
polmodel_cli
setjy setjy_cli SetjyConfig
split split_cli SplitConfig
spwglue_cli
uvsub uvsub_cli UvsubConfig
commandline
''').split ()


# Some utilities

from .util import sanitize_unicode as b

precal_doc = \
"""
*** Pre-applied calibrations:

gaintable=
  Comma-separated list of calibration tables to apply on-the-fly
  before solving

gainfield=
  SEMICOLON-separated list of field selections to apply for each gain table.
  If there are fewer items than there are gaintable items, the list is
  padded with blank items, implying no selection by field.

interp=
  COMMA-separated list of interpolation types to use for each gain
  table. If there are fewer items, the list is padded with 'linear'
  entries. Allowed values:
    nearest linear cubic spline

spwmap=
  SEMICOLON-separated list of spectral window mappings for each
  existing gain table; each record is a COMMA-separated list of
  integers. For the i'th spw in the dataset, spwmap[i] specifies
  the record in the gain table to use. For instance [0, 0, 1, 1]
  maps four spws in the UV data to just two spectral windows in
  the preexisting gain table.

opacity=
  Comma-separated list of opacities in nepers. One for each spw; if
  there are more spws than entries, the last entry is used for the
  remaining spws.

gaincurve=
  Whether to apply VLA-specific built in gain curve correction
  (default: false)

parang=
  Whether to apply parallactic angle rotation correction
  (default: false)
"""

stdsel_doc = \
"""
*** Standard data selection keywords:

antenna=
array=
correlation=
field=
intent=
observation=
scan=
spw=
taql=
timerange=
uvrange=
"""

loglevel_doc = \
"""
loglevel=
  Level of detail from CASA logging system. Default: warn; allowed:
    severe warn info info1 info2 info3 info4 info5 debug1 debug2 debugging
"""

def extractmsselect (cfg, havearray=False, havecorr=False, haveintent=True,
                     intenttoscanintent=False, taqltomsselect=True,
                     observationtoobs=False):
    # expects cfg to have:
    #  antenna [correlation] field intent observation scan spw taql timerange uvrange
    # fills a dict with:
    #  baseline [correlation] field intent (msselect|taql) observation scan spw time uvrange

    selkws = {}

    direct = 'field scan spw uvrange'.split ()
    indirect = 'antenna:baseline timerange:time'.split ()

    if havearray:
        indirect.append ('array:subarray')

    if havecorr:
        direct.append ('correlation')

    if haveintent:
        if intenttoscanintent:
            indirect.append ('intent:scanintent')
        else:
            direct.append ('intent')

    if observationtoobs:
        indirect.append ('observation:obs')
    else:
        direct.append ('observation')

    if taqltomsselect:
        indirect.append ('taql:msselect')
    else:
        direct.append ('taql')

    for k in direct:
        selkws[k] = getattr (cfg, k) or ''

    for p in indirect:
        ck, sk = p.split (':')
        selkws[sk] = getattr (cfg, ck) or ''

    return selkws


def applyonthefly (cb, cfg):
    # expects cfg to have:
    #   gaintable gainfield interp spwmap opacity gaincurve parang

    n = len (cfg.gaintable)

    # fill in missing values, taking care not to mutate cfg.

    gainfields = list (cfg.gainfield)
    interps = list (cfg.interp)
    spwmaps = list (cfg.spwmap)

    if len (gainfields) < n:
        gainfields += [''] * (n - len (gainfields))
    elif len (gainfields) > n:
        raise ValueError ('more "gainfield" entries than "gaintable" entries')

    if len (interps) < n:
        interps += ['linear'] * (n - len (interps))
    elif len (interps) > n:
        raise ValueError ('more "interp" entries than "gaintable" entries')

    if len (spwmaps) < n:
        spwmaps += [[-1]] * (n - len (spwmaps))
    elif len (spwmaps) > n:
        raise ValueError ('more "spwmap" entries than "gaintable" entries')

    for table, field, interp, spwmap in zip (cfg.gaintable, gainfields,
                                             interps, spwmaps):
        cb.setapply (table=b(table), field=b(field), interp=b(interp), spwmap=b(spwmap),
                     t=0., calwt=True)

    if len (cfg.opacity):
        cb.setapply (type=b'TOPAC', opacity=b(cfg.opacity), t=-1, calwt=True)

    if cfg.gaincurve:
        cb.setapply (type=b'GAINCURVE', t=-1, calwt=True)

    if cfg.parang:
        cb.setapply (type=b'P')


def makekwcli (doc, cfgclass, impl):
    def kwclifunc (argv):
        check_usage (doc, argv, usageifnoargs=True)
        cfg = cfgclass ().parse (argv[1:])
        util.logger (cfg.loglevel)
        impl (cfg)
    return kwclifunc


# applycal

applycal_doc = \
"""
casatask applycal vis=<MS> [keywords]

Fill in the CORRECTED_DATA column of a visibility dataset using
the raw data and a set of calibration tables.

vis=
  The MS to modify

calwt=
  Write out calibrated weights as well as calibrated visibilities.
  Default: false
""" + precal_doc + stdsel_doc + loglevel_doc


class ApplycalConfig (ParseKeywords):
    vis = Custom (str, required=True)
    calwt = False
    # skipping: flagbackup

    gaintable = [str]
    gainfield = Custom ([str], sep=';')
    interp = [str]
    @Custom ([str], sep=';')
    def spwmap (v):
        return [map (int, e.split (',')) for e in v]
    opacity = [float]
    gaincurve = False
    parang = False

    antenna = str
    field = str
    intent = str
    observation = str
    scan = str
    spw = str
    taql = str
    timerange = str
    uvrange = str

    applymode = 'calflag' # almost never want to change this

    loglevel = 'warn'


def applycal (cfg):
    cb = util.tools.calibrater ()
    cb.open (filename=b(cfg.vis), compress=False, addcorr=True, addmodel=False)

    selkws = extractmsselect (cfg)
    selkws['chanmode'] = 'none' # ?
    cb.selectvis (**b(selkws))

    applyonthefly (cb, cfg)

    cb.correct (b(cfg.applymode))
    cb.close ()


applycal_cli = makekwcli (applycal_doc, ApplycalConfig, applycal)


# bpplot
#
# plotcal() can do this, but it is slow, ugly, and has not-very-informative
# output. Yes, CASA is so crappy that I can even speed up bandpass plotting
# by an order of magnitude.

bpplot_doc = \
"""
casatask bpplot caltable= dest=

Plot a bandpass calibration table. Currently, the supported format is a series
of pages showing amplitude and phase against normalized channel number, with
each page showing a particular antenna and polarization. Polarizations are
always reported as "R" and "L" since the relevant information is not stored
within the bandpass data set.

caltable=MS
  The input calibration Measurement Set

dest=PATH
  If specified, plots are saved to this file -- the format is inferred
  from the extension, which must allow multiple pages to be saved. If
  unspecified, the plots are displayed using a Gtk3 backend.

dims=WIDTH,HEIGHT
  If saving to a file, the dimensions of a each page. These are in points
  for vector formats (PDF, PS) and pixels for bitmaps (PNG). Defaults to
  1000, 600.

margins=TOP,RIGHT,LEFT,BOTTOM
  If saving to a file, the plot margins in the same units as the dims.
  The default is 4 on every side.
""" + loglevel_doc


class BpplotConfig (ParseKeywords):
    caltable = Custom (str, required=True)
    dest = str
    dims = [1000, 600]
    margins = [4, 4, 4, 4]
    loglevel = 'warn'


def bpplot (cfg):
    import omega as om, omega.render
    from ... import numutil

    if isinstance (cfg.dest, omega.render.Pager):
        # This is for non-CLI invocation.
        pager = cfg.dest
    elif cfg.dest is None:
        import omega.gtk3
        pager = om.makeDisplayPager ()
    else:
        pager = om.makePager (cfg.dest,
                              dims=cfg.dims,
                              margins=cfg.margins,
                              style=om.styles.ColorOnWhiteVector ())

    tb = util.tools.table ()

    tb.open (binary_type (cfg.caltable), nomodify=True)
    spws = tb.getcol (b'SPECTRAL_WINDOW_ID')
    ants = tb.getcol (b'ANTENNA1')
    vals = tb.getcol (b'CPARAM')
    flags = tb.getcol (b'FLAG')
    tb.close ()

    tb.open (binary_type (os.path.join (cfg.caltable, 'ANTENNA')), nomodify=True)
    names = tb.getcol (b'NAME')
    tb.close ()

    npol, nchan, nsoln = vals.shape

    # see what we've got

    antpols = {}
    seenspws = set ()

    for ipol in range (npol):
        for isoln in range (nsoln):
            if not flags[ipol,:,isoln].all ():
                k = (ants[isoln], ipol)
                byspw = antpols.get (k)
                if byspw is None:
                    antpols[k] = byspw = []

                byspw.append ((spws[isoln], isoln))
                seenspws.add (spws[isoln])

    seenspws = sorted (seenspws)
    spw_to_offset = dict ((spwid, spwofs * nchan)
                          for spwofs, spwid in enumerate (seenspws))

    # normalize phases to avoid distracting wraps

    for iant, ipol in sorted (six.iterkeys (antpols)):
        for ispw, isoln in antpols[iant,ipol]:
            f = flags[ipol,:,isoln]
            meanph = np.angle (vals[ipol,~f,isoln].mean ())
            vals[ipol,:,isoln] *= np.exp (-1j * meanph)

    # find plot limits

    okvals = vals[np.where (~flags)]

    max_am = np.abs (okvals).max ()
    min_am = np.abs (okvals).min ()
    span = max_am - min_am
    max_am += 0.05 * span
    min_am -= 0.05 * span

    max_ph = np.angle (okvals, deg=True).max ()
    min_ph = np.angle (okvals, deg=True).min ()
    span = max_ph - min_ph
    max_ph += 0.05 * span
    min_ph -= 0.05 * span
    if max_ph > 160:
        max_ph = 180
    if min_ph < -160:
        min_ph = -180

    polnames = 'RL' # XXX: identification doesn't seem to be stored in cal table

    # plot away

    for iant, ipol in sorted (six.iterkeys (antpols)):
        p_am = om.RectPlot ()
        p_ph = om.RectPlot ()

        for ispw, isoln in antpols[iant,ipol]:
            f = flags[ipol,:,isoln]
            a = np.abs (vals[ipol,:,isoln])
            p = np.angle (vals[ipol,:,isoln], deg=True)
            w = np.where (~f)[0]

            for s in numutil.slice_around_gaps (w, 1):
                wsub = w[s]
                if wsub.size == 0:
                    continue # Should never happen, but eh.
                else:
                    # It'd also be pretty weird to have a spectral window
                    # containing just one (valid) channel, but it could
                    # happen.
                    lines = (wsub.size > 1)

                p_am.addXY (wsub + spw_to_offset[ispw], a[wsub], None,
                            lines=lines, dsn=ispw)
                p_ph.addXY (wsub + spw_to_offset[ispw], p[wsub], None,
                            lines=lines, dsn=ispw)

        p_am.setBounds (xmin=0,
                        xmax=len (seenspws) * nchan,
                        ymin=min_am,
                        ymax=max_am)
        p_ph.setBounds (xmin=0,
                        xmax=len (seenspws) * nchan,
                        ymin=min_ph,
                        ymax=max_ph)
        p_am.addKeyItem ('%s %s' % (names[iant], polnames[ipol]))

        p_am.bpainter.paintLabels = False
        p_am.setYLabel ('Amplitude')
        p_ph.setLabels ('Normalized channel', 'De-meaned Phase (deg)')

        vb = om.layout.VBox (2)
        vb[0] = p_am
        vb[1] = p_ph
        vb.setWeight (0, 2.5)
        pager.send (vb)

bpplot_cli = makekwcli (bpplot_doc, BpplotConfig, bpplot)


# clearcal

clearcal_doc = \
"""
casatask clearcal [-w] <vis1> [more vises...]

Fill the imaging and calibration columns (MODEL_DATA, CORRECTED_DATA,
IMAGING_WEIGHT) of each measurement set with default values, creating
the columns if necessary.

If you want to reset calibration models, these days you probably want
delmod. If you want to quickly make the columns go away, you probably
want delcal.

-w - only create and fill the IMAGING_WEIGHT column
"""

clearcal_imaging_col_tmpl = {'_c_order': True,
                             'comment': '',
                             'dataManagerGroup': '',
                             'dataManagerType': '',
                             'keywords': {},
                             'maxlen': 0,
                             'ndim': 1,
                             'option': 0,
                             'shape': None,
                             'valueType': 'float'}
clearcal_imaging_dminfo_tmpl = {'TYPE': 'TiledShapeStMan',
                                'SPEC': {'DEFAULTTILESHAPE': [32, 128]},
                                'NAME': 'imagingweight'}

def clearcal (vis, weightonly=False):
    tb = util.tools.table ()
    cb = util.tools.calibrater ()

    # cb.open() will create the tables if they're not present, so
    # if that's the case, we don't actually need to run initcalset()

    tb.open (b(vis), nomodify=False)
    colnames = tb.colnames ()
    needinit = ('MODEL_DATA' in colnames) or ('CORRECTED_DATA' in colnames)
    if 'IMAGING_WEIGHT' not in colnames:
        c = dict (clearcal_imaging_col_tmpl)
        c['shape'] = tb.getcell (b'DATA', 0).shape[-1:]
        tb.addcols ({b'IMAGING_WEIGHT': c}, clearcal_imaging_dminfo_tmpl)
    tb.close ()

    if not weightonly:
        cb.open (b(vis))
        if needinit:
            cb.initcalset ()
        cb.close ()


def clearcal_cli (argv):
    check_usage (clearcal_doc, argv, usageifnoargs=True)

    argv = list (argv)
    weightonly = '-w' in argv
    if weightonly:
        sys.argv.remove ('-w')

    if len (argv) < 2:
        wrong_usage (clearcal_doc, 'need at least one argument')

    util.logger ()
    for vis in argv[1:]:
        clearcal (b(vis), weightonly=weightonly)


# concat

concat_doc = \
"""
casatask concat [-s] <input vises ...> <output vis>

Concatenate the visibility measurement sets.

-s - sort the output in time
"""

concat_freqtol = 1e-5
concat_dirtol = 1e-5

def concat (invises, outvis, timesort=False):
    tb = util.tools.table ()
    ms = util.tools.ms ()

    if os.path.exists (outvis):
        raise RuntimeError ('output "%s" already exists' % outvis)

    for invis in invises:
        if not os.path.isdir (invis):
            raise RuntimeError ('input "%s" does not exist' % invis)

    tb.open (b(invises[0]))
    tb.copy (b(outvis), deep=True, valuecopy=True)
    tb.close ()

    ms.open (b(outvis), nomodify=False)

    for invis in invises[1:]:
        ms.concatenate (msfile=b(invis), freqtol=b(concat_freqtol),
                        dirtol=b(concat_dirtol))

    ms.writehistory (message=b'taskname=tasklib.concat', origin=b'tasklib.concat')
    ms.writehistory (message=b('vis = ' + ', '.join (invises)), origin=b'tasklib.concat')
    ms.writehistory (message=b('timesort = ' + 'FT'[int (timesort)]), origin=b'tasklib.concat')

    if timesort:
        ms.timesort ()

    ms.close ()


def concat_cli (argv):
    check_usage (concat_doc, argv, usageifnoargs=True)

    argv = list (argv)
    timesort = '-s' in argv
    if timesort:
        sys.argv.remove ('-s')

    if len (argv) < 3:
        wrong_usage (concat_doc, 'need at least two arguments')

    util.logger ()
    concat (argv[1:-1], argv[-1], timesort)


# delcal
#
# Not a CASA task. Delmod on steroids, at least when delmod
# is operating on scratch columns and not OTF models.

delcal_doc = \
"""
casatask delcal <ms> [mses...]

Delete the MODEL_DATA and CORRECTED_DATA columns from MSes.
"""


def delcal (mspath):
    wantremove = 'MODEL_DATA CORRECTED_DATA'.split ()
    tb = util.tools.table ()
    tb.open (b(mspath), nomodify=False)
    cols = frozenset (tb.colnames ())
    toremove = [b(c) for c in wantremove if c in cols]
    if len (toremove):
        tb.removecols (toremove)
    tb.close ()
    return toremove


def delcal_cli (argv):
    check_usage (delcal_doc, argv, usageifnoargs=True)
    util.logger ()

    for mspath in argv[1:]:
        removed = delcal (mspath)
        if len (removed):
            print ('%s: removed %s' % (mspath, ', '.join (removed)))
        else:
            print ('%s: nothing to remove' % mspath)


# delmod

delmod_doc = \
"""
casatask delmod <MS...>

Delete the "on-the-fly" model information from the specified
Measurement Set(s).

If you want to delete the scratch columns, use delcal. If you
want to clear the scratch columns, use clearcal. I'm torn
between wanting better terminology and not wanting to be
gratuitously different from CASA.
"""

def delmod_cli (argv, alter_logger=True):
    check_usage (delmod_doc, argv, usageifnoargs=True)
    if alter_logger:
        util.logger ()

    cb = util.tools.calibrater ()

    for mspath in argv[1:]:
        cb.open (b(mspath), addcorr=False, addmodel=False)
        cb.delmod (otf=True, scr=False)
        cb.close ()


# dftdynspec
#
# Shim for a separate module

def dftdynspec_cli (argv):
    from .dftdynspec import dftdynspec_cli
    dftdynspec_cli (argv)


# dftphotom
#
# Shim for a separate module

def dftphotom_cli (argv):
    from .dftphotom import dftphotom_cli
    dftphotom_cli (argv)


# dftspect
#
# Shim for a separate module

def dftspect_cli (argv):
    from .dftspect import dftspect_cli
    dftspect_cli (argv)


# extractbpflags
#
# Not a CASA task, but I've found this to be very useful.

extractbpflags_doc = \
"""
When CASA encounters flagged channels in bandpass calibration tables, it
interpolates over them as best it can -- even if interp='<any>,nearest'. This
means that if certain channels are unflagged in some target data but entirely
flagged in your BP cal, they'll get multiplied by some number that may or may
not be reasonable, not flagged. This is scary if, for instance, you're using
an automated system to find RFI, or you flag edge channels in some uneven way.

This script writes out a list of flagging commands corresponding to the
flagged channels in the bandpass table to ensure that the data without
bandpass solutions are flagged.

Note that, because we can't select by antpol, we can't express a situation in
which the R and L bandpass solutions have different flags. But in CASA the
flags seem to always be the same.

We're assuming that the channelization of the bandpass solution and the data
are the same.
"""

def extractbpflags (calpath, deststream):
    tb = util.tools.table ()
    tb.open (b(os.path.join (calpath, 'ANTENNA')))
    antnames = tb.getcol (b'NAME')
    tb.close ()

    tb.open (b(calpath))
    try:
        t = tb.getkeyword (b'VisCal')
    except RuntimeError:
        raise PKError ('no "VisCal" keyword in %s; it doesn\'t seem to be a '
                       'bandpass calibration table', calpath)

    if t != 'B Jones':
        raise PKError ('table %s doesn\'t seem to be a bandpass calibration '
                       'table; its type is "%s"', calpath, t)

    def emit (antidx, spwidx, chanstart, chanend):
        # Channel ranges are inclusive, unlike Python syntax.
        print ("antenna='%s&*' spw='%d:%d~%d' reason='BANDPASS_FLAGGED'" % \
               (antnames[antidx], spwidx, chanstart, chanend), file=deststream)

    for row in range (tb.nrows ()):
        ant = tb.getcell (b'ANTENNA1', row)
        spw = tb.getcell (b'SPECTRAL_WINDOW_ID', row)
        flag = tb.getcell (b'FLAG', row)

        # This is the logical 'or' of the two polarizations: i.e., anything that
        # is flagged in either poln is flagged in this.
        sqflag = ~((~flag).prod (axis=0, dtype=np.bool))

        runstart = None

        for i in range (sqflag.size):
            if sqflag[i]:
                # This channel is flagged. Start a run if not already in one.
                if runstart is None:
                    runstart = i
            elif runstart is not None:
                # The current run just ended.
                emit (ant, spw, runstart, i - 1)
                runstart = None

        if runstart is not None:
            emit (ant, spw, runstart, i)

    tb.close ()


def extractbpflags_cli (argv):
    check_usage (extractbpflags_doc, argv, usageifnoargs='long')

    if len (argv) != 2:
        wrong_usage (extractbpflags_doc, 'expect one MS name as an argument')

    extractbpflags (argv[1], sys.stdout)


# flagmanager. Not really complicated enough to make it worth making a
# modular implementation to be driven from the CLI.

flagmanager_doc = \
"""
casatask flagmanager list <ms>
casatask flagmanager save <ms> <name>
casatask flagmanager restore <ms> <name>
casatask flagmanager delete <ms> <name>
"""

def flagmanager_cli (argv, alter_logger=True):
    check_usage (flagmanager_doc, argv, usageifnoargs=True)

    if len (argv) < 3:
        wrong_usage (flagmanager_doc, 'expect at least a mode and an MS name')

    mode = argv[1]
    ms = argv[2]

    if alter_logger:
        if mode == 'list':
            util.logger ('info')
        elif mode == 'delete':
            # it WARNs 'deleting version xxx' ... yargh
            util.logger ('severe')
        else:
            util.logger ()

    try:
        factory = util.tools.agentflagger
    except AttributeError:
        factory = util.tools.testflagger

    af = factory ()
    af.open (b(ms))

    if mode == 'list':
        if len (argv) != 3:
            wrong_usage (flagmanager_doc, 'expect exactly one argument in list mode')
        af.getflagversionlist ()
    elif mode == 'save':
        if len (argv) != 4:
            wrong_usage (flagmanager_doc, 'expect exactly two arguments in save mode')
        from time import strftime
        name = argv[3]
        af.saveflagversion (versionname=b(name), merge=b'replace',
                            comment=b('created %s (casatask flagmanager)'
                                      % strftime ('%Y-%m-%dT%H:%M:%SZ')))
    elif mode == 'restore':
        if len (argv) != 4:
            wrong_usage (flagmanager_doc, 'expect exactly two arguments in restore mode')
        name = argv[3]
        af.restoreflagversion (versionname=b(name), merge=b'replace')
    elif mode == 'delete':
        if len (argv) != 4:
            wrong_usage (flagmanager_doc, 'expect exactly two arguments in delete mode')
        name = argv[3]

        if not os.path.isdir (os.path.join (ms + '.flagversions', 'flags.' + name)):
            # This condition only results in a WARN from deleteflagversion ()!
            raise RuntimeError ('version "%s" doesn\'t exist in "%s.flagversions"'
                                % (name, ms))

        af.deleteflagversion (versionname=b(name))
    else:
        wrong_usage (flagmanager_doc, 'unknown flagmanager mode "%s"' % mode)

    af.done ()


# flaglist. Not quite a CASA task; something like
# flagcmd (vis=, inpmode='list', inpfile=, flagbackup=False)
#
# We have to reproduce a lot of the dumb logic from the flaghelper.py module
# because we can't import it because it drags in the whole casapy pile of
# stuff.

flaglist_doc = \
"""
casatask flaglist vis= inpfile= [datacol=]

Flag data using a list of flagging commands stored in a text file. This
is approximately equivalent to 'flagcmd(vis=, inpfile=, inpmode='list',
flagbackup=False)'.

This implementation must emulate the CASA modules that load up the
flagging commands and may not be precisely compatible with the CASA
version.
"""

class FlaglistConfig (ParseKeywords):
    vis = Custom (str, required=True)
    inpfile = Custom (str, required=True)
    datacol = 'data'
    loglevel = 'warn'


def flaglist (cfg):
    from ast import literal_eval

    try:
        factory = util.tools.agentflagger
    except AttributeError:
        factory = util.tools.testflagger

    af = factory ()
    af.open (b(cfg.vis), 0.0)
    af.selectdata ()

    for row, origline in enumerate (open (cfg.inpfile)):
        origline = origline.rstrip ()
        if not len (origline):
            continue
        if origline[0] == '#':
            continue

        # emulating flaghelper.py here and elsewhere ...
        bits = origline.replace ('true', 'True').replace ('false', 'False').split (' ')
        params = {}
        lastkey = None

        for bit in bits:
            subbits = bit.split ('=', 1)

            if len (subbits) == 1:
                assert lastkey is not None, 'illegal flag list syntax'
                params[lastkey] += ' ' + bit
            else:
                params[subbits[0]] = subbits[1]
                lastkey = subbits[0]

        assert 'ntime' not in params, 'cannot handle "ntime" flag key'

        for key in params.keys ():
            val = params[key]

            try:
                val = literal_eval (val)
            except ValueError:
                val = val.strip ('\'"')

            params[key] = val

        params['name'] = 'agent_%d' % row
        params['datacolumn'] = cfg.datacol.upper ()
        params['apply'] = True

        params.setdefault ('mode', 'manual')

        if not af.parseagentparameters (b(params)):
            raise Exception ('cannot parse flag line: %s' % origline)

    af.init ()
    # A summary report would be nice. run() should return
    # info but I can't get it to do so. (I'm just trying to
    # copy the task_flagdata.py implementation.)
    af.run (True, True)
    af.done ()


flaglist_cli = makekwcli (flaglist_doc, FlaglistConfig, flaglist)


# flagzeros. Not quite a CASA task; something like
# flagdata (vis=, mode='clip', clipzeros=True, flagbackup=False)

flagzeros_doc = \
"""
casatask flagzeros vis= [datacol=]

Flag zero data points in the specified data column.
"""

class FlagzerosConfig (ParseKeywords):
    vis = Custom (str, required=True)
    datacol = 'data'
    loglevel = 'warn'


def flagzeros (cfg):
    try:
        factory = util.tools.agentflagger
    except AttributeError:
        factory = util.tools.testflagger

    af = factory ()
    af.open (b(cfg.vis), 0.0)
    af.selectdata ()
    pars = dict (datacolumn=cfg.datacol.upper (),
                 clipzeros=True, name='CLIP', mode='clip',
                 apply=True)
    af.parseagentparameters (pars)
    af.init ()
    # A summary report would be nice. run() should return
    # info but I can't get it to do so. (I'm just trying to
    # copy the task_flagdata.py implementation.)
    af.run (True, True)
    af.done ()


flagzeros_cli = makekwcli (flagzeros_doc, FlagzerosConfig, flagzeros)


# fluxscale

fluxscale_doc = \
"""
casatask fluxscale vis=<MS> caltable=<MS> fluxtable=<new MS> reference=<fields> transfer=<fields> [keywords]

Write a new calibation table determining the fluxes for intermediate calibrators
from known reference sources

vis=
  The visibility dataset. (Shouldn't be needed, but ...)

caltable=
  The preexisting calibration table with gains associated with more than one source.

fluxtable=
  The path of a new calibration table to create

reference=
  Comma-separated names of sources whose model fluxes are assumed to be well-known.

transfer=
  Comma-separated names of sources whose fluxes should be computed from the gains.

listfile=
  If specified, write out flux information to this path.

append=
  Boolean, default false. If true, append to existing cal table rather than
  overwriting.

refspwmap=
  Comma-separated list of integers. If gains are only available for some spws,
  map from the data to the gains. For instance, refspwmap=1,1,3,3 means that spw 0
  will have its flux calculated using the gains for spw 1.
""" + loglevel_doc

# Not supported in CASA 3.4:
#incremental=
#  Boolean, default false. If true, create an "incremental" table where the amplitudes
#  are correction factors, not absolute gains. (I.e., for the reference sources,
#  the amplitudes will be unity.)


class FluxscaleConfig (ParseKeywords):
    vis = Custom (str, required=True)
    caltable = Custom (str, required=True)
    fluxtable = Custom (str, required=True)
    reference = Custom ([str], required=True)
    transfer = Custom ([str], required=True)

    listfile = str
    append = False
    refspwmap = [int]
    #incremental = False

    loglevel = 'warn'


def fluxscale (cfg):
    cb = util.tools.calibrater ()

    reference = cfg.reference
    if isinstance (reference, (list, tuple)):
        reference = ','.join (reference)

    transfer = cfg.transfer
    if isinstance (transfer, (list, tuple)):
        transfer = ','.join (transfer)

    refspwmap = cfg.refspwmap
    if not len (refspwmap):
        refspwmap = [-1]

    cb.open (b(cfg.vis), compress=False, addcorr=False, addmodel=False)
    result = cb.fluxscale (tablein=b(cfg.caltable), tableout=b(cfg.fluxtable),
                           reference=b(reference), transfer=b(transfer),
                           listfile=b(cfg.listfile or ''),
                           append=cfg.append, refspwmap=b(refspwmap))
                           #incremental=cfg.incremental)
    cb.close ()
    return result


fluxscale_cli = makekwcli (fluxscale_doc, FluxscaleConfig, fluxscale)


# ft
#
# We derive 'nterms' from len(model), and always derive reffreq
# from the model images. These seem like safe constraints?

ft_doc = \
"""
casatask ft vis=<MS> [keywords]

Fill in (or update) the MODEL_DATA column of a Measurement Set with
visibilities computed from an image or list of components.

vis=
  The path to the measurement set

model=
  Comma-separated list of model images, each giving successive
  Taylor terms of a spectral model for each source. (It's fine
  to have just one model, and this will do what you want.) The
  reference frequency for the Taylor expansion is taken from
  the first image.

complist=
  Path to a CASA ComponentList Measurement Set to use in the modeling.
  I don't know what happens if you specify both this and "model".
  They might both get applied?

incremental=
  Bool, default false, meaning that the MODEL_DATA column will be
  replaced with the new values computed here. If true, the new values
  are added to whatever's already in MODEL_DATA.

wprojplanes=
  Optional integer. If provided, W-projection will be used in the computation
  of the model visibilities, using the specified number of planes. Note that
  this *does* make a difference even now that this task only embeds information
  in a MS to enable later on-the-fly computation of the UV model.

usescratch=
  Foo.
""" + stdsel_doc + loglevel_doc


class FtConfig (ParseKeywords):
    vis = Custom (str, required=True)
    model = [str]
    complist = str
    incremental = False
    wprojplanes = int
    usescratch = False

    antenna = str
    field = str
    observation = str
    scan = str
    spw = str
    taql = str
    timerange = str
    uvrange = str

    loglevel = 'warn'


def ft (cfg):
    im = util.tools.imager ()

    im.open (b(cfg.vis), usescratch=cfg.usescratch)
    im.selectvis (**b(extractmsselect (cfg, haveintent=False,
                                       taqltomsselect=False)))
    nmodel = len (cfg.model)

    if nmodel > 1:
        ia = util.tools.image ()
        ia.open (b(cfg.model[0]))
        # This gives Hz:
        reffreq = ia.coordsys ().referencevalue (type=b'spectral')['numeric'][0]
        ia.close ()
        im.settaylorterms (ntaylorterms=nmodel, reffreq=reffreq)

    if cfg.wprojplanes is not None:
        im.defineimage ()
        im.setoptions (ftmachine=b'wproject', wprojplanes=cfg.wprojplanes)

    im.ft (model=b(cfg.model),
           complist=b(cfg.complist or ''),
           incremental=cfg.incremental)
    im.close ()


ft_cli = makekwcli (ft_doc, FtConfig, ft)


# gaincal
#
# I've folded in the bandpass functionality since there's
# so much overlap. Some limitations that this leads to:
#
# - bandpass solint has a frequency component that we don't support
# - bandpass combine defaults to 'scan'

gaincal_doc = \
"""
casatask gaincal vis=<MS> caltable=<TBL> [keywords]

Compute calibration parameters from data. Encompasses the functionality
of CASA tasks 'gaincal' *and* 'bandpass'.

vis=
  Input dataset

caltable=
  Output calibration table (can exist if append=True)

gaintype=
  Kind of gain solution:
    G       - gains per poln and spw (default)
    T       - like G, but one value for all polns
    GSPLINE - like G, with a spline fit. "Experimental"
    B       - bandpass
    BPOLY   - bandpass with polynomial fit. "Somewhat experimental"
    K       - antenna-based delays
    KCROSS  - global cross-hand delay ; use parang=True
    D       - solve for instrumental leakage
    Df      - above with per-channel leakage terms
    D+QU    - solve for leakage and apparent source polarization
    Df+QU   - above with per-channel leakage terms
    X       - solve for absolute position angle phase term
    Xf      - above with per-channel phase terms
    D+X     - D and X. "Not normally done"
    Df+X    - Df and X. Presumably also not normally done.
    XY+QU   - ?
    XYf+QU  - ?

calmode=
  What parameters to solve for: amplitude ("a"), phase ("p"), or both
  ("ap"). Default is "ap". Not used in bandpass solutions.

solint=
  Solution interval; this is an upper bound, but solutions
  will be broken across certain boundaries according to "combine".
  'inf'    - solutions as long as possible (the default)
  'int'    - one solution per integration
  (str)    - a specific time with units (e.g. '5min')
  (number) - a specific time in seconds

combine=
  Comma-separated list of boundary types; solutions will NOT be
  broken across these boundaries. Types are:
    field scan spw

refant=
  Comma-separated list of reference antennas in decreasing
  priority order.

solnorm=
  Normalize solution amplitudes to 1 after solving (only applies
  to gaintypes G, T, B). Also normalizes bandpass phases to zero
  when solving for bandpasses. Default: false.

append=
  Whether to append solutions to an existing table. If the table
  exists and append=False, the table is overwritten! (Default: false)
""" + precal_doc + """
*** Low-level parameters:

minblperant=
  Number of baselines for each ant in order to solve (default: 4)

minsnr=
  Min. SNR for acceptable solutions (default: 3.0)

preavg=
  Interval for pre-averaging data within each solution interval,
  in seconds. Default is -1, meaning not to pre-average.
""" + stdsel_doc + loglevel_doc


class GaincalConfig (ParseKeywords):
    vis = Custom (str, required=True)
    caltable = Custom (str, required=True)
    gaintype = 'G'
    calmode = 'ap'

    solint = 'inf'
    combine = [str]
    refant = [str]
    solnorm = False
    append = False
    minblperant = 4
    minsnr = 3.0
    preavg = -1.0

    gaintable = [str]
    gainfield = Custom ([str], sep=';')
    interp = [str]
    opacity = [float]
    gaincurve = False
    parang = False

    @Custom ([str], sep=';')
    def spwmap (v):
        return [map (int, e.split (',')) for e in v]

    # gaincal keywords: splinetime npointaver phasewrap smodel
    # bandpass keywords: fillgaps degamp degphase visnorm maskcenter
    #   maskedge

    antenna = str
    field = str
    intent = str
    observation = str
    scan = str
    spw = str
    taql = str # msselect
    timerange = str
    uvrange = str

    loglevel = 'warn' # teeny hack for CLI


def gaincal (cfg):
    cb = util.tools.calibrater ()
    cb.open (filename=b(cfg.vis), compress=False, addcorr=False, addmodel=False)

    selkws = extractmsselect (cfg)
    selkws['chanmode'] = 'none' # ?
    cb.selectvis (**b(selkws))

    applyonthefly (cb, cfg)

    # Solve

    solkws = {}

    for k in 'append preavg minblperant minsnr solnorm'.split ():
        solkws[k] = getattr (cfg, k)

    for p in 'caltable:table calmode:apmode'.split ():
        ck, sk = p.split (':')
        solkws[sk] = getattr (cfg, ck)

    if isinstance (cfg.solint, (int, float)):
        solkws['t'] = '%fs' % cfg.solint # sugar
    else:
        solkws['t'] = str (cfg.solint)

    if isinstance (cfg.refant, six.string_types):
        solkws['refant'] = cfg.refant
    else:
        solkws['refant'] = ','.join (cfg.refant)

    solkws['combine'] = ','.join (cfg.combine)
    solkws['phaseonly'] = False
    solkws['type'] = cfg.gaintype.upper ()

    if solkws['type'] == 'GSPLINE':
        cb.setsolvegainspline (**b(solkws))
    elif solkws['type'] == 'BPOLY':
        cb.setsolvebandpoly (**b(solkws))
    else:
        cb.setsolve (**b(solkws))

    cb.solve ()
    cb.close ()


gaincal_cli = makekwcli (gaincal_doc, GaincalConfig, gaincal)


# gencal

gencal_doc = \
"""
casatask gencal vis=<MS> caltable=<TBL> caltype=<TYPE> [keywords...]

Generate certain calibration tables that don't need to be solved for from
the actual data.

vis=
  Input dataset

caltable=
  Output calibration table (appended to if preexisting)

caltype=
  The kind of table to generate:
  amp       - generic amplitude correction; needs parameter(s)
  ph        - generic phase correction; needs parameter(s)
  sbd       - single-band delay: phase slope for each SPW; needs parameter(s)
  mbd       - multi-band delay: phase slope for all SPWs; needs parameter(s)
  antpos    - antenna position corrections in ITRF; what you want; accepts parameter(s)
  antposvla - antenna position corrections in VLA frame; not what you want; accepts parameter(s)
  tsys      - tsys from ALMA syscal table
  swpow     - EVLA switched-power and requantizer gains ("experimental")
  opac      - tropospheric opacity; needs parameter
  gc        - (E)VLA elevation-dependent gain curve
  eff       - (E)VLA antenna efficiency correction
  gceff     - combination of "gc" and "eff"
  rq        - EVLA requantizer gains; not what you want
  swp/rq    - EVLA switched-power gains divided by "rq"; not what you want

parameter=
  Custom parameters for various caltypes. Dimensionality depends on selections applied.
  amp       - gain; dimensionless
  ph        - phase; degrees
  sbd       - delay; nanosec
  mbd       - delay; nanosec
  antpos    - position offsets; ITRF meters (or look up automatically for EVLA if unspecified)
  antposvla - position offsets; meters in VLA reference frame
  opac      - opacity; dimensionless (nepers?)

antenna=
pol=
spw=
  Selection keywords governing which solutions to generate and controlling shape
  of "parameter" keyword.
""" + loglevel_doc


class GencalConfig (ParseKeywords):
    vis = Custom (str, required=True)
    caltable = Custom (str, required=True)
    caltype = Custom (str, required=True)
    parameter = [float]

    antenna = str
    pol = str
    spw = str

    loglevel = 'warn'


def gencal (cfg):
    cb = util.tools.calibrater ()
    cb.open (filename=b(cfg.vis), compress=False, addcorr=False, addmodel=False)

    antenna = cfg.antenna or ''
    parameter = cfg.parameter

    if cfg.caltype == 'antpos' and cfg.antenna is None:
        # There's a Python module in casapy that implements this; I don't want
        # to shadow it entirely ...
        import cPickle as pickle
        from .scripting import CasapyScript

        script = os.path.join (os.path.dirname (__file__), 'cscript_genantpos.py')

        with CasapyScript (script, vis=cfg.vis) as cs:
            try:
                with open (os.path.join (cs.workdir, 'info.pkl'), 'rb') as f:
                    antenna = pickle.load (f)
                    parameter = pickle.load (f)
            except Exception:
                reraise_context ('interal casapy script seemingly failed; no info.pkl')

    cb.specifycal (caltable=b(cfg.caltable), time=b'', spw=b(cfg.spw or ''),
                   antenna=b(antenna), pol=b(cfg.pol or ''), caltype=b(cfg.caltype),
                   parameter=b(parameter))
    cb.close ()


gencal_cli = makekwcli (gencal_doc, GencalConfig, gencal)


# getopacities
#
# This is a casapy script.

getopacities_doc = \
"""
casatask getopacities <MS> <plotdest> [spwwidth1,spwwidth2]

Calculate atmospheric opacities in the MS's spectral windows from its weather
data. Output the opacities and save a plot of the weather conditions. Optionally
output opacities averaged over spectral windows; for instance, in a standard
VLA wideband setup, in which the data come in 16 spectral windows,

  casatask getopacities unglued.ms weather.png 8,8

will print 2 values, averaged over 8 spectral windows each.
"""

def getopacities (ms, plotdest):
    import cPickle as pickle
    from .scripting import CasapyScript

    script = os.path.join (os.path.dirname (__file__), 'cscript_getopacities.py')

    with CasapyScript (script, ms=ms, plotdest=plotdest) as cs:
        try:
            with open (os.path.join (cs.workdir, 'opac.npy'), 'rb') as f:
                opac = pickle.load (f)
        except Exception:
            reraise_context ('interal casapy script seemingly failed; no opac.npy')

    return opac


def getopacities_cli (argv):
    check_usage (getopacities_doc, argv, usageifnoargs=True)

    if len (argv) not in (3, 4):
        wrong_usage (getopacities_doc, 'expected 2 or 3 arguments')

    opac = getopacities (argv[1], argv[2])

    if len (argv) > 3:
        spwwidths = [int (x) for x in argv[3].split (',')]
        averaged = []
        idx = 0

        for width in spwwidths:
            averaged.append (opac[idx:idx+width].mean ())
            idx += width

        opac = averaged

    print ('opacity = [%s]' % (', '.join ('%.5f' % q for q in opac)))


# gpdetrend

gpdetrend_doc = \
"""
casatask gpdetrend caltable=

Remove long-term phase trends from a complex-gain calibration table. For each
antenna/spw/pol, the complex gains are divided into separate chunks (e.g., the
intention is for one chunk for each visit to the complex-gain calibrator). The
mean phase within each chunk is divided out. The effect is to remove long-term
phase trends from the calibration table, but preserve short-term ones.

caltable=MS
  The input calibration Measurement Set

maxtimegap=int
  Measured in minutes. Gaps between solutions of this duration or longer will
  lead to a new segment being considered. Default is four times the smallest
  time gap seen in the data set.
""" + loglevel_doc


class GpdetrendConfig (ParseKeywords):
    caltable = Custom (str, required=True)
    maxtimegap = int
    loglevel = 'warn'


def gpdetrend (cfg):
    from ... import numutil

    tb = util.tools.table ()

    tb.open (binary_type (cfg.caltable), nomodify=False)
    #fields = tb.getcol (b'FIELD_ID')
    spws = tb.getcol (b'SPECTRAL_WINDOW_ID')
    ants = tb.getcol (b'ANTENNA1')
    vals = tb.getcol (b'CPARAM')
    flags = tb.getcol (b'FLAG')
    times = tb.getcol (b'TIME')

    npol, unused, nsoln = vals.shape
    assert unused == 1, 'unexpected gain table structure!'
    vals = vals[:,0,:]
    flags = flags[:,0,:]

    # see what we've got

    def seen_values (data):
        return [idx for idx, count in enumerate (np.bincount (data)) if count]

    any_ok = ~(np.all (flags, axis=0)) # mask for solns where at least one pol is unflagged
    #seenfields = seen_values (fields[any_ok])
    seenspws = seen_values (spws[any_ok])
    seenants = seen_values (ants[any_ok])

    # time gap?

    if cfg.maxtimegap is not None:
        maxtimegap = cfg.maxtimegap * 60 # min => seconds
    else:
        stimes = np.sort (times)
        dt = np.diff (stimes)
        dt = dt[dt > 0]
        maxtimegap = 4 * dt.min ()

    # Remove average phase of each chunk

    for iant in seenants:
        for ipol in range (npol):
            filter = (ants == iant) & ~flags[ipol]

            for ispw in seenspws:
                # XXX: do something by field?
                sfilter = filter & (spws == ispw)
                t = times[sfilter]
                if not t.size:
                    continue

                for s in numutil.slice_around_gaps (t, maxtimegap):
                    w = np.where (sfilter)[0][s]
                    meanph = np.angle (vals[ipol,w].mean ())
                    vals[ipol,w] *= np.exp (-1j * meanph)

    # Rewrite and we're done.

    tb.putcol (b'CPARAM', vals.reshape ((npol, 1, nsoln)))
    tb.close ()


gpdetrend_cli = makekwcli (gpdetrend_doc, GpdetrendConfig, gpdetrend)


# gpdiagnostics
#
# Shim for a separate module

def gpdiagnostics_cli (argv):
    from .gpdiagnostics import gpdiagnostics_cli
    gpdiagnostics_cli (argv)


# gpplot
#
# See bpplot() -- CASA plotcal can do this in a certain sense, but it's slow
# and ugly.

gpplot_doc = \
"""
casatask gpplot caltable= dest=

Plot a gain calibration table. Currently, the supported format is a series of
pages showing amplitude and phase against time, with each page showing a
particular antenna and polarization. Polarizations are always reported as "R"
and "L" since the relevant information is not stored within the bandpass data
set.

caltable=MS
  The input calibration Measurement Set

dest=PATH
  If specified, plots are saved to this file -- the format is inferred
  from the extension, which must allow multiple pages to be saved. If
  unspecified, the plots are displayed using a Gtk3 backend.

dims=WIDTH,HEIGHT
  If saving to a file, the dimensions of a each page. These are in points
  for vector formats (PDF, PS) and pixels for bitmaps (PNG). Defaults to
  1000, 600.

margins=TOP,RIGHT,LEFT,BOTTOM
  If saving to a file, the plot margins in the same units as the dims.
  The default is 4 on every side.

maxtimegap=10
  Solutions separated by more than this number of minutes will be drawn
  with separate lines for clarity.

mjdrange=START,STOP
  If specified, gain solutions outside of the MJDs STOP and START will be
  ignored.
""" + loglevel_doc


class GpplotConfig (ParseKeywords):
    caltable = Custom (str, required=True)
    dest = str
    dims = [1000, 600]
    margins = [4, 4, 4, 4]
    maxtimegap = 10. # minutes
    mjdrange = [float]
    loglevel = 'warn'


def gpplot (cfg):
    import omega as om, omega.render
    from ... import numutil

    if len (cfg.mjdrange) not in (0, 2):
        raise Exception ('"mjdrange" parameter must provide exactly 0 or 2 numbers')

    if isinstance (cfg.dest, omega.render.Pager):
        # This is for non-CLI invocation.
        pager = cfg.dest
    elif cfg.dest is None:
        import omega.gtk3
        pager = om.makeDisplayPager ()
    else:
        pager = om.makePager (cfg.dest,
                              dims=cfg.dims,
                              margins=cfg.margins,
                              style=om.styles.ColorOnWhiteVector ())

    tb = util.tools.table ()

    tb.open (binary_type (cfg.caltable), nomodify=True)
    fields = tb.getcol (b'FIELD_ID')
    spws = tb.getcol (b'SPECTRAL_WINDOW_ID')
    ants = tb.getcol (b'ANTENNA1')
    vals = tb.getcol (b'CPARAM')
    flags = tb.getcol (b'FLAG')
    times = tb.getcol (b'TIME')
    tb.close ()

    tb.open (binary_type (os.path.join (cfg.caltable, 'ANTENNA')), nomodify=True)
    names = tb.getcol (b'NAME')
    tb.close ()

    npol, unused, nsoln = vals.shape
    assert unused == 1, 'unexpected gain table structure!'
    vals = vals[:,0,:]
    flags = flags[:,0,:]

    # Apply date filter by futzing with flags

    times /= 86400. # convert to MJD

    if len (cfg.mjdrange):
        flags |= (times < cfg.mjdrange[0]) | (times > cfg.mjdrange[1])

    # see what we've got

    def seen_values (data):
        return [idx for idx, count in enumerate (np.bincount (data)) if count]

    any_ok = ~(np.all (flags, axis=0)) # mask for solns where at least one pol is unflagged
    seenfields = seen_values (fields[any_ok])
    seenspws = seen_values (spws[any_ok])
    seenants = seen_values (ants[any_ok])

    field_offsets = dict ((fieldid, idx) for idx, fieldid in enumerate (seenfields))
    spw_offsets = dict ((spwid, idx) for idx, spwid in enumerate (seenspws))
    ant_offsets = dict ((antid, idx) for idx, antid in enumerate (seenants))

    # normalize phases to avoid distracting wraps

    mean_amps = np.ones ((len (seenants), npol, len (seenspws)))

    for iant in seenants:
        for ipol in range (npol):
            filter = (ants == iant) & ~flags[ipol]

            for ispw in seenspws:
                sfilter = filter & (spws == ispw)
                if not sfilter.any ():
                    continue

                meanph = np.angle (vals[ipol,sfilter].mean ())
                vals[ipol,sfilter] *= np.exp (-1j * meanph)
                meanam = np.abs (vals[ipol,sfilter]).mean ()
                mean_amps[ant_offsets[iant],ipol,spw_offsets[ispw]] = meanam

    # find plot limits

    min_time = times[any_ok].min ()
    max_time = times[any_ok].max ()
    mjdref = int (np.floor (min_time))
    times -= mjdref # convert to delta-MJD
    min_time = (min_time - mjdref)
    max_time = (max_time - mjdref)
    span = max_time - min_time
    max_time += 0.05 * span
    min_time -= 0.05 * span

    okvals = vals[np.where (~flags)]
    max_am = np.abs (okvals).max ()
    min_am = np.abs (okvals).min ()
    span = max_am - min_am
    max_am += 0.05 * span
    min_am -= 0.05 * span

    max_ph = np.angle (okvals, deg=True).max ()
    min_ph = np.angle (okvals, deg=True).min ()
    span = max_ph - min_ph
    max_ph += 0.05 * span
    min_ph -= 0.05 * span
    if max_ph > 160:
        max_ph = 180
    if min_ph < -160:
        min_ph = -180

    polnames = 'RL' # XXX: identification doesn't seem to be stored in cal table
    maxtimegap = cfg.maxtimegap / 1440 # => units of days

    for iant in seenants:
        for ipol in range (npol):
            filter = (ants == iant) & ~flags[ipol]
            p_am = om.RectPlot ()
            p_ph = om.RectPlot ()
            anyseen = False

            for ispw in seenspws:
                # XXX: do something by field
                sfilter = filter & (spws == ispw)
                t = times[sfilter]
                if not t.size:
                    continue

                v = vals[ipol,sfilter]
                a = np.abs (v)
                p = np.angle (v, deg=True)
                kt = '%s %s spw#%d' % (names[iant], polnames[ipol], ispw)

                for s in numutil.slice_around_gaps (t, maxtimegap):
                    tsub, asub, psub = t[s], a[s], p[s]
                    if tsub.size == 0:
                        continue # Should never happen, but eh.
                    else:
                        lines = (tsub.size > 1)

                    p_am.addXY (tsub, asub, kt, lines=lines, dsn=spw_offsets[ispw])
                    p_ph.addXY (tsub, psub, None, lines=lines, dsn=spw_offsets[ispw])
                    anyseen = True

                    if kt is not None: # hack for per-spw "anyseen"-type checking
                        p_am.addHLine (mean_amps[ant_offsets[iant],ipol,spw_offsets[ispw]], None,
                                       zheight=-1, lineStyle={'color': 'faint'})
                    kt = None

            if not anyseen:
                continue

            p_ph.addHLine (0, None, zheight=-1, lineStyle={'color': 'faint'})

            p_am.setBounds (xmin=min_time,
                            xmax=max_time,
                            ymin=min_am,
                            ymax=max_am)
            p_ph.setBounds (xmin=min_time,
                            xmax=max_time,
                            ymin=min_ph,
                            ymax=max_ph)

            p_am.bpainter.paintLabels = False
            p_am.setYLabel ('Amplitude')
            p_ph.setLabels ('Time (MJD - %d)' % mjdref, 'De-meaned Phase (deg)')

            vb = om.layout.VBox (2)
            vb[0] = p_am
            vb[1] = p_ph
            vb.setWeight (0, 2.5)
            pager.send (vb)

gpplot_cli = makekwcli (gpplot_doc, GpplotConfig, gpplot)


# image2fits
#
# This is basically the "exportfits" task with fewer options and a slightly
# clearer name.

image2fits_doc = \
"""
casatask image2fits <input MS image> <output FITS image>

Convert an image in MS format to FITS format.
"""

def image2fits (mspath, fitspath, velocity=False, optical=False, bitpix=-32,
                minpix=0, maxpix=-1, overwrite=False, dropstokes=False, stokeslast=True,
                history=True, **kwargs):
    ia = util.tools.image ()
    ia.open (b(mspath))
    ia.tofits (outfile=b(fitspath), velocity=velocity, optical=optical, bitpix=bitpix,
               minpix=minpix, maxpix=maxpix, overwrite=overwrite, dropstokes=dropstokes,
               stokeslast=stokeslast, history=history, **kwargs)
    ia.close ()


def image2fits_cli (argv):
    check_usage (image2fits_doc, argv, usageifnoargs=True)
    util.logger ()

    if len (argv) != 3:
        wrong_usage (image2fits_doc, 'expected exactly 2 arguments')

    image2fits (argv[1], argv[2])


# importevla
#
# This is a casapy script. We don't reeeallly need to be, but there's enough
# logic in CASA's task_importevla.py that I'm not thrilled to copy/paste it
# all.

importevla_doc = \
"""
casatask importevla <ASDM> <MS>

Convert an EVLA low-level ASDM dataset to Measurement Set format. This
implementation automatically infers the value of the "tbuff" parameter.
"""

def importevla (asdm, ms):
    from .scripting import CasapyScript

    # Here's the best way I can figure to find the recommended value of tbuff
    # (= 1.5 * integration time). Obviously you might have different
    # integration times in the dataset and such, and we're just going to
    # ignore that possibility.

    bdfstem = os.listdir (os.path.join (asdm, 'ASDMBinary'))[0]
    bdf = os.path.join (asdm, 'ASDMBinary', bdfstem)
    tbuff = None

    with open (bdf) as f:
        for linenum, line in enumerate (f):
            if linenum > 60:
                raise PKError ('cannot find integration time info in %s', bdf)

            if not line.startswith ('<sdmDataSubsetHeader'):
                continue

            try:
                i1 = line.index ('<interval>') + len ('<interval>')
                i2 = line.index ('</interval>')
                if i2 <= i1:
                    raise ValueError ()
            except ValueError:
                raise PKError ('cannot parse integration time info in %s', bdf)

            tbuff = float (line[i1:i2]) * 1.5e-9 # nanosecs, and want 1.5x
            break

    if tbuff is None:
        raise PKError ('found no integration time info')

    print ('importevla: %s -> %s with tbuff=%.2f' % (asdm, ms, tbuff))

    script = os.path.join (os.path.dirname (__file__), 'cscript_importevla.py')
    with CasapyScript (script, asdm=asdm, ms=ms, tbuff=tbuff) as cs:
        pass


def importevla_cli (argv):
    check_usage (importevla_doc, argv, usageifnoargs=True)

    if len (argv) != 3:
        wrong_usage (importevla_doc, 'expected exactly 2 arguments')

    importevla (argv[1], argv[2])


# listobs
#
# This one is mostly about the CLI.

listobs_doc = \
"""
casatask listobs <MS>

Generate a standard "listobs" listing of visibility MS contents. If
standard output is a TTY, the listing will be paged.
"""

def listobs (vis):
    """Generates a set of lines of output. Errors are only detected by looking
    at the output."""

    def inner_list (sink):
        try:
            ms = util.tools.ms ()
            ms.open (vis)
            ms.summary (verbose=True)
            ms.close ()
        except Exception as e:
            sink.post (b'listobs failed: %s' % e, priority=b'SEVERE')

    for line in util.forkandlog (inner_list):
        info = line.rstrip ().split ('\t', 3) # date, priority, origin, message
        if len (info) > 3:
            yield info[3]
        else:
            yield ''


def listobs_cli (argv):
    check_usage (listobs_doc, argv, usageifnoargs=True)

    if len (argv) != 2:
        wrong_usage (listobs_doc, 'expect exactly one argument, the MS path')

    vis = argv[1]

    proc = None
    out = sys.stdout

    if sys.stdout.isatty () or \
           (hasattr (sys.stdout, 'stream') and sys.stdout.stream.isatty ()):
        # Send our output to a pager!
        import subprocess
        pager = os.environ.get ('PAGER') or 'less -SRFX'
        try:
            proc = subprocess.Popen (pager, stdin=subprocess.PIPE, close_fds=True,
                                     shell=True)
        except Exception as e:
            warn ('couldn\'t start pager %r: %s', pager, e)
        else:
            out = proc.stdin

    for line in listobs (vis):
        print (line, file=out)

    if proc is not None:
        proc.stdin.close ()
        proc.wait () # ignore return code

# mfsclean
#
# This isn't a CASA task, but we're pulling out a subset of the functionality
# of clean, which has a bajillion options and has a really gross implementation
# in the library.

mfsclean_doc = \
"""
casatask mfsclean vis=[] [keywords]

Drive the CASA imager with a very restricted set of options.

For W-projection, set ftmachine='wproject' and wprojplanes=64 (or so).

vis=
  Input visibility MS

imbase=
  Base name of output files. We create files named "imbaseEXT"
  where EXT is all of "mask", "modelTT", "imageTT", "residualTT",
  and "psfTT", and TT is empty if nterms = 1, and "ttN." otherwise.

cell = 1 [arcsec]
ftmachine = 'ft' or 'wproject'
gain = 0.1
imsize = 256,256
mask = (blank) or path of CASA-format region text file
niter = 500
nterms = 1
phasecenter = (blank) or 'J2000 12h34m56.7 -12d34m56.7'
reffreq = 0 [GHz]
stokes = I
threshold = 0 [mJy]
weighting = 'briggs' (robust=0.5) or 'natural'
wprojplanes = 1

""" + stdsel_doc + loglevel_doc.replace ('warn;', 'info;')

class MfscleanConfig (ParseKeywords):
    vis = Custom (str, required=True)
    imbase = Custom (str, required=True)

    cell = 1. # arcsec
    ftmachine = 'ft'
    gain = 0.1
    imsize = [256, 256]
    mask = str
    minpb = 0.2
    niter = 500
    nterms = 1
    phasecenter = str
    reffreq = 0. # GHz; 0 -> be sensible
    stokes = 'I'
    threshold = 0. # mJy
    weighting = 'briggs'
    wprojplanes = 1

    # allowchunk = False
    # cyclefactor = 1.5
    # cyclespeedup = -1
    # imagermode = csclean
    # interactive = False
    # gridmode = '' -- related to ftmachine keyword
    # mode = mfs
    # modelimage = []
    # multiscale = []
    # nchan = -1
    # npixels = 0
    # pbcor = False
    # psfmode = clark
    # restoringbeam = []
    # robust = 0.5
    # smallscalebias = 0.6
    # usescratch = False
    # uvtaper = False
    # veltype = radio
    # width = 1

    antenna = str
    field = str
    observation = str
    scan = str
    spw = str
    timerange = str
    uvrange = str
    taql = str

    loglevel = 'info'


specframenames = 'REST LSRK LSRD BARY GEO TOPO GALACTO LGROUP CMB'.split ()


def mfsclean (cfg):
    ms = util.tools.ms ()
    im = util.tools.imager ()
    tb = util.tools.table ()
    qa = util.tools.quanta ()
    ia = util.tools.image ()

    # Filenames. TODO: make sure nothing exists

    mask = cfg.imbase + 'mask'
    pb = cfg.imbase + 'flux'

    if cfg.nterms == 1:
        models = [cfg.imbase + 'model']
        restoreds = [cfg.imbase + 'image']
        resids = [cfg.imbase + 'residual']
        psfs = [cfg.imbase + 'psf']
    else:
        # Note: the names for the 'alpha' and 'alpha.error' images are
        # generated automatically inside the C++ stuff by looking for image
        # names ending in 'tt0', so we're limited in our naming flexibility
        # here.
        models, restoreds, resids, psfs = [], [], [], []

        for i in range (cfg.nterms):
            models.append (cfg.imbase + 'model.tt%d' % i)
            restoreds.append (cfg.imbase + 'image.tt%d' % i)
            resids.append (cfg.imbase + 'residual.tt%d' % i)
            psfs.append (cfg.imbase + 'psf.tt%d' % i)

    # Get info on our selected data for various things we need to figure
    # out later.

    selkws = extractmsselect (cfg, havearray=False, haveintent=False, taqltomsselect=False)
    ms.open (b(cfg.vis))
    ms.msselect (b(selkws))
    rangeinfo = ms.range (b'data_desc_id field_id'.split ())
    ddids = rangeinfo['data_desc_id']
    fields = rangeinfo['field_id']

    # Get the spectral frame from the first spw of the selected data

    tb.open (b(os.path.join (cfg.vis, 'DATA_DESCRIPTION')))
    ddspws = tb.getcol (b'SPECTRAL_WINDOW_ID')
    tb.close ()
    spw0 = ddspws[ddids[0]]

    tb.open (b(os.path.join (cfg.vis, 'SPECTRAL_WINDOW')))
    specframe = specframenames[tb.getcell (b'MEAS_FREQ_REF', spw0)]
    tb.close ()

    # Choose phase center

    if cfg.phasecenter is None:
        phasecenter = int (fields[0])
    else:
        phasecenter = cfg.phasecenter

        if ':' in phasecenter:
            # phasecenter='J2000 19:06:48.568 40:11:06.68'
            # parses to "07:06:48.57 297.13.19.80"
            # Using hm/dm instead of ::/:: as separators makes it parse
            # correctly. No idea how the colons manage to parse both
            # without warning and totally wrongly.
            warn ('you moron, who uses colons in sexagesimal?')
            tmp1 = phasecenter.split ()
            twiddled = False
            if len (tmp1) == 3:
                tmp2 = tmp1[1].split (':')
                tmp3 = tmp1[2].split (':')
                if len (tmp2) == 3 and len (tmp3) == 3:
                    tmp2 = tmp2[0] + 'h' + tmp2[1] + 'm' + tmp2[2]
                    tmp3 = tmp3[0] + 'd' + tmp3[1] + 'm' + tmp3[2]
                    phasecenter = ' '.join ([tmp1[0], tmp2, tmp3])
                    twiddled = True
            if twiddled:
                warn ('attempted to fix it up: "%s"\n\n', phasecenter)
            else:
                warn ('couldn\'t parse, left as-is; good luck\n\n')

    # Set up all of this junk

    im.open (b(cfg.vis), usescratch=False)
    im.selectvis (nchan=-1, start=0, step=1, usescratch=False, writeaccess=False, **b(selkws))
    im.defineimage (nx=cfg.imsize[0], ny=cfg.imsize[1],
                    cellx=qa.quantity (b(cfg.cell), b'arcsec'),
                    celly=qa.quantity (b(cfg.cell), b'arcsec'),
                    outframe=b(specframe), phasecenter=b(phasecenter),
                    stokes=cfg.stokes,
                    spw=-1, # to verify: selectvis (spw=) good enough?
                    restfreq=b'', mode=b'mfs', veltype=b'radio',
                    nchan=-1, start=0, step=1, facets=1)

    if cfg.weighting == 'briggs':
        im.weight (type=b'briggs', rmode=b'norm', robust=0.5, npixels=0) #noise=, mosaic=
    elif cfg.weighting == 'natural':
        im.weight (type=b'natural', rmode=b'none')
    else:
        raise ValueError ('unknown weighting type "%s"' % cfg.weighting)

    # im.filter (...)
    im.setscales (scalemethod=b'uservector', uservector=[0])
    im.setsmallscalebias (0.6)
    im.setmfcontrol ()
    im.setvp (dovp=True)
    im.makeimage (type=b'pb', image=b(pb), compleximage=b'', verbose=False)
    im.setvp (dovp=False, verbose=False)
    im.setoptions (ftmachine=b(cfg.ftmachine), wprojplanes=b(cfg.wprojplanes),
                   freqinterp=b'linear', padding=1.2, pastep=360.0, pblimit=b(cfg.minpb),
                   applypointingoffsets=False, dopbgriddingcorrections=True)

    if cfg.nterms > 1:
        im.settaylorterms (ntaylorterms=cfg.nterms, reffreq=cfg.reffreq * 1e9)

    im.setmfcontrol (stoplargenegatives=-1, cyclefactor=1.5,
                     cyclespeedup=-1, minpb=cfg.minpb)

    # Create the mask

    if cfg.mask is None:
        maskstr = ''
    else:
        maskstr = mask
        im.make (b(mask))
        ia.open (b(mask))
        maskcs = ia.coordsys ()
        maskcs.setreferencecode (b(specframe), b'spectral', True)
        ia.setcoordsys (maskcs.torecord ())

        if cfg.mask is not None:
            rg = util.tools.regionmanager ()
            regions = rg.fromtextfile (filename=b(cfg.mask),
                                       shape=ia.shape (),
                                       csys=maskcs.torecord ())
            im.regiontoimagemask (mask=b(mask), region=regions)

        ia.close ()

    # Create blank model(s). Diverging from task_clean even more
    # significantly than usual here.

    for model in models:
        im.make (b(model))

    # Go!

    im.clean (algorithm=b'msmfs', niter=cfg.niter, gain=cfg.gain,
              threshold=qa.quantity (b(cfg.threshold), b'mJy'),
              model=b(models), residual=b(resids), image=b(restoreds),
              psfimage=b(psfs), mask=b(maskstr), interactive=False)
    im.close ()


mfsclean_cli = makekwcli (mfsclean_doc, MfscleanConfig, mfsclean)


# mjd2date

mjd2date_doc = \
"""
casatask mjd2date <date>

Convert an MJD to a date in the format used by CASA.

"""
def mjd2date (mjd, precision=3):
    from astropy.time import Time
    dt = Time (mjd, format='mjd', scale='utc').to_datetime ()
    fracsec = ('%.*f' % (precision, 1e-6 * dt.microsecond)).split ('.')[1]
    return '%04d/%02d/%02d/%02d:%02d:%02d.%s' % (
        dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, fracsec
    )


def mjd2date_cli (argv):
    check_usage (mjd2date_doc, argv, usageifnoargs=True)

    if len (argv) != 2:
        wrong_usage (mjd2date_doc, 'expect exactly one argument')

    print (mjd2date (float (argv[1])))


# mstransform

mstransform_doc = \
"""
casatask mstransform vis=[] [keywords]

vis=
  Input visibility MS

out=
  Output visibility MS

datacolumn=corrected
  The data column on which to operate. Comma-separated list of:
    data, model, corrected, float_data, lag_data, all

realmodelcol=False
  If true, turn a virtual model column into a real one.

keepflags=True
  If false, discard completely-flagged rows.

usewtspectrum=False
  If true, fill in a WEIGHT_SPECTRUM column in the output data set.

combinespws=False
  If true, combine spectral windows

chanaverage=False
  If true, average the data in frequency. NOT WIRED UP.

hanning=False
  If true, Hanning smooth the data spectrally to remove Gibbs ringing.

regridms=False
  If true, put the data on a new spectral window structure or reference frame.

timebin=<seconds>
  If specified, time-average the visibilities with the specified binning.

timespan=<undefined>
  Allow averaging to span over potential discontinuities in the data set.
  Comma-separated list of options; allowed values are:
    scan, state

""" + stdsel_doc + loglevel_doc

class MstransformConfig (ParseKeywords):
    vis = Custom (str, required=True)
    out = Custom (str, required=True)

    datacolumn = 'corrected'
    realmodelcol = False
    keepflags = True
    usewtspectrum = False
    combinespws = False
    chanaverage = False
    hanning = False
    regridms = False

    timebin = float # seconds
    timespan = [str] # containing 'scan', 'state'

    antenna = str
    array = str
    correlation = str
    field = str
    intent = str
    observation = str
    scan = str
    spw = str
    timerange = str
    uvrange = str
    taql = str

    loglevel = 'warn'


def mstransform (cfg):
    mt = util.tools.mstransformer ()
    qa = util.tools.quanta ()

    mtconfig = extractmsselect (cfg, havearray=True, havecorr=True, taqltomsselect=False)
    mtconfig['inputms'] = cfg.vis
    mtconfig['outputms'] = cfg.out

    if not cfg.keepflags:
        if len (mtconfig.get ('taql', '')):
            mtconfig['taql'] += 'AND NOT (FLAG_ROW OR ALL(FLAG))'
        else:
            mtconfig['taql'] = 'NOT (FLAG_ROW OR ALL(FLAG))'

    mtconfig['datacolumn'] = cfg.datacolumn

    if 'MODEL' in cfg.datacolumn.upper () or cfg.datacolumn.upper () == 'ALL':
        mtconfig['realmodelcol'] = cfg.realmodelcol

    mtconfig['combinespws'] = bool (cfg.combinespws)
    mtconfig['hanning'] = bool (cfg.hanning)

    if cfg.chanaverage:
        raise NotImplementedError ('mstransform: chanaverage')

    if cfg.regridms:
        raise NotImplementedError ('mstransform: regridms')

    if cfg.timebin is not None:
        mtconfig['timeaverage'] = True
        mtconfig['timebin'] = str (cfg.timebin) + 's'
        mtconfig['timespan'] = ','.join (cfg.timespan)
        # not implemented: maxuvwdistance

    mt.config (b(mtconfig))
    mt.open ()
    mt.run ()
    mt.done ()

    # not implemented: updating FLAG_CMD table
    # not implemented: updating history


mstransform_cli = makekwcli (mstransform_doc, MstransformConfig, mstransform)


# plotants

plotants_doc = \
"""
casatask plotants <MS> <figfile>

Plot the physical layout of the antennas described in the MS.
"""

def plotants (vis, figfile):
    from .scripting import CasapyScript

    script = os.path.join (os.path.dirname (__file__), 'cscript_plotants.py')

    with CasapyScript (script, vis=vis, figfile=figfile) as cs:
        pass


def plotants_cli (argv):
    check_usage (plotants_doc, argv, usageifnoargs=True)

    if len (argv) != 3:
        wrong_usage (plotants_doc, 'expect exactly two arguments')

    plotants (argv[1], argv[2])


# plotcal

plotcal_doc = \
"""
casatask plotcal caltable=<MS> [keywords]

Plot values from a calibration dataset in any of a variety of ways.

caltable=
  The calibration MS to plot

xaxis=
  amp antenna chan freq imag phase real snr time

yaxis=
  amp antenna imag phase real snr

iteration=
  antenna field spw time

*** Data selection

antenna=
field=
poln=
  RL R L XY X Y '/'
spw=
timerange=

*** Plot appearance options

plotsymbol=
plotcolor=
fontsize=
figfile=

""" + loglevel_doc


class PlotcalConfig (ParseKeywords):
    caltable = Custom (str, required=True)
    xaxis = 'time'
    yaxis = 'amp'
    iteration = ''

    # not implemented: subplot, overplot, clearpanel, plotrange,
    # showflags, showgui

    plotsymbol = '.'
    plotcolor = 'blue'
    markersize = 5.
    fontsize = 10.
    figfile = str

    antenna = ''
    field = ''
    poln = 'RL'
    spw = ''
    timerange = ''


def plotcal (cfg):
    # casa-tools plotting relies on invoking matplotlib in an internal Python
    # interpreter (!), and it uses a very old version of matplotlib that's
    # essentially incompatible with what's available in any up-to-date Python
    # environment (e.g. Anaconda). Therefore we have to launch any plotcal
    # operations as casapy scripts to ensure compatibility.

    from .scripting import CasapyScript

    script = os.path.join (os.path.dirname (__file__), 'cscript_plotcal.py')

    selectcals = b(dict (antenna = cfg.antenna,
                         field = cfg.field,
                         poln = cfg.poln.upper (),
                         spw = cfg.spw,
                         time = cfg.timerange))

    plotoptions = b(dict (iteration = cfg.iteration,
                          plotrange = [0.0]*4,
                          plotsymbol = cfg.plotsymbol,
                          plotcolor = cfg.plotcolor,
                          markersize = cfg.markersize,
                          fontsize = cfg.fontsize))

    with CasapyScript (script,
                       caltable=b(cfg.caltable),
                       selectcals=selectcals,
                       plotoptions=plotoptions,
                       xaxis=b(cfg.xaxis.upper ()),
                       yaxis=b(cfg.yaxis.upper ()),
                       figfile=b(cfg.figfile)) as cs:
        pass


def plotcal_cli (argv):
    check_usage (plotcal_doc, argv, usageifnoargs=True)
    cfg = PlotcalConfig ().parse (argv[1:])
    plotcal (cfg)


# polmodel
#
# Shim for a separate module

def polmodel_cli (argv):
    from .polmodel import polmodel_cli
    polmodel_cli (argv)


# setjy

setjy_doc = \
"""
casatask setjy vis= [keywords]

Insert model data into a measurement set. We force usescratch=False
and scalebychan=True. You probably want to specify "field".

fluxdensity=
  Up to four comma-separated numbers giving Stokes IQUV intensities in
  Jy. Default values are [-1, 0, 0, 0]. If the Stokes I intensity is
  negative (i.e., the default), a "sensible default" will be used:
  detailed spectral models if the source is known (see "standard"), or
  1 otherwise. If it is zero and "modimage" is used, the flux density
  of the model image is used. The built-in standards do NOT have
  polarimetric information, so for pol cal you do need to manually
  specify the flux density information -- or see the program
  "mspolmodel".

modimage=
  An image to use as the basis for the source's spatial structure and,
  potentialy, flux density (if fluxdensity=0). Only usable for Stokes
  I.  If the verbatim value of "modimage" can't be opened as a path,
  it is assumed to be relative to the CASA data directory; a typical
  value might be "nrao/VLA/CalModels/3C286_C.im".

spindex=
reffreq=
  If using fluxdensity, these specify the spectral dependence of the
  values, such that S = fluxdensity * (freq/reffreq)**spindex. Reffreq
  is in GHz. Default values are 0 and 1, giving no spectral
  dependence.

standard='Perley-Butler 2013'
  Solar-system standards are not supported by this implementation, so
  acceptable values are: Baars, Perley 90, Perley-Taylor 95,
  Perley-Taylor 99, Perley-Butler 2010, Perley-Butler 2013.

*** Supported data selection keywords:

field=
observation=
scan=
spw=
timerange=
""" + loglevel_doc

class SetjyConfig (ParseKeywords):
    vis = Custom (str, required=True)
    modimage = str
    fluxdensity = [-1., 0., 0., 0.]
    spindex = 0.
    reffreq = 1. # GHz
    standard = 'Perley-Butler 2013'

    field = str
    observation = str
    scan = str
    spw = str
    timerange = str

    loglevel = 'warn'


def setjy (cfg):
    kws = {}

    for kw in 'field fluxdensity observation scan spw standard'.split ():
        kws[kw] = getattr (cfg, kw) or ''

    kws['time'] = cfg.timerange or ''
    kws['reffreq'] = str (cfg.reffreq) + 'GHz'
    kws['spix'] = cfg.spindex
    kws['scalebychan'] = True # don't think you'd ever want false??

    if cfg.modimage is None:
        kws['modimage'] = ''
    else:
        if os.path.isdir (cfg.modimage):
            mi = cfg.modimage
        else:
            mi = util.datadir (cfg.modimage)
            if not os.path.isdir (mi):
                raise RuntimeError ('no model image "%s" or "%s"' % (cfg.modimage, mi))
        kws['modimage'] = mi

    im = util.tools.imager ()
    im.open (b(cfg.vis), usescratch=False) # don't think you'll ever want True?
    im.setjy (**b(kws))
    im.close ()


setjy_cli = makekwcli (setjy_doc, SetjyConfig, setjy)


# split
#
# note: spw=999 -> exception; scan=999 -> no output, or error, generated

split_doc = \
"""
casatask split vis=<MS> out=<MS> [keywords...]

timebin=
  Time-average data into bins of "timebin" seconds; defaults to no averaging

step=
  Frequency-average data in bins of "step" channels; defaults to no averaging

col=all
  Extract the column "col" as the DATA column. If "all", copy all available
  columns without renaming. Possible values:
    all DATA MODEL_DATA CORRECTED_DATA FLOAT_DATA LAG_DATA

combine=[col1,col2,...]
  When time-averaging, don't start a new bin when the specified columns change.
  Acceptable column names:
    scan state
""" + stdsel_doc + loglevel_doc

class SplitConfig (ParseKeywords):
    vis = Custom (str, required=True)
    out = Custom (str, required=True)

    timebin = float # seconds
    step = 1
    col = 'all'
    combine = [str]

    antenna = str
    array = str
    correlation = str
    field = str
    intent = str
    observation = str # renamed from obs for consistency
    scan = str
    spw = str
    taql = str
    timerange = str
    uvrange = str

    loglevel = 'warn'


def split (cfg):
    import tempfile, shutil

    kws = extractmsselect (cfg, havearray=True, havecorr=True,
                           observationtoobs=True, taqltomsselect=False)
    kws['whichcol'] = cfg.col
    kws['combine'] = ','.join (cfg.combine)
    kws['step'] = [cfg.step] # can be done on per-spw basis; we skip that

    if cfg.timebin is None:
        kws['timebin'] = '-1s'
    else:
        kws['timebin'] = str (cfg.timebin) + 's'

    ms = util.tools.ms ()
    ms.open (b(cfg.vis))

    # split() will merrily overwrite an existing MS, which I think is
    # very bad behavior. We try to prevent this in two steps: 1) claim
    # the desired output name in a way that will error out if it
    # already exists; 2) tell split() to create its outputs in an
    # empty temporary directory, to minimize the chances of blowing
    # away anything preexisting. In the pathological case, there's a
    # chance for someone with our UID to move something into the
    # temporary directory with our target name and have us delete
    # it. There's nothing we can do about that so long as split() is
    # happy to overwrite existing data.
    #
    # It's also possible for someone with our UID to spoil our rename
    # by changing the permissions on our placeholder output directory
    # and stuffing something in it, but this failure mode doesn't
    # involve data loss.
    #
    # We put the temporary working directory adjacent to the destination
    # to make sure it's on the same device.

    didntmakeit = True
    renamed = False
    workdir = None

    try:
        didntmakeit = os.mkdir (cfg.out, 0) # error raised if already exists.

        try:
            workdir = tempfile.mkdtemp (dir=os.path.dirname (cfg.out),
                                        prefix=os.path.basename (cfg.out) + '_')
            kws['outputms'] = os.path.join (workdir, os.path.basename (cfg.out))
            ms.split (**b(kws))
            os.rename (kws['outputms'], cfg.out)
            renamed = True
        finally:
            if workdir is not None:
                shutil.rmtree (workdir, ignore_errors=True)
    finally:
        if not didntmakeit and not renamed:
            try:
                os.rmdir (cfg.out)
            except:
                pass

    ms.close ()


split_cli = makekwcli (split_doc, SplitConfig, split)


# spwglue
#
# Shim for a separate module

def spwglue_cli (argv):
    from .spwglue import spwglue_cli
    spwglue_cli (argv)


# uvsub
#
# We add UV selection keywords not supported by the CASA task.
# I assume that they're honored ...

uvsub_doc = \
"""
casatask uvsub vis= [keywords]

Set the CORRECTED_DATA column to the difference of DATA and MODEL_DATA.

vis=
  The input data set.

reverse=
  Boolean, default false, which means to set CORRECTED = DATA - MODEL. If
  true, CORRECTED = DATA + MODEL.
""" + stdsel_doc + loglevel_doc


class UvsubConfig (ParseKeywords):
    vis = Custom (str, required=True)
    reverse = False

    antenna = str
    array = str
    field = str
    intent = str
    observation = str
    scan = str
    spw = str
    timerange = str
    uvrange = str
    taql = str

    loglevel = 'warn'


def uvsub (cfg):
    ms = util.tools.ms ()

    ms.open (b(cfg.vis), nomodify=False)
    ms.msselect (b(extractmsselect (cfg,
                                    havearray=True,
                                    intenttoscanintent=True,
                                    taqltomsselect=False)))
    ms.uvsub (reverse=cfg.reverse)
    ms.close ()


uvsub_cli = makekwcli (uvsub_doc, UvsubConfig, uvsub)


# Driver for command-line access. I wrote this before multitool, and it
# doesn't seem particularly valuable to convert them to Multitool since the
# current system works fine.

def cmdline_usage (stream, exitcode):
    print ('usage: casatask <task> [task-specific arguments]', file=stream)
    print (file=stream)
    print ('Supported tasks:', file=stream)
    print (file=stream)

    for name in sorted (six.iterkeys (globals ())):
        if name.endswith ('_cli'):
            print (name[:-4], file=stream)

    raise SystemExit (exitcode)


def commandline (argv=None):
    if argv is None:
        argv = sys.argv
        from ... import cli
        cli.propagate_sigint ()
        cli.backtrace_on_usr1 ()
        cli.unicode_stdio ()

    if len (argv) < 2 or argv[1] == '--help':
        cmdline_usage (sys.stdout, 0)

    driver = globals ().get (argv[1] + '_cli')
    if driver is None:
        die ('unknown task "%s"; run with no arguments for a list', argv[1])

    subargv = [' '.join (argv[:2])] + argv[2:]
    driver (subargv)
