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

from . import util
from ... import binary_type, reraise_context, text_type
from ...cli import check_usage, wrong_usage, warn, die
from ...kwargv import ParseKeywords, Custom

# Keep the tasks alphabetized!

__all__ = (b'''
applycal applycal_cli ApplycalConfig
clearcal clearcal_cli
concat concat_cli
delcal delcal_cli
delmod delmod_cli
flagmanager_cli
flaglist flaglist_cli FlaglistConfig
flagzeros flagzeros_cli FlagzerosConfig
fluxscale fluxscale_cli FluxscaleConfig
ft ft_cli FtConfig
gaincal gaincal_cli GaincalConfig
gencal gencal_cli GencalConfig
getopacities_cli
image2fits image2fits_cli
mfsclean mfsclean_cli MfscleanConfig
plotants plotants_cli
plotcal plotcal_cli plotcal_multipage_page PlotcalConfig
setjy setjy_cli SetjyConfig
split split_cli SplitConfig
spwglue_cli
uvsub uvsub_cli UvsubConfig
cmdline_driver
''').split ()


# Some utilities

def b (item):
    if isinstance (item, text_type):
        return item.encode ('utf8')
    return item


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
        cb.setapply (table=table, field=field, interp=interp, spwmap=spwmap,
                     t=0., calwt=True)

    if len (cfg.opacity):
        cb.setapply (type='TOPAC', opacity=cfg.opacity, t=-1, calwt=True)

    if cfg.gaincurve:
        cb.setapply (type='GAINCURVE', t=-1, calwt=True)

    if cfg.parang:
        cb.setapply (type='P')


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
    cb.open (filename=cfg.vis, compress=False, addcorr=True, addmodel=False)

    selkws = extractmsselect (cfg)
    selkws['chanmode'] = 'none' # ?
    cb.selectvis (**selkws)

    applyonthefly (cb, cfg)

    cb.correct (cfg.applymode)
    cb.close ()


applycal_cli = makekwcli (applycal_doc, ApplycalConfig, applycal)


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

    tb.open (vis, nomodify=False)
    colnames = tb.colnames ()
    needinit = ('MODEL_DATA' in colnames) or ('CORRECTED_DATA' in colnames)
    if 'IMAGING_WEIGHT' not in colnames:
        c = dict (clearcal_imaging_col_tmpl)
        c['shape'] = tb.getcell ('DATA', 0).shape[-1:]
        tb.addcols ({'IMAGING_WEIGHT': c}, clearcal_imaging_dminfo_tmpl)
    tb.close ()

    if not weightonly:
        cb.open (vis)
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
        clearcal (vis, weightonly=weightonly)


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
    import os.path
    tb = util.tools.table ()
    ms = util.tools.ms ()

    if os.path.exists (outvis):
        raise RuntimeError ('output "%s" already exists' % outvis)

    for invis in invises:
        if not os.path.isdir (invis):
            raise RuntimeError ('input "%s" does not exist' % invis)

    tb.open (invises[0])
    tb.copy (outvis, deep=True, valuecopy=True)
    tb.close ()

    ms.open (outvis, nomodify=False)

    for invis in invises[1:]:
        ms.concatenate (msfile=invis, freqtol=concat_freqtol,
                        dirtol=concat_dirtol)

    ms.writehistory (message='taskname=tasklib.concat', origin='tasklib.concat')
    ms.writehistory (message='vis = ' + ', '.join (invises), origin='tasklib.concat')
    ms.writehistory (message='timesort = ' + 'FT'[int (timesort)], origin='tasklib.concat')

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
    tb.open (mspath, nomodify=False)
    cols = frozenset (tb.colnames ())
    toremove = [c for c in wantremove if c in cols]
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
        cb.open (mspath, addcorr=False, addmodel=False)
        cb.delmod (otf=True, scr=False)
        cb.close ()


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
    import os.path
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
    af.open (ms)

    if mode == 'list':
        if len (argv) != 3:
            wrong_usage (flagmanager_doc, 'expect exactly one argument in list mode')
        af.getflagversionlist ()
    elif mode == 'save':
        if len (argv) != 4:
            wrong_usage (flagmanager_doc, 'expect exactly two arguments in save mode')
        from time import strftime
        name = argv[3]
        af.saveflagversion (versionname=name, merge='replace',
                            comment='created %s (casatask flagmanager)'
                            % strftime ('%Y-%m-%dT%H:%M:%SZ'))
    elif mode == 'restore':
        if len (argv) != 4:
            wrong_usage (flagmanager_doc, 'expect exactly two arguments in restore mode')
        name = argv[3]
        af.restoreflagversion (versionname=name, merge='replace')
    elif mode == 'delete':
        if len (argv) != 4:
            wrong_usage (flagmanager_doc, 'expect exactly two arguments in delete mode')
        name = argv[3]

        if not os.path.isdir (os.path.join (ms + '.flagversions', 'flags.' + name)):
            # This condition only results in a WARN from deleteflagversion ()!
            raise RuntimeError ('version "%s" doesn\'t exist in "%s.flagversions"'
                                % (name, ms))

        af.deleteflagversion (versionname=name)
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
    af.open (cfg.vis, 0.0)
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

        if not af.parseagentparameters (params):
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
    af.open (cfg.vis, 0.0)
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

    cb.open (cfg.vis, compress=False, addcorr=False, addmodel=False)
    result = cb.fluxscale (tablein=cfg.caltable, tableout=cfg.fluxtable,
                           reference=reference, transfer=transfer,
                           listfile=cfg.listfile or '',
                           append=cfg.append, refspwmap=refspwmap)
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

    im.open (cfg.vis, usescratch=cfg.usescratch)
    im.selectvis (**extractmsselect (cfg, haveintent=False,
                                     taqltomsselect=False))
    nmodel = len (cfg.model)

    if nmodel > 1:
        ia = util.tools.image ()
        ia.open (cfg.model[0])
        # This gives Hz:
        reffreq = ia.coordsys ().referencevalue (type='spectral')['numeric'][0]
        ia.close ()
        im.settaylorterms (ntaylorterms=nmodel, reffreq=reffreq)

    if cfg.wprojplanes is not None:
        im.defineimage ()
        im.setoptions (ftmachine='wproject', wprojplanes=cfg.wprojplanes)

    im.ft (model=cfg.model,
           complist=cfg.complist or '',
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
    cb.open (filename=cfg.vis, compress=False, addcorr=False, addmodel=False)

    selkws = extractmsselect (cfg)
    selkws['chanmode'] = 'none' # ?
    cb.selectvis (**selkws)

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

    if isinstance (cfg.refant, basestring):
        solkws['refant'] = cfg.refant
    else:
        solkws['refant'] = ','.join (cfg.refant)

    solkws['combine'] = ','.join (cfg.combine)
    solkws['phaseonly'] = False
    solkws['type'] = cfg.gaintype.upper ()

    if solkws['type'] == 'GSPLINE':
        cb.setsolvegainspline (**solkws)
    elif solkws['type'] == 'BPOLY':
        cb.setsolvebandpoly (**solkws)
    else:
        cb.setsolve (**solkws)

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
    cb.open (filename=cfg.vis, compress=False, addcorr=False, addmodel=False)

    antenna = cfg.antenna or ''
    parameter = cfg.parameter

    if cfg.caltype == 'antpos' and cfg.antenna is None:
        from correct_ant_posns import correct_ant_posns
        info = correct_ant_posns (cfg.vis, False)
        if len (info) != 3 or info[0] != 0 or not len (info[1]):
            import sys
            print ('correct_ant_posns: got %r' % info, file=sys.stderr)
            raise RuntimeError ('failed to fetch VLA antenna positions')

        antenna = info[1]
        parameter = info[2]

    cb.specifycal (caltable=cfg.caltable, time='', spw=(cfg.spw or ''),
                   antenna=antenna, pol=(cfg.pol or ''), caltype=cfg.caltype,
                   parameter=parameter)
    cb.close ()


gencal_cli = makekwcli (gencal_doc, GencalConfig, gencal)


# getopacities
#
# This is a casapy script. TODO: we could make the script callable as a
# function and not CLI-only with a bit of work.

getopacities_doc = \
"""
casatask getopacities <MS> <spwwidth1,spwwidth2,...> <plotdest>

Calculate atmospheric opacities in the MS's spectral windows from its weather
data. Output the opacities and save a plot of the weather conditions.
"""

def getopacities (ms, plotdest):
    import os.path, cPickle as pickle
    from .scripting import CasapyScript

    script = os.path.join (os.path.dirname (__file__), 'cscript_getopacities.py')

    with CasapyScript (script, [ms, plotdest]) as cs:
        try:
            with open (os.path.join (cs.workdir, 'opac.npy'), 'rb') as f:
                opac = pickle.load (f)
        except Exception:
            reraise_context ('interal casapy script seemingly failed; no opac.npy')

    return opac


def getopacities_cli (argv):
    check_usage (getopacities_doc, argv, usageifnoargs=True)

    if len (argv) != 3:
        wrong_usage (getopacities_doc, 'expected exactly 2 arguments')

    opac = getopacities (argv[1], argv[2])
    print ('opac = ', opac)


# image2fits
#
# This is basically the "exportfits" task with fewer options and a slightly
# clearer name (which is also consistent with the pwimage2fits script).

image2fits_doc = \
"""
casatask image2fits <input MS image> <output FITS image>

Convert an image in MS format to FITS format.
"""

def image2fits (mspath, fitspath):
    ia = util.tools.image ()
    ia.open (mspath)
    ia.tofits (outfile=fitspath, velocity=False, optical=False, bitpix=-32,
               minpix=0, maxpix=-1, overwrite=False, dropstokes=False,
               stokeslast=True, history=True)
    ia.close ()


def image2fits_cli (argv):
    check_usage (image2fits_doc, argv, usageifnoargs=True)
    util.logger ()

    if len (argv) != 3:
        wrong_usage (image2fits_doc, 'expected exactly 2 arguments')

    image2fits (argv[1], argv[2])


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
    import os.path

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

        for i in xrange (cfg.nterms):
            models.append (cfg.imbase + 'model.tt%d' % i)
            restoreds.append (cfg.imbase + 'image.tt%d' % i)
            resids.append (cfg.imbase + 'residual.tt%d' % i)
            psfs.append (cfg.imbase + 'psf.tt%d' % i)

    # Get info on our selected data for various things we need to figure
    # out later.

    selkws = extractmsselect (cfg, havearray=False, haveintent=False, taqltomsselect=False)
    ms.open (cfg.vis)
    ms.msselect (selkws)
    rangeinfo = ms.range ('data_desc_id field_id'.split ())
    ddids = rangeinfo['data_desc_id']
    fields = rangeinfo['field_id']

    # Get the spectral frame from the first spw of the selected data

    tb.open (os.path.join (cfg.vis, 'DATA_DESCRIPTION'))
    ddspws = tb.getcol ('SPECTRAL_WINDOW_ID')
    tb.close ()
    spw0 = ddspws[ddids[0]]

    tb.open (os.path.join (cfg.vis, 'SPECTRAL_WINDOW'))
    specframe = specframenames[tb.getcell ('MEAS_FREQ_REF', spw0)]
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

    im.open (cfg.vis, usescratch=False)
    im.selectvis (nchan=-1, start=0, step=1, usescratch=False, writeaccess=False, **selkws)
    im.defineimage (nx=cfg.imsize[0], ny=cfg.imsize[1],
                    cellx=qa.quantity (cfg.cell, 'arcsec'),
                    celly=qa.quantity (cfg.cell, 'arcsec'),
                    outframe=specframe, phasecenter=phasecenter,
                    stokes=cfg.stokes,
                    spw=-1, # to verify: selectvis (spw=) good enough?
                    restfreq='', mode='mfs', veltype='radio',
                    nchan=-1, start=0, step=1, facets=1)

    if cfg.weighting == 'briggs':
        im.weight (type='briggs', rmode='norm', robust=0.5, npixels=0) #noise=, mosaic=
    elif cfg.weighting == 'natural':
        im.weight (type='natural', rmode='none')
    else:
        raise ValueError ('unknown weighting type "%s"' % cfg.weighting)

    # im.filter (...)
    im.setscales (scalemethod='uservector', uservector=[0])
    im.setsmallscalebias (0.6)
    im.setmfcontrol ()
    im.setvp (dovp=True)
    im.makeimage (type='pb', image=pb, compleximage='', verbose=False)
    im.setvp (dovp=False, verbose=False)
    im.setoptions (ftmachine=cfg.ftmachine, wprojplanes=cfg.wprojplanes,
                   freqinterp='linear', padding=1.2, pastep=360.0, pblimit=cfg.minpb,
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
        im.make (mask)
        ia.open (mask)
        maskcs = ia.coordsys ()
        maskcs.setreferencecode (specframe, 'spectral', True)
        ia.setcoordsys (maskcs.torecord ())

        if cfg.mask is not None:
            rg = util.tools.regionmanager ()
            regions = rg.fromtextfile (filename=cfg.mask,
                                       shape=ia.shape (),
                                       csys=maskcs.torecord ())
            im.regiontoimagemask (mask=mask, region=regions)

        ia.close ()

    # Create blank model(s). Diverging from task_clean even more
    # significantly than usual here.

    for model in models:
        im.make (model)

    # Go!

    im.clean (algorithm='msmfs', niter=cfg.niter, gain=cfg.gain,
              threshold=qa.quantity (cfg.threshold, 'mJy'),
              model=models, residual=resids, image=restoreds,
              psfimage=psfs, mask=maskstr, interactive=False)
    im.close ()


mfsclean_cli = makekwcli (mfsclean_doc, MfscleanConfig, mfsclean)


# plotants

plotants_doc = \
"""
casatask plotants <MS>

Plot the physical layout of the antennas described in the MS.
"""

def plotants (vis):
    import pylab as pl
    mp = util.tools.msplot ()

    mp.open (vis)
    pl.ion ()
    pl.clf ()
    mp.plotoptions (plotsymbol='ro')
    mp.plot (type='array')
    mp.reset ()
    pl.axis ('equal')
    pl.axis ('scaled')
    pl.title ('Antenna configuration stored in ' + vis)


def plotants_cli (argv):
    import os, pylab as pl

    check_usage (plotants_doc, argv, usageifnoargs=True)

    if len (argv) != 2:
        wrong_usage (plotants_doc, 'expect exactly one argument, the MS path')

    util.logger ('warn')
    plotants (argv[1])
    pl.show ()


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

    loglevel = 'warn'


def plotcal (cfg):
    cp = util.tools.calplot ()

    cp.open (b(cfg.caltable))
    cp.selectcal (antenna=b(cfg.antenna), field=b(cfg.field),
                  poln=b(cfg.poln.upper ()), spw=b(cfg.spw), time=b(cfg.timerange))
    cp.plotoptions (iteration=b(cfg.iteration), plotrange=[0.0]*4,
                    plotsymbol=b(cfg.plotsymbol),
                    plotcolor=b(cfg.plotcolor),
                    markersize=b(cfg.markersize),
                    fontsize=b(cfg.fontsize))
    cp.plot (b(cfg.xaxis.upper ()), b(cfg.yaxis.upper ()))

    if cfg.figfile is not None:
        cp.savefig (b(cfg.figfile))


def plotcal_multipage_pdf (cfg, pdfpath, **kwargs):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    cp = util.tools.calplot ()
    cp.open (cfg.caltable)
    cp.selectcal (antenna=cfg.antenna, field=cfg.field,
                  poln=cfg.poln.upper (), spw=cfg.spw, time=cfg.timerange)
    cp.plotoptions (iteration=cfg.iteration, plotrange=[0.0]*4,
                    plotsymbol=cfg.plotsymbol,
                    plotcolor=cfg.plotcolor,
                    markersize=cfg.markersize,
                    fontsize=cfg.fontsize)

    pdf = PdfPages (pdfpath)
    try:
        cp.plot (cfg.xaxis.upper (), cfg.yaxis.upper ())
        pdf.savefig (**kwargs)

        while cp.next ():
            pdf.savefig (**kwargs)

        plt.close ()
    finally:
        pdf.close ()


def plotcal_cli (argv):
    import os, pylab as pl
    check_usage (plotcal_doc, argv, usageifnoargs=True)

    # I can't find where in the source this shows up (maybe removed by 4.0?)
    # but if I turn off the check, things seem to still work without the
    # annoying warnings.
    os.environ['CASA_NOLUSTRE_CHECK'] = '1'

    cfg = PlotcalConfig ().parse (argv[1:])
    util.logger (cfg.loglevel)
    plotcal (cfg)
    pl.show ()


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
    import os.path
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
    im.open (cfg.vis, usescratch=False) # don't think you'll ever want True?
    im.setjy (**kws)
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
    import os.path, tempfile, shutil

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
    ms.open (cfg.vis)

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
    # involved data loss.
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
            ms.split (**kws)
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
    from . import spwglue
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

    ms.open (cfg.vis, nomodify=False)
    ms.msselect (extractmsselect (cfg, havearray=True,
                                  intenttoscanintent=True,
                                  taqltomsselect=False))
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

    for name in sorted (globals ().iterkeys ()):
        if name.endswith ('_cli'):
            print (name[:-4], file=stream)

    raise SystemExit (exitcode)


def commandline (argv=None):
    import sys

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
