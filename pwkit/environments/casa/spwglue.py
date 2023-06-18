# -*- mode: python; coding: utf-8 -*-
# Copyright 2013-2015, 2017 Peter Williams <peter@newton.cx> and collaborators
# Licensed under the MIT License

"""pwkit.environments.casa.spwglue - merge spectral windows in a MeasurementSet

I find that merging windows in this way offers a lot of advantages. This
procesing step is very slow, however.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = str("Progress Config spwglue spwglue_cli").split()

import numpy as np
from ...kwargv import ParseKeywords, Custom
from ...cli import check_usage, die
from . import util
from .util import sanitize_unicode as b


def _sfmt(t):
    # Format timespan (t in seconds)
    if t < 90:
        return "%.0fs" % t
    if t < 4000:
        return "%.1fm" % (t / 60)
    if t < 90000:
        return "%.1fh" % (t / 3600)
    return "%.1fd" % (t / 86400)


class Progress(object):
    """This could be split out; it's useful."""

    elapsed = None

    prefix = None
    totitems = None
    tstart = None
    unbufout = None
    tlastprint = None

    def __enter__(self):
        return self

    def __exit__(self, etype, eval, etb):
        if self.tstart is not None:
            self.finish(etype is None)
        return False

    def start(self, totitems, prefix=""):
        import time, os

        self.totitems = totitems
        self.prefix = prefix
        self.tstart = time.time()
        self.tlastprint = 0
        self.unbufout = os.fdopen(os.dup(1), "wb", 0)

    def progress(self, curitems):
        import time

        now = time.time()
        if now - self.tlastprint < 1:
            return

        elapsed = now - self.tstart
        li = len(str(self.totitems))

        if curitems == 0:
            msg = "%5.1f%% (%*d/%d) elapsed %s ETA %s total %s" % (
                0.0,
                li,
                0,
                self.totitems,
                _sfmt(elapsed),
                "?",
                "?",
            )
        else:
            pct = 100.0 * curitems / self.totitems
            total = 1.0 * elapsed * self.totitems / curitems
            eta = total - elapsed
            msg = "%5.1f%% (%*d/%d) elapsed %s ETA %s total %s" % (
                pct,
                li,
                curitems,
                self.totitems,
                _sfmt(elapsed),
                _sfmt(eta),
                _sfmt(total),
            )

        full_msg = "%s %s\r" % (self.prefix, msg.ljust(60))
        self.unbufout.write(full_msg.encode("utf8"))
        self.tlastprint = now

    def finish(self, success=True):
        import time

        now = time.time()
        self.elapsed = now - self.tstart

        if not success:
            print(file=self.unbufout)
        else:
            msg = "100.0%% (%d/%d) elapsed %s ETA 0s total %s" % (
                self.totitems,
                self.totitems,
                _sfmt(self.elapsed),
                _sfmt(self.elapsed),
            )
            full_msg = "%s %s\n" % (self.prefix, msg.ljust(60))
            self.unbufout.write(full_msg.encode("utf8"))

        self.unbufout.close()
        self.tstart = self.unbufout = None
        self.tlastprint = self.totitems = None


spwglue_doc = """
casatask spwglue vis=MS out=MS mapping=SPWi-SPWj[,SPWl-SPWm,...]

vis=
  The input MS.

out=
  The output MS. Will be created. If the value contains a comma, it is treated
  as a list of MSes, and the "field" keyword should be provided and contain
  an equal number of values. The single input MS will be split into different
  output MSes for each field.

mapping=
  A comma-separated list of spectral window mappings. Each entry takes the
  form <j>-<k>, where <j> and <k> are integers, k >= j. The input spectral
  windows in this range (inclusive, unlike Python) are glued into a single
  output spectral window, with the order matching the order of the keyword.
  For most wideband VLA datasets you want
    mapping=0-7,8-15

field=
  Optional field ID number. If specified, the new MS will contain data
  for only this field. If the value contains a comma, it is treated as a list
  of fields, and the "out" keyword should be a list of MS names.

hackfield=
  Another optional field ID number, used for fixing up MSes where one
  pointing is accidentally assigned two field IDs. Both "field" and
  "hackfield" are extracted from the dataset, but the output is changed
  so that all the records refer to "field" only. Disallowed if "field" is
  a list of fields.

meanbp=
  Path to a saved numpy array giving the mean amplitude bandpass of the
  *glued* SPWs. The data are divided by the square of this array, thereby
  taking out this effect. (Each visibility is affected by the bandpasses of
  the two antennas contributing to its baseline, which is why we square the
  bandpass solution.) This makes it easier to run automated RFI flaggers on
  the data without losing excessive numbers of edge channels.

corr_to_main=false
  If true, move the CORRECTED_DATA column to the main DATA column while gluing.

loglevel=
  Logging detail level. Default is info. Options are
    severe warn info info1 info2 info3 info4 info5 debug1 debug2 debugging
"""


class Config(ParseKeywords):
    vis = Custom(str, required=True)
    out = Custom([str], required=True)

    @Custom([str], required=True)
    def mapping(bits):
        def parse(item):
            t1, t2 = list(map(int, item.split("-")))
            assert t2 >= t1, "can't do reverse-order mappings"
            return list(range(t1, t2 + 1))

        try:
            return [parse(b) for b in bits]
        except Exception as e:
            die('unparseable or illegal "mapping" value "%s": %s', ",".join(bits), e)

    field = [int]
    hackfield = int
    meanbp = str
    corr_to_main = False

    loglevel = "info"  # XXXXXX


# Different things that we do with columns in the SPECTRAL_WINDOW table:
# match  - scalar values that should all agree
# first  - scalar values and we'll keep the first
# scsum  - scalar values and we'll keep the sum
# concat - vectors that we'll concatenate
# empty  - column should be empty (ndim = -1)
_spw_match_cols = frozenset(
    "MEAS_FREQ_REF FLAG_ROW FREQ_GROUP FREQ_GROUP_NAME "
    "IF_CONV_CHAIN NET_SIDEBAND BBC_NO".split()
)
_spw_first_cols = frozenset("DOPPLER_ID REF_FREQUENCY NAME".split())
_spw_scsum_cols = frozenset("NUM_CHAN TOTAL_BANDWIDTH".split())
_spw_concat_cols = frozenset("CHAN_FREQ CHAN_WIDTH EFFECTIVE_BW RESOLUTION".split())
_spw_empty_cols = frozenset("ASSOC_SPW_ID ASSOC_NATURE".split())


def _spwproc_match(spwtb, colname, inspws, outdata):
    theval = None

    for inspw in inspws:
        v = b(spwtb.getcell(b(colname), inspw))
        if theval is None:
            theval = v
        elif theval != v:
            die("glued spws should all have same value of %s column but don't", colname)

    outdata[colname] = theval


def _spwproc_first(spwtb, colname, inspws, outdata):
    outdata[colname] = b(spwtb.getcell(b(colname), inspws[0]))


def _spwproc_scsum(spwtb, colname, inspws, outdata):
    sum = 0
    for inspw in inspws:
        sum += spwtb.getcell(b(colname), inspw)
    outdata[colname] = sum


def _spwproc_concat(spwtb, colname, inspws, outdata):
    clist = []
    for inspw in inspws:
        clist += b(list(spwtb.getcell(b(colname), inspw)))
    outdata[colname] = clist


# Similar info for the main visibility data:
# ident   - define a unique visibility dump
# smatch  - scalars that should match precisely
# sapprox - scalars that should match approximately (to 1e-5) [1]
# vmatch  - vectors that should match precisely
# or      - values that should be boolean-OR'ed together
# pconcat - non-data values that should be concatted across windows
# data    - main data columns
# empty   - columns that should be blank
#
# [1] This feature implemented since EVLA dataset
# 11A-266.sb4865287.eb4875705.55772.08031621527.ms has 22 rows out of ~9
# million that have an EXPOSURE value that differs from the others by 1 part
# in ~10^9.

_vis_ident_cols = "ARRAY_ID FIELD_ID TIME ANTENNA1 ANTENNA2".split()
_vis_smatch_cols = frozenset(
    "FEED1 FEED2 OBSERVATION_ID PROCESSOR_ID SCAN_NUMBER STATE_ID".split()
)
_vis_sapprox_cols = frozenset("EXPOSURE INTERVAL TIME_CENTROID".split())
_vis_vmatch_cols = frozenset("UVW WEIGHT SIGMA".split())
_vis_or_cols = frozenset("FLAG_ROW".split())
_vis_pconcat_cols = frozenset(
    "FLAG DATA MODEL_DATA CORRECTED_DATA WEIGHT_SPECTRUM".split()
)
_vis_data_cols = frozenset("DATA MODEL_DATA CORRECTED_DATA".split())
_vis_empty_cols = frozenset("FLAG_CATEGORY".split())
_vis_pconcat_dtypes = {
    "FLAG": bool,
    "DATA": np.complex128,
    "MODEL_DATA": np.complex128,
    "CORRECTED_DATA": np.complex128,
    "WEIGHT_SPECTRUM": np.float64,
}

_np_converters = {
    # np.bool_: bool,
    # np.int32: int,
    # np.float64: float,
    np.ndarray: lambda x: x,
    np.bool_: np.int32,
    np.int32: lambda x: x,
    np.float64: lambda x: x,
    int: np.int32,
}


def spwglue(cfg):
    nout = len(cfg.out)
    nfield = len(cfg.field)

    if (nout != 1 or nfield != 0) and (nout != nfield):
        die("%d outputs requested, but only %d fields listed", nout, nfield)
    if nout != 1 and cfg.hackfield is not None:
        die('"hackfield" keyword is not compatible with multiple output sets')

    if nfield == 0:
        cfg.field = [None]

    with Progress() as p:
        for idx, (out, field) in enumerate(zip(cfg.out, cfg.field)):
            _spwglue(cfg, p, out, field, nfield, idx)


def fillsmall(path, rows):
    tb = util.tools.table()
    tb.open(b(path), nomodify=False)
    tb.addrows(len(rows))

    try:
        for i, data in enumerate(rows):
            for col, val in data.items():
                tb.putcell(b(col), i, val)
    except Exception as e:
        raise Exception(
            "error putting: %d %s %s (%s): %s" % (i, col, val, val.__class__, e)
        )

    tb.close()


def _spwglue(cfg, prog, thisout, thisfield, nfields, fieldidx):
    import os.path

    if cfg.hackfield is not None:
        assert thisfield is not None

    nout = len(cfg.mapping)

    if cfg.meanbp is None:
        invsqmeanbp = None
    else:
        try:
            invsqmeanbp = np.load(cfg.meanbp)
        except Exception as e:
            die(
                'couldn\'t open bandpass file "%s": %s (%s)',
                cfg.meanbp,
                e,
                e.__class__.__name__,
            )

        if np.any(invsqmeanbp <= 0):
            die('illegal bandpass file "%s": some values are nonpositive', cfg.meanbp)

        invsqmeanbp **= -2

    # Read in spw and dd info and make sure everything is in order.

    tb = util.tools.table()
    tb.open(b(os.path.join(cfg.vis, "SPECTRAL_WINDOW")))
    spwcols = tb.colnames()
    spwout = [dict() for i in range(nout)]

    for col in spwcols:
        if col in _spw_match_cols:
            for i in range(nout):
                _spwproc_match(tb, col, cfg.mapping[i], spwout[i])
        elif col in _spw_first_cols:
            for i in range(nout):
                _spwproc_first(tb, col, cfg.mapping[i], spwout[i])
        elif col in _spw_scsum_cols:
            for i in range(nout):
                _spwproc_scsum(tb, col, cfg.mapping[i], spwout[i])
        elif col in _spw_concat_cols:
            for i in range(nout):
                _spwproc_concat(tb, col, cfg.mapping[i], spwout[i])
        elif col in _spw_empty_cols:
            pass
        else:
            die(
                'don\'t know how to handle SPECTRAL_WINDOW column "%s" in %s"',
                col,
                cfg.vis,
            )

    numchans = tb.getcol(b"NUM_CHAN")
    tb.close()
    inspw2outspw = {}

    for i in range(nout):
        ofs = 0

        for inspw in cfg.mapping[i]:
            if inspw in inspw2outspw:
                die("spw %d gets duplicated in mapping", inspw)
            inspw2outspw[inspw] = (i, ofs)
            ofs += numchans[inspw]

    # We only handle 1:1 mappings between DD and SPW.
    tb.open(b(os.path.join(cfg.vis, "DATA_DESCRIPTION")))
    ddflagrows = np.asarray(tb.getcol(b"FLAG_ROW"))
    ddpids = np.asarray(tb.getcol(b"POLARIZATION_ID"))
    ddspws = np.asarray(tb.getcol(b"SPECTRAL_WINDOW_ID"))
    tb.close()
    indd2outdd = {}
    ddout = [{"SPECTRAL_WINDOW_ID": np.int32(i)} for i in range(nout)]

    for i in range(ddspws.size):
        inspw = ddspws[i]

        if inspw not in inspw2outspw:
            continue  # this dd gets dropped in the processing

        outspw, outofs = inspw2outspw[inspw]
        indd2outdd[i] = (outspw, outofs)

        fr = ddout[outspw].get("FLAG_ROW")
        if fr is None:
            ddout[outspw]["FLAG_ROW"] = np.int32(
                ddflagrows[i]
            )  # hack: have to convert bool->int for some reason
        elif ddflagrows[i] != fr:
            die(
                "dd %d glued into output spw %d has different FLAG_ROW ident", i, outspw
            )

        pid = ddout[outspw].get("POLARIZATION_ID")
        if pid is None:
            ddout[outspw]["POLARIZATION_ID"] = ddpids[i]
        elif ddpids[i] != pid:
            die(
                "dd %d glued into output spw %d has different POLARIZATION ident",
                i,
                outspw,
            )

    # We don't actually do any sanity checking with the SOURCE table, but we
    # do need to modify it. Let's just load it up here while we're loading up
    # all of the other small tables.

    tb = util.tools.table()
    tb.open(b(os.path.join(cfg.vis, "SOURCE")))
    srccols = [
        c for c in tb.colnames() if tb.iscelldefined(c, 0)
    ]  # at least POSITION is undefined
    srckeys = set()
    srcout = []

    for i in range(tb.nrows()):
        data = dict((c, b(tb.getcell(b(c), i))) for c in srccols)
        m = inspw2outspw.get(data["SPECTRAL_WINDOW_ID"])
        if m is None:
            continue  # this spw is being dropped

        data["SPECTRAL_WINDOW_ID"] = np.int32(m[0])
        key = (data["SOURCE_ID"], data["TIME"], data["INTERVAL"], m[0])

        if key in srckeys:
            # XXX: should verify that rest of data are identical, but too lazy.
            continue

        srcout.append(data)
        srckeys.add(key)

    # Need this for buffer prealloc

    tb.open(b(os.path.join(cfg.vis, "POLARIZATION")))
    numcorrs = tb.getcol(b"NUM_CORR")
    tb.close()

    # We've done all the preparation we can. Time to start copying. tb.copy()
    # will clobber existing tables, so do our best to ensure that there isn't
    # anything preexisting so far. Of course someone could put a table in the
    # destination between this mkdir and the actual copy.

    os.mkdir(thisout)

    # Copy the overall table structure without contents.

    tb.open(b(cfg.vis))
    tb.copy(
        newtablename=b(thisout), deep=True, valuecopy=True, norows=True
    ).close()  # copy() returns the new table.
    tb.close()

    for item in os.listdir(cfg.vis):
        if not os.path.exists(os.path.join(cfg.vis, item, "table.dat")):
            continue
        if item in "SPECTRAL_WINDOW DATA_DESCRIPTION SOURCE".split():
            continue  # will handle these manually
        if item == "SYSPOWER":
            continue  # XXX; large and unused
        if item == "SORTED_TABLE":
            continue  # XXX; copies main data rows; not sure what to do about it

        tb.open(b(os.path.join(cfg.vis, item)))
        tb.copy(
            b(os.path.join(thisout, item)), deep=False, valuecopy=True, norows=False
        ).close()
        tb.close()

    # The rewritten small tables.

    fillsmall(os.path.join(thisout, "SPECTRAL_WINDOW"), spwout)
    fillsmall(os.path.join(thisout, "DATA_DESCRIPTION"), ddout)
    fillsmall(os.path.join(thisout, "SOURCE"), srcout)

    # Rewriting the main table.

    tb.open(b(cfg.vis))
    colnames = [c for c in tb.colnames() if c not in _vis_empty_cols]
    nchunk = 1024

    dt = util.tools.table()
    dt.open(b(thisout), nomodify=False)

    if cfg.corr_to_main:
        dt.removecols([b"CORRECTED_DATA"])

    outrow = 0

    for outspw in range(nout):
        q = " or ".join(
            "DATA_DESC_ID == %d" % t[0] for t in indd2outdd.items() if t[1][0] == outspw
        )
        if cfg.hackfield is not None:
            q = "(FIELD_ID == %s or FIELD_ID == %s) and (%s)" % (
                thisfield,
                cfg.hackfield,
                q,
            )
        elif thisfield is not None:
            q = "FIELD_ID == %s and (%s)" % (thisfield, q)
        if cfg.hackfield is not None:
            sortlist = "ARRAY_ID, TIME, ANTENNA1, ANTENNA2, DATA_DESC_ID"
        else:
            sortlist = "ARRAY_ID, FIELD_ID, TIME, ANTENNA1, ANTENNA2, DATA_DESC_ID"
        st = tb.query(b(q), sortlist=b(sortlist))
        inrow = 0
        nq = st.nrows()

        ncorr = numcorrs[ddout[outspw]["POLARIZATION_ID"]]
        nchan = spwout[outspw]["NUM_CHAN"]
        curident = None
        curout = {}

        if invsqmeanbp is not None and nchan != invsqmeanbp.size:
            die(
                "need to apply bandpass of %d channels, but output spw %d has "
                "%d channels",
                invsqmeanbp.size,
                outspw,
                nchan,
            )

        for col in colnames:
            if col in _vis_pconcat_cols:
                curout[col] = np.zeros((ncorr, nchan), dtype=_vis_pconcat_dtypes[col])
            else:
                curout[col] = None

        def dump():
            # increment outrow after calling.
            dt.addrows(1)
            for col in colnames:
                v = curout[col]
                v = _np_converters[v.__class__](v)

                if invsqmeanbp is not None and col in _vis_data_cols:
                    v *= invsqmeanbp

                if not cfg.corr_to_main:
                    dt.putcell(b(col), outrow, v)
                elif col == "DATA":
                    pass  # ignore; will be overwritten by CORRECTED_DATA
                else:
                    if col == "CORRECTED_DATA":
                        outcol = "DATA"
                    else:
                        outcol = col
                    dt.putcell(b(outcol), outrow, v)

                if isinstance(v, np.ndarray):
                    v.fill(0)
                else:
                    curout[col] = None

        prog.start(nq, "%d/%d" % (nout * fieldidx + outspw + 1, nout * nfields))

        while inrow < nq:
            prog.progress(inrow)

            vdata = {}
            for col in colnames:
                vdata[col] = st.getcol(col, inrow, nchunk)

            nread = vdata["TIME"].size

            if cfg.hackfield is not None:
                vdata["FIELD_ID"].fill(thisfield)

            for i in range(nread):
                outspw, outofs = indd2outdd[vdata["DATA_DESC_ID"][i]]
                newident = tuple(vdata[c][i] for c in _vis_ident_cols)

                if newident != curident:
                    # Starting a new record
                    if curident is not None:
                        dump()
                        outrow += 1
                    curident = newident

                    for col in colnames:
                        if (
                            col in _vis_ident_cols
                            or col in _vis_smatch_cols
                            or col in _vis_sapprox_cols
                            or col in _vis_or_cols
                        ):
                            curout[col] = vdata[col][i]
                        elif col in _vis_vmatch_cols:
                            curout[col] = vdata[col][..., i]
                        elif col in _vis_pconcat_cols:
                            d = vdata[col][..., i]
                            curout[col][:, outofs : outofs + d.shape[1]] = d
                        elif col == "DATA_DESC_ID":
                            curout["DATA_DESC_ID"] = outspw
                        else:
                            die("unhandled vis column %s", col)
                else:
                    # Continuing an existing record
                    for col in colnames:
                        if (
                            col in _vis_ident_cols
                            or col == "DATA_DESC_ID"
                            or col == "WEIGHT"
                            or col == "SIGMA"
                        ):
                            pass
                        elif col in _vis_smatch_cols:
                            if vdata[col][i] != curout[col]:
                                die(
                                    "changing value for column %s within a glued record",
                                    col,
                                )
                        elif col in _vis_sapprox_cols:
                            if abs((vdata[col][i] - curout[col]) / curout[col]) > 1e-5:
                                die(
                                    "excessively changing value for column %s within a glued record",
                                    col,
                                )
                        elif col in _vis_vmatch_cols:
                            if not np.all(vdata[col][..., i] == curout[col]):
                                die(
                                    "changing value for column %s within a glued record: %r , %r",
                                    col,
                                    vdata[col][..., i],
                                    curout[col],
                                )
                        elif col in _vis_or_cols:
                            curout[col] |= vdata[col][i]
                        elif col in _vis_pconcat_cols:
                            d = vdata[col][..., i]
                            curout[col][:, outofs : outofs + d.shape[1]] = d
                        else:
                            die("unhandled vis column %s", col)

            inrow += nread

        if curout["TIME"] is not None:
            dump()  # finish this last record.
            outrow += 1

        prog.finish()

    dt.close()
    tb.close()


def spwglue_cli(argv):
    check_usage(spwglue_doc, argv, usageifnoargs=True)
    cfg = Config().parse(argv[1:])
    util.logger(cfg.loglevel)
    spwglue(cfg)
