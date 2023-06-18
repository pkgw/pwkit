# -*- mode: python; coding: utf-8 -*-
# Copyright 2013-2016 Peter Williams <peter@newton.cx> and collaborators.
# Licensed under the MIT License.

"""Insert a simple polarized point source model into a dataset.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = str("Config polmodel polmodel_cli").split()

import numpy as np, tempfile

from ...astutil import A2R, D2R, sphdist
from ...cli import check_usage, die
from ...io import Path
from ...kwargv import ParseKeywords, Custom
from . import util
from .util import sanitize_unicode as b

polmodel_doc = """
casatask polmodel vis=<MS> field=<field specification>

Insert polarization information for a model into a Measurement Set. Uses a
built-in table of polarization properties to generate Stokes QUV information
from CASA's built-in Stokes I models.

The only currently supported source is 3C286 in C band.

"""


class Config(ParseKeywords):
    vis = Custom(str, required=True)
    field = Custom(str, required=True)


class PolSource(object):
    name = None  # fed to imager.predictcomp(objname)
    ra = None  # rad
    dec = None  # rad
    models = None

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class PolModel(object):
    name = None
    fmin = None  # GHz
    fmax = None  # GHz
    polfrac = None  # [0,1]
    polpa = None  # degr

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def getquv(self, i):
        # In the future, we might have different models that derive QUV
        # from I in a different way. Probably won't, though.

        a = self.polpa * D2R
        p = i * self.polfrac
        return p * np.cos(a), p * np.sin(a), 0.0


position_tolerance = 1 * A2R
fluxscale_standard = "Perley-Butler 2010"

sources = [
    PolSource(
        name="3C286",
        ra=-2.74392753,
        dec=0.53248521,
        models=[PolModel(name="C", fmin=3.9, fmax=8.1, polfrac=0.112, polpa=66.0)],
    )
]


def polmodel(cfg):
    ms = util.tools.ms()
    tb = util.tools.table()
    im = util.tools.imager()
    cl = util.tools.componentlist()
    vis = Path(cfg.vis)

    # Set up MS selection so we know what data we actually care about.

    ms.open(b(cfg.vis))
    ms.msselect(b(dict(field=cfg.field)))
    rangeinfo = ms.range(b"data_desc_id field_id".split())
    ddids = rangeinfo["data_desc_id"]
    fields = rangeinfo["field_id"]

    # Check that we know the field and pull up its model

    if fields.size != 1:
        die("selection should pick exactly one field, but got %d", fields.size)

    tb.open(b(vis / "FIELD"))
    refdir = tb.getcell(b"REFERENCE_DIR", fields[0])
    tb.close()

    if refdir.shape[1] != 1:
        die(
            "selected field %s has a time-variable reference direction, which I can't handle",
            cfg.field,
        )

    ra, dec = refdir[:, 0]

    for source in sources:
        if sphdist(dec, ra, source.dec, source.ra) < position_tolerance:
            break
    else:
        die("found no match in my data table for field %s", cfg.field)

    # Now we can get the spws and check that we have a model for them.

    tb.open(b(vis / "DATA_DESCRIPTION"))
    ddspws = tb.getcol(b"SPECTRAL_WINDOW_ID")
    tb.close()

    spws = list(set(ddspws[ddid] for ddid in ddids))

    freqranges = {}
    models = {}
    allfreqs = []
    tb.open(b(vis / "SPECTRAL_WINDOW"))

    for spw in spws:
        freqs = tb.getcell(b"CHAN_FREQ", spw)
        freqranges[spw] = (freqs[0], freqs[-1])
        allfreqs += [freqs[0], freqs[-1]]

        for model in source.models:
            if freqs[0] >= model.fmin * 1e9 and freqs[-1] <= model.fmax * 1e9:
                models[spw] = model
                break
        else:
            die(
                "spw %d is out of frequency bounds for all of my models of "
                "field %s (%s): spw range is (%f, %f) GHz",
                spw,
                cfg.field,
                source.name,
                freqs[0] * 1e-9,
                freqs[-1] * 1e-9,
            )

    tb.close()

    # Now it's worth using predictcomp() to get the Stokes I fluxes.

    workdir = tempfile.mkdtemp(prefix="mspolmodel")
    try:
        cp = im.predictcomp(
            objname=b(source.name),
            standard=b(fluxscale_standard),
            freqs=allfreqs,
            pfx=b(workdir + "/"),
        )
        cl.open(cp)
        if cl.length() != 1:
            die(
                "expected one component in predicted list; got %d (%s)", cl.length(), cp
            )
        stokesi = cl.getcomponent(0)["spectrum"]["ival"]
        # log=False: we'll have to run the risk that the user won't be aware that
        # we closed the component list structure. Scary.
        cl.close(log=False)
    finally:
        Path(workdir).rmtree()

    # And now we have everything we need. Invoke setjy() a bunch.

    im.open(b(vis), usescratch=False)

    for i, spw in enumerate(spws):
        model = models[spw]
        f1, f2 = freqranges[spw]
        i1, i2 = stokesi[i * 2 : i * 2 + 2]

        spindex = np.log(i2 / i1) / np.log(f2 / f1)
        q, u, v = model.getquv(i1)
        reffreq = "%.3fMHz" % (f1 * 1e-6)

        # print ('%2d/%d: %d %.3f-%.3f %.3f-%.3f [%.3f %.3f %.3f %3f] %.3f %s' \
        #      % (i + 1, len (spws), spw, f1*1e-9, f2*1e-9, i1, i2,
        #         i1, q, u, v, spindex, reffreq))
        im.setjy(
            field=b(cfg.field),
            spw=b(str(spw)),
            modimage=b"",
            fluxdensity=[i1, q, u, v],
            spix=spindex,
            standard=b(fluxscale_standard),
            scalebychan=True,
            reffreq=b(reffreq),
        )

    im.close()


def polmodel_cli(argv):
    check_usage(polmodel_doc, argv, usageifnoargs=True)
    cfg = Config().parse(argv[1:])
    polmodel(cfg)
