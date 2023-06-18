# -*- mode: python; coding: utf-8 -*-
# Copyright 2014 Peter Williams <peter@newton.cx> and collaborators.
# Licensed under the MIT License.

"""pwkit.cli.imtool - the 'imtool' program.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = str("commandline").split()

import numpy as np, sys

from . import multitool
from . import *
from .. import astimage
from ..io import Path


def load_ndshow():
    try:
        from .. import ndshow_gtk3 as ndshow
    except ImportError as e:
        die("cannot load graphics backend for viewing images: %s", e)

    return ndshow


# The commands.


class BlinkCommand(multitool.Command):
    name = "blink"
    argspec = "[-f] <images...>"
    summary = "Blink zero or more images."
    more_help = """
-f  - Show the 2D FFT of the images

WCS support isn't fantastic and sometimes causes crashes.

If an input image has multiple planes, they will be showed separately.
"""

    def _load(self, path, fft, maxnorm):
        try:
            img = astimage.open(path, "r", eat_warnings=True)
        except Exception as e:
            die("can't open path “%s”: %s", path, e)

        def getplanes():
            try:
                simg = img.simple()
            except Exception as e:
                # Multi-dimensional image -- iterate over outer planes
                fulldata = img.read(flip=True)

                for index in np.ndindex(*img.shape[:-2]):
                    yield fulldata[index], None, " " + repr(index)
            else:
                yield simg.read(flip=True), img.toworld, ""

        for data, toworld, desc in getplanes():
            if fft:
                from numpy.fft import ifftshift, fft2, fftshift

                data = np.abs(ifftshift(fft2(fftshift(data.filled(0)))))
                data = np.ma.MaskedArray(data)
                toworld = None

            if maxnorm:
                data /= np.ma.max(data)

            yield data, toworld, desc

    def invoke(self, args, **kwargs):
        fft = pop_option("f", args)
        maxnorm = pop_option("m", args)
        ndshow = load_ndshow()

        images = []
        toworlds = []
        names = []

        for path in args:
            for image, toworld, desc in self._load(path, fft, maxnorm):
                images.append(image)
                toworlds.append(toworld)
                names.append(path + desc)

        if not len(images):
            return

        shape = images[0].shape
        for i, im in enumerate(images[1:]):
            if im.shape != shape:
                die(
                    "shape of “%s” (%s) does not agree with that " "of “%s” (%s)",
                    args[i + 1],
                    "×".join(map(str, im.shape)),
                    args[0],
                    "×".join(map(str, shape)),
                )

        # Merge masks. This is more complicated than you might think since you
        # can't "or" nomask with itself.

        jointmask = np.ma.nomask

        for i in range(len(images)):
            if jointmask is np.ma.nomask:
                if images[i].mask is not np.ma.nomask:
                    jointmask = images[i].mask
            else:
                np.logical_or(jointmask, images[i].mask, jointmask)

        for im in images:
            im.mask = jointmask

        ndshow.cycle(images, names, toworlds=toworlds, yflip=True)


class FitsrcCommand(multitool.ArgparsingCommand):
    name = "fitsrc"
    summary = "Fit a compact-source model to a location in an image."

    def get_arg_parser(self, **kwargs):
        ap = super(FitsrcCommand, self).get_arg_parser(**kwargs)
        ap.add_argument(
            "--forcepoint",
            "-p",
            action="store_true",
            help="Force the use of a point-source model.",
        )
        ap.add_argument(
            "--display",
            "-d",
            action="store_true",
            help="Display the fit results graphically.",
        )
        ap.add_argument(
            "--format",
            metavar="FORMAT",
            default="shell",
            help='Output format for results: "shell", "yaml".',
        )
        ap.add_argument("--dest", metavar="PATH", help="Path in which to save results.")
        ap.add_argument(
            "--section",
            metavar="LOCATION.TO.SECTION",
            help="Save results in the named dot-separated section of the output file.",
        )
        ap.add_argument("image", metavar="IMAGE-PATH", help="The path to an image.")
        ap.add_argument(
            "x",
            metavar="X-PIXEL",
            type=int,
            help="The X pixel coordinate near which to search for a source.",
        )
        ap.add_argument(
            "y",
            metavar="Y-PIXEL",
            type=int,
            help="The Y pixel coordinate near which to search for a source.",
        )
        return ap

    def invoke(self, args, **kwargs):
        from ..immodel import fit_one_source

        # TODO: would be nice to have a consistent API for outputting
        # structured data in a variety of formats, preferably supporting
        # streaming.

        close_me = None

        if args.format == "shell":
            if args.dest is not None:
                dest = close_me = open(args.dest, "wt")
            else:
                import sys

                dest = sys.stdout

            def report_func(key, fmt, value):
                print(key, "=", fmt % value, sep="", file=dest)

        elif args.format == "yaml":
            from collections import OrderedDict

            try:
                from ruamel import yaml as ruamel_yaml
            except ImportError:
                import ruamel_yaml  # how Conda packages it, grr ...

            if args.dest is None:
                toplevel = None
            else:
                with Path(args.dest).try_open("rt") as f:
                    toplevel = ruamel_yaml.load(f, ruamel_yaml.RoundTripLoader)

            if toplevel is None:
                toplevel = OrderedDict()

            dest_section = toplevel

            if args.section is not None:
                for piece in args.section.split("."):
                    dest_section = dest_section.setdefault(piece, OrderedDict())

            dest_section.clear()

            def report_func(key, fmt, value):
                if isinstance(value, np.number):
                    value = value.item()
                dest_section[key] = value

        else:
            die("unrecognized output format %r", args.format)

        im = astimage.open(args.image, "r", eat_warnings=True).simple()
        fit_one_source(
            im,
            args.x,
            args.y,
            forcepoint=args.forcepoint,
            display=args.display,
            report_func=report_func,
        )

        if close_me is not None:
            close_me.close()

        if args.format == "yaml":
            if args.dest is None:
                import sys

                ruamel_yaml.dump(
                    toplevel, stream=sys.stdout, Dumper=ruamel_yaml.RoundTripDumper
                )
            else:
                with Path(args.dest).open("wt") as f:
                    ruamel_yaml.dump(
                        toplevel, stream=f, Dumper=ruamel_yaml.RoundTripDumper
                    )


class HackdataCommand(multitool.Command):
    name = "hackdata"
    argspec = "<inpath> <outpath>"
    summary = "Blindly copy pixel data from one image to another."

    def invoke(self, args, **kwargs):
        if len(args) != 2:
            raise multitool.UsageError("expect exactly two arguments")

        inpath, outpath = args

        try:
            with astimage.open(inpath, "r", eat_warnings=True) as imin:
                indata = imin.read()
        except Exception as e:
            die('cannot open input "%s": %s', inpath, e)

        try:
            with astimage.open(outpath, "rw") as imout:
                if imout.size != indata.size:
                    die(
                        "cannot import data: input has %d pixels; output has %d",
                        indata.size,
                        imout.size,
                    )

                imout.write(indata)
        except Exception as e:
            die('cannot write to output "%s": %s', outpath, e)


class InfoCommand(multitool.ArgparsingCommand):
    # XXX terrible code duplication while my output format stuff is half-baked.

    name = "info"
    summary = "Print properties of one or more images."

    def get_arg_parser(self, **kwargs):
        ap = super(InfoCommand, self).get_arg_parser(**kwargs)
        ap.add_argument(
            "--format",
            metavar="FORMAT",
            default="pretty",
            help='Output format for results: "pretty", "yaml".',
        )
        ap.add_argument("--dest", metavar="PATH", help="Path in which to save results.")
        ap.add_argument(
            "--section",
            metavar="LOCATION.TO.SECTION",
            help="Save results in the named dot-separated section of the output file.",
        )
        ap.add_argument(
            "images", metavar="IMAGE-PATH", nargs="+", help="The path to an image."
        )
        return ap

    def invoke(self, args, **kwargs):
        if args.format == "pretty":
            self._invoke_pretty_format(args.images)
        elif args.format == "yaml":
            self._invoke_yaml_format(args)
        else:
            die("unrecognized output format %r", args.format)

    def _invoke_pretty_format(self, impaths):
        if len(impaths) == 1:
            self._print(impaths[0])
        else:
            for i, path in enumerate(impaths):
                if i > 0:
                    print()
                print("path     =", path)
                self._print(path)

    def _print(self, path):
        from ..astutil import fmtradec, R2A, R2D

        try:
            im = astimage.open(path, "r", eat_warnings=True)
        except Exception as e:
            die('can\'t open "%s": %s', path, e)

        print("kind     =", im.__class__.__name__)

        latcell = loncell = None

        if im.toworld is not None:
            latax, lonax = im._latax, im._lonax
            delta = 1e-6
            p = 0.5 * (np.asfarray(im.shape) - 1)
            w1 = im.toworld(p)
            p[latax] += delta
            w2 = im.toworld(p)
            latcell = (w2[latax] - w1[latax]) / delta
            p[latax] -= delta
            p[lonax] += delta
            w2 = im.toworld(p)
            loncell = (w2[lonax] - w1[lonax]) / delta * np.cos(w2[latax])

        if im.pclat is not None:
            print("center   =", fmtradec(im.pclon, im.pclat), "# pointing")
        elif im.toworld is not None:
            w = im.toworld(0.5 * (np.asfarray(im.shape) - 1))
            print("center   =", fmtradec(w[lonax], w[latax]), "# lattice")

        if im.shape is not None:
            print("shape    =", " ".join(str(x) for x in im.shape))
            npix = 1
            for x in im.shape:
                npix *= x
            print("npix     =", npix)

        if im.axdescs is not None:
            print("axdescs  =", " ".join(x for x in im.axdescs))

        if im.charfreq is not None:
            print("charfreq = %f GHz" % im.charfreq)

        if im.mjd is not None:
            from time import gmtime, strftime

            posix = 86400.0 * (im.mjd - 40587.0)
            ts = strftime("%Y-%m-%dT%H-%M-%SZ", gmtime(posix))
            print("mjd      = %f # %s" % (im.mjd, ts))

        if latcell is not None:
            print("ctrcell  = %fʺ × %fʺ # lat, lon" % (latcell * R2A, loncell * R2A))

        if im.bmaj is not None:
            print(
                "beam     = %fʺ × %fʺ @ %f°"
                % (im.bmaj * R2A, im.bmin * R2A, im.bpa * R2D)
            )

            if latcell is not None:
                bmrad2 = 2 * np.pi * im.bmaj * im.bmin / (8 * np.log(2))
                cellrad2 = latcell * loncell
                print("ctrbmvol = %f px" % np.abs(bmrad2 / cellrad2))

        if im.units is not None:
            print("units    =", im.units)

    def _invoke_yaml_format(self, args):
        from collections import OrderedDict
        from ..astutil import fmtdeglat, fmthours, R2A, R2D

        if len(args.images) > 1:
            die("can only use YAML output format with one image")

        try:
            im = astimage.open(args.images[0], "r", eat_warnings=True)
        except Exception as e:
            die('can\'t open "%s": %s', args.images[0], e)

        try:
            from ruamel import yaml as ruamel_yaml
        except ImportError:
            import ruamel_yaml  # how Conda packages it, grr ...

        if args.dest is None:
            toplevel = None
        else:
            with Path(args.dest).try_open("rt") as f:
                toplevel = ruamel_yaml.load(f, ruamel_yaml.RoundTripLoader)

        if toplevel is None:
            toplevel = OrderedDict()

        dest_section = toplevel

        if args.section is not None:
            for piece in args.section.split("."):
                dest_section = dest_section.setdefault(piece, OrderedDict())

        def okeys(**kwargs):
            return OrderedDict(sorted(kwargs.items(), key=lambda t: t[0]))

        dest_section.clear()
        dest_section["kind"] = im.__class__.__name__

        latcell = loncell = None

        if im.toworld is not None:
            latax, lonax = im._latax, im._lonax
            delta = 1e-6
            p = 0.5 * (np.asfarray(im.shape) - 1)
            w1 = im.toworld(p)
            p[latax] += delta
            w2 = im.toworld(p)
            latcell = (w2[latax] - w1[latax]) / delta
            p[latax] -= delta
            p[lonax] += delta
            w2 = im.toworld(p)
            loncell = (w2[lonax] - w1[lonax]) / delta * np.cos(w2[latax])

        if im.pclat is not None:
            dest_section["center"] = okeys(
                kind="pointing",
                lat=fmtdeglat(im.pclat),
                lon=fmthours(im.pclon),
            )
        elif im.toworld is not None:
            w = im.toworld(0.5 * (np.asfarray(im.shape) - 1))
            dest_section["center"] = okeys(
                kind="lattice",
                lat=fmtdeglat(w[latax]),
                lon=fmthours(w[lonax]),
            )

        if im.shape is not None:
            dest_section["shape"] = [x.item() for x in im.shape]
            npix = 1
            for x in im.shape:
                npix *= x
            dest_section["npix"] = npix.item()

        if im.axdescs is not None:
            dest_section["axdescs"] = list(im.axdescs)

        if im.charfreq is not None:
            dest_section["charfreq"] = im.charfreq

        if im.mjd is not None:
            dest_section["mjd"] = im.mjd

        if latcell is not None:
            dest_section["ctrcell"] = okeys(
                lat=latcell.item(),
                lon=loncell.item(),
            )

        if im.bmaj is not None:
            dest_section["beam"] = okeys(
                major=im.bmaj,
                minor=im.bmin,
                posang=im.bpa,
            )

            if latcell is not None:
                bmrad2 = 2 * np.pi * im.bmaj * im.bmin / (8 * np.log(2))
                cellrad2 = latcell * loncell
                dest_section["ctrbmvol"] = np.abs(bmrad2 / cellrad2).item()

        if im.units is not None:
            dest_section["units"] = im.units

        if args.dest is None:
            import sys

            ruamel_yaml.dump(
                toplevel, stream=sys.stdout, Dumper=ruamel_yaml.RoundTripDumper
            )
        else:
            with Path(args.dest).open("wt") as f:
                ruamel_yaml.dump(toplevel, stream=f, Dumper=ruamel_yaml.RoundTripDumper)


class SetrectCommand(multitool.Command):
    name = "setrect"
    argspec = "<image> <x> <y> <halfwidth> <value>"
    summary = "Set a rectangle in an image to a constant."

    def invoke(self, args, **kwargs):
        if len(args) != 5:
            raise multitool.UsageError("expected exactly 5 arguments")

        path = args[0]

        try:
            x = int(args[1])
            y = int(args[2])
            halfwidth = int(args[3])
            value = float(args[4])
        except ValueError:
            raise multitool.UsageError("could not parse one of the numeric arguments")

        try:
            img = astimage.open(path, "rw", eat_warnings=True)
        except Exception as e:
            die("can't open path “%s”: %s", path, e)

        data = img.read()
        data[..., y - halfwidth : y + halfwidth, x - halfwidth : x + halfwidth] = value
        img.write(data)


class ShowCommand(multitool.Command):
    name = "show"
    argspec = "[--no-coords] [-f] <image> [images...]"
    summary = "Show images interactively."
    more_help = """--no-coords - Do not show coordinates even if available
-f          - Show the 2D FFT of the image

WCS support isn't fantastic and sometimes causes crashes."""

    def invoke(self, args, **kwargs):
        anyfailures = False
        ndshow = load_ndshow()

        fft = pop_option("f", args)
        no_coords = pop_option("no-coords", args)

        for path in args:
            try:
                img = astimage.open(path, "r", eat_warnings=True)
            except Exception as e:
                print(
                    "imtool show: can't open path “%s”: %s" % (path, e), file=sys.stderr
                )
                anyfailures = True
                continue

            try:
                img = img.simple()
            except Exception as e:
                print(
                    "imtool show: can't convert “%s” to simple 2D sky image; taking "
                    " first plane" % path,
                    file=sys.stderr,
                )
                data = img.read(flip=True)[
                    tuple(np.zeros(img.shape.size - 2, dtype=int))
                ]
                toworld = None
            else:
                data = img.read(flip=True)
                toworld = img.toworld

            if fft:
                from numpy.fft import ifftshift, fft2, fftshift

                data = np.abs(ifftshift(fft2(fftshift(data.filled(0)))))
                data = np.ma.MaskedArray(data)
                toworld = None

            if no_coords:
                toworld = None

            ndshow.view(
                data, title=path + " — Array Viewer", toworld=toworld, yflip=True
            )

        sys.exit(int(anyfailures))


class StatsCommand(multitool.Command):
    name = "stats"
    argspec = "<images...>"
    summary = "Compute and print statistics of a 64×64 patch at image center."

    def _print(self, path):
        try:
            img = astimage.open(path, "r", eat_warnings=True)
        except Exception as e:
            die('error: can\'t open "%s": %s', path, e)

        try:
            img = img.simple()
        except Exception as e:
            print(
                "imstats: can't convert “%s” to simple 2D sky image; "
                "taking first plane" % path,
                file=sys.stderr,
            )
            data = img.read()[tuple(np.zeros(img.shape.size - 2))]
        else:
            data = img.read()

        h, w = data.shape
        patchhalfsize = 32

        p = data[
            h // 2 - patchhalfsize : h // 2 + patchhalfsize,
            w // 2 - patchhalfsize : w // 2 + patchhalfsize,
        ]

        mx = p.max()
        mn = p.min()
        med = np.median(p)
        rms = np.sqrt((p**2).mean())

        sc = max(abs(mx), abs(mn))
        if sc <= 0:
            expt = 0
        else:
            expt = 3 * (int(np.floor(np.log10(sc))) // 3)
        f = 10**-expt

        print("min  = %.2f * 10^%d" % (f * mn, expt))
        print("max  = %.2f * 10^%d" % (f * mx, expt))
        print("med  = %.2f * 10^%d" % (f * med, expt))
        print("rms  = %.2f * 10^%d" % (f * rms, expt))

    def invoke(self, args, **kwargs):
        if len(args) == 1:
            self._print(args[0])
        else:
            for i, path in enumerate(args):
                if i > 0:
                    print()
                print("path =", path)
                self._print(path)


# The driver.

from .multitool import HelpCommand


class Imtool(multitool.Multitool):
    cli_name = "imtool"
    summary = "Perform miscellaneous tasks with astronomical images."


def commandline():
    multitool.invoke_tool(globals())
