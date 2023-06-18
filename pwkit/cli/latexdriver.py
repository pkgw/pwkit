# -*- mode: python; coding: utf-8 -*-
# Copyright 2014-2018 Peter Williams <peter@newton.cx> and collaborators.
# Licensed under the MIT License.

"""pwkit.cli.latexdriver - the 'latexdriver' program.

This used to be a nice little shell script, but for portability it's better to
do this in Python. And now we can optionally provide some BibTeX-related
magic.

"""
from __future__ import absolute_import, division, print_function

__all__ = "commandline".split()

import signal, subprocess, sys

from .. import PKError
from ..io import Path
from . import *


# This batch of code implements the magic BibTeX merging feature.


def cited_names_from_aux_file(stream):
    """Parse a LaTeX ".aux" file and generate a list of names cited according to
    LaTeX ``\\citation`` commands. Repeated names are generated only once. The
    argument should be a opened I/O stream.

    """
    cited = set()

    for line in stream:
        if not line.startswith(r"\citation{"):
            continue

        line = line.rstrip()
        if line[-1] != "}":
            continue  # should issue a warning or something

        entries = line[10:-1]

        for name in entries.split(","):
            name = name.strip()

            if name not in cited:
                yield name
                cited.add(name)


def merge_bibtex_collections(citednames, maindict, extradicts, allow_missing=False):
    """There must be a way to be efficient and stream output instead of loading
    everything into memory at once, but, meh.

    Note that we augment `citednames` with all of the names in `maindict`. The
    intention is that if we've gone to the effort of getting good data for
    some record, we don't want to trash it if the citation is temporarily
    removed (even if it ought to be manually recoverable from version
    control). Seems better to err on the side of preservation; I can write a
    quick pruning tool later if needed.

    """
    allrecords = {}

    for ed in extradicts:
        allrecords.update(ed)

    allrecords.update(maindict)

    missing = []
    from collections import OrderedDict

    records = OrderedDict()
    from itertools import chain

    wantednames = sorted(chain(citednames, maindict.keys()))

    for name in wantednames:
        rec = allrecords.get(name)
        if rec is None:
            missing.append(name)
        else:
            records[name] = rec

    if len(missing) and not allow_missing:
        # TODO: custom exception so caller can actually see what's missing;
        # could conceivably stub out missing records or something.
        raise PKError("missing BibTeX records: %s", " ".join(missing))

    return records


def get_bibtex_dict(stream):
    from bibtexparser.bparser import BibTexParser

    parser = BibTexParser()
    parser.ignore_nonstandard_types = False
    parser.homogenise_fields = False

    # TODO: one bit of homogenization that might be nice: it seems that
    # newlines get preserved, in `author` records at least. Those should be
    # replaced with spaces (and multiple spaces collapsed if needed).

    return parser.parse_file(stream).get_entry_dict()


def write_bibtex_dict(stream, entries):
    """bibtexparser.write converts the entire database to one big string and
    writes it out in one go. I'm sure it will always all fit in RAM but some
    things just will not stand.

    """
    from bibtexparser.bwriter import BibTexWriter

    writer = BibTexWriter()
    writer.indent = "  "
    writer.entry_separator = ""
    first = True

    for rec in entries:
        if first:
            first = False
        else:
            stream.write(b"\n")
        stream.write(writer._entry_to_bibtex(rec).encode("utf8"))


def merge_bibtex_with_aux(
    auxpath, mainpath, extradir, parse=get_bibtex_dict, allow_missing=False
):
    """Merge multiple BibTeX files into a single homogeneously-formatted output,
    using a LaTeX .aux file to know which records are worth paying attention
    to.

    The file identified by `mainpath` will be overwritten with the new .bib
    contents. This function is intended to be used in a version-control
    context.

    Files matching the glob "*.bib" in `extradir` will be read in to
    supplement the information in `mainpath`. Records already in the file in
    `mainpath` always take precedence.

    """
    auxpath = Path(auxpath)
    mainpath = Path(mainpath)
    extradir = Path(extradir)

    with auxpath.open("rt") as aux:
        citednames = sorted(cited_names_from_aux_file(aux))

    main = mainpath.try_open(mode="rt")
    if main is None:
        maindict = {}
    else:
        maindict = parse(main)
        main.close()

    def gen_extra_dicts():
        # If extradir does not exist, Path.glob() will return an empty list,
        # which seems acceptable to me.
        for item in sorted(extradir.glob("*.bib")):
            with item.open("rt") as extra:
                yield parse(extra)

    merged = merge_bibtex_collections(
        citednames, maindict, gen_extra_dicts(), allow_missing=allow_missing
    )

    with mainpath.make_tempfile(want="handle", resolution="overwrite") as newbib:
        write_bibtex_dict(newbib, merged.values())


# This batch of code implements the filename-recorder-to-Makefile magic.


def convert_fls_to_makefile(
    flspath, finalpath, prefix, replacement, work, mfpath, foreign_deps_ok
):
    texpwd = None

    with flspath.open("rt") as fls, mfpath.open("wt") as mf:
        mf.write(str(finalpath))
        mf.write(":")

        for line in fls:
            kind, path = line.strip().split(None, 1)
            path = Path(path)

            if kind == "PWD":
                texpwd = path  # this is always the first line
                if prefix is not None:
                    r_prefix = str((texpwd / prefix).resolve())
                r_work = str((texpwd / work).resolve())

            if kind != "INPUT":
                continue

            # properly handles absolute and relative `path`:
            r_full = str((texpwd / path).resolve())

            if r_full.startswith(r_work):
                continue  # ignore .bbl, etc

            if prefix is not None:
                if r_full.startswith(r_prefix):
                    r_full = r_full[len(r_prefix) + 1 :]

                    if replacement is not None:
                        r_full = replacement + r_full
                elif not foreign_deps_ok:
                    warn("unexpected dependent file path %r" % r_full)
                    continue

            mf.write(" ")
            mf.write(r_full)

        mf.write("\n")


# The actual command-line program

usage = """latexdriver [-lAxbBRq] [-eSTYLE] [-ESTYLE] [-M<args>] input.tex output.pdf

Drive (xe)latex sensibly. Create output.pdf from input.tex, rerunning as
necessary, silencing chatter, and hiding intermediate files in the directory
.latexwork/.

-l      - Add "-papersize letter" argument.
-A      - Add "-papersize A4" argument.
-x      - Use xetex.
-b      - Use bibtex.
-B      - Use bibtex with auto-merging and homogenization; requires `bibtexparser`.
-eSTYLE - Use 'bib' tool with bibtex style STYLE.
-ESTYLE - Optionally use the 'bib' tool in conjunction with '-B' option.
-MPREFIX[,REPLACEMENT],DEST  - Use '-recorder' option to output Makefile-format
          dependency info to DEST, stripping prefix PREFIX, optionally replacing it
          with REPLACEMENT.
-f      - Allow "foreign" dependencies from the '-recorder' option that do not
          begin with the PREFIX given to the '-M' option.
-R      - Be reckless and ignore errors from tools.
-q      - Be quiet and avoid printing anything on success.

"""

default_args = ["-interaction", "nonstopmode", "-halt-on-error", "-file-line-error"]

max_iterations = 10


def logrun(command, boring_args, interesting_arg, logpath, quiet=False, reckless=False):
    if not quiet:
        if len(boring_args):
            print("+", command, "...", interesting_arg)
        else:
            print("+", command, interesting_arg)

    argv = [command] + boring_args + [interesting_arg]

    try:
        with logpath.open("wb") as f:
            print("## running:", " ".join(argv), file=f)
            f.flush()
            subprocess.check_call(argv, stdout=f, stderr=f)
    except subprocess.CalledProcessError as e:
        if quiet:
            print("ran:", " ".join(argv), file=sys.stderr)

        with logpath.open("rt") as f:
            for line in f:
                print(line, end="", file=sys.stderr)
        print(file=sys.stderr)

        if e.returncode == -signal.SIGINT:
            raise KeyboardInterrupt()  # make sure to propagate SIGINT

        if e.returncode > 0:
            msg = 'command "%s" failed with exit status %d' % (
                " ".join(argv),
                e.returncode,
            )
        else:
            msg = 'command "%s" killed by signal %d' % (" ".join(argv), -e.returncode)

        if reckless:
            warn(msg + "; ignoring")
        else:
            die(msg)
    except Exception:
        if quiet:
            print("ran:", " ".join(argv), file=sys.stderr)
        raise


def bib_export(
    style, auxpath, bibpath, no_tool_ok=False, quiet=False, ignore_missing=False
):
    args = ["bib", "btexport"]
    if ignore_missing:
        args += ["-i"]
    args += [style, str(auxpath)]

    if not quiet:
        print("+", " ".join(args), ">" + str(bibpath))

    try:
        with bibpath.open("wb") as f:
            subprocess.check_call(args, stdout=f)
    except OSError as e:
        if e.errno == 2 and no_tool_ok:
            bibpath.try_unlink()
            return
        if quiet:
            print("ran:", " ".join(args), file=sys.stderr)
        raise
    except subprocess.CalledProcessError as e:
        if quiet:
            print("ran:", " ".join(args), file=sys.stderr)
        if e.returncode > 0:
            die(
                'command "%s >%s" failed with exit status %d',
                " ".join(args),
                bibpath,
                e.returncode,
            )
        elif e.returncode == -signal.SIGINT:
            raise KeyboardInterrupt()  # make sure to propagate SIGINT
        else:
            die(
                'command "%s >%s" killed by signal %d',
                " ".join(args),
                bibpath,
                -e.returncode,
            )


def just_smart_bibtools(bib_style, aux, bib):
    """Tectonic has taken over most of the features that this tool used to provide,
    but here's a hack to keep my smart .bib file generation working.

    """
    extradir = Path(".bibtex")
    extradir.ensure_dir(parents=True)

    bib_export(
        bib_style,
        aux,
        extradir / "ZZ_bibtools.bib",
        no_tool_ok=True,
        quiet=True,
        ignore_missing=True,
    )
    merge_bibtex_with_aux(aux, bib, extradir)


def commandline(argv=None):
    if argv is None:
        argv = sys.argv
        propagate_sigint()
        unicode_stdio()

    check_usage(usage, argv, usageifnoargs="long")

    bib_style = None
    makefile_prefix = None
    makefile_replacement = None
    makefile_dest = None
    engine_args = default_args
    engine = "pdflatex"

    # Hooray hack

    if argv[1] == "--just-smart-bibtools":
        if len(argv) == 4:
            bib_style = argv[2]
            auxpath = argv[3]
            assert auxpath.endswith(".aux")
            bibpath = auxpath[:-4] + ".bib"
        elif len(argv) == 5:
            bib_style = argv[2]
            auxpath = argv[3]
            bibpath = argv[4]
        else:
            die("--just-smart-bibtools expects exactly 2 or 3 additional arguments")
        just_smart_bibtools(bib_style, auxpath, bibpath)
        return

    # I should probably start using a real arg parser.

    do_bibtex = pop_option("b", argv)
    do_smart_bibtex = pop_option("B", argv)
    do_xetex = pop_option("x", argv)
    do_letterpaper = pop_option("l", argv)
    do_a4paper = pop_option("A", argv)
    do_reckless = pop_option("R", argv)
    do_foreign_deps = pop_option("f", argv)
    quiet = pop_option("q", argv)
    do_smart_bibtools = False

    for i in range(1, len(argv)):
        if argv[i].startswith("-e") or argv[i].startswith("-E"):
            do_smart_bibtools = argv[i].startswith("-E")
            bib_style = argv[i][2:]
            del argv[i]
            break

    for i in range(1, len(argv)):
        if argv[i].startswith("-M"):
            pieces = argv[i][2:].split(",", 2)
            if len(pieces) == 2:
                makefile_prefix, makefile_dest = pieces
            elif len(pieces) == 3:
                makefile_prefix, makefile_replacement, makefile_dest = pieces
            else:
                wrong_usage(usage, "could not parse -M argument")
            del argv[i]
            break

    if len(argv) != 3:
        wrong_usage(usage, "expect exactly 2 non-option arguments")

    if do_letterpaper and do_a4paper:
        wrong_usage(usage, 'only one of "-l" and "-A" may be specified')

    input = Path(argv[1])
    output = Path(argv[2])

    if do_smart_bibtools:
        do_smart_bibtex = True
    if bib_style is not None or do_smart_bibtex:
        do_bibtex = True
    if do_xetex:
        engine = "xelatex"
    if do_letterpaper:
        engine_args += ["-papersize", "letter"]
    if do_a4paper:
        engine_args += ["-papersize", "A4"]
    if makefile_dest is not None:
        engine_args += ["-recorder"]

    if not input.exists():
        die('input "%s" does not exist', input)

    base = input.stem
    if not len(base):
        die('failed to strip extension from input path "%s"', input)

    # I stash the annoying LaTeX output files in a hidden directory called
    # .latexwork. However, some LaTeX distributions refuse to write to hidden
    # paths by default. I figured out how to hack the configuration, but
    # that's not a scalable solution. Instead I just create a temporary
    # symlink with an acceptable name -- good jorb security.
    workdir = input.with_name(".latexwork")
    workalias = input.with_name("_latexwork")

    workdir.ensure_dir(parents=True)
    workalias.rellink_to(workdir, force=True)

    job = workalias / base
    tlog = workalias / (base + ".hllog")
    blog = workalias / (base + ".hlblg")
    engine_args += ["-jobname", str(job)]

    try:
        logrun(engine, engine_args, base, tlog, quiet=quiet)

        if do_bibtex:
            bib = input.with_suffix(".bib")
            aux = job.with_suffix(".aux")

            if do_smart_bibtex:
                extradir = input.with_name(".bibtex")

                if bib_style is not None:
                    extradir.ensure_dir(parents=True)
                    bib_export(
                        bib_style,
                        aux,
                        extradir / "ZZ_bibtools.bib",
                        no_tool_ok=True,
                        quiet=quiet,
                        ignore_missing=True,
                    )

                if not quiet:
                    print("+", "(generate and normalize)", bib)
                merge_bibtex_with_aux(aux, bib, extradir)
            elif bib_style is not None:
                bib_export(bib_style, aux, bib, quiet=quiet)

            job.with_suffix(".bib").rellink_to(bib, force=True)
            logrun("bibtex", [], str(job), blog, reckless=do_reckless, quiet=quiet)

            with blog.open("rt") as f:
                for line in f:
                    if "Warning" in line:
                        print(line, end="", file=sys.stderr)

            # force at least one extra run:
            logrun(engine, engine_args, base, tlog, quiet=quiet)

        for _ in range(max_iterations):
            keepgoing = False

            # longtables seem to always tell you to rerun latex. Stripping out
            # lines containing "longtable" makes us ignore these prompts.
            with tlog.open("rt") as f:
                for line in f:
                    if "longtable" in line:
                        continue
                    if "Rerun" in line:
                        keepgoing = True
                        break

            if not keepgoing:
                break

            logrun(engine, engine_args, base, tlog, quiet=quiet)
        else:
            # we didn't break out of the loop -- ie hit max_iterations
            die('too many iterations; check "%s"', tlog)

        job.with_suffix(".pdf").rename(output)

        if makefile_dest is not None:
            convert_fls_to_makefile(
                job.with_suffix(".fls"),
                output,
                Path(makefile_prefix),
                makefile_replacement,
                workalias,
                Path(makefile_dest),
                do_foreign_deps,
            )
    finally:
        workalias.unlink()
