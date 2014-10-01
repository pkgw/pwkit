# -*- mode: python; coding: utf-8 -*-
# Copyright 2014 Peter Williams <peter@newton.cx> and collaborators.
# Licensed under the MIT License.

"""pwkit.cli.latexdriver - the 'latexdriver' program.

This used to be a nice little shell script, but for portability it's better to
do this in Python.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = [b'commandline']

import io, os.path, subprocess, sys

from .. import PKError
from ..io import ensure_dir, ensure_symlink
from . import *

usage = """latexdriver [-x] [-b] [-l] [-eSTYLE] input.tex output.pdf

Drive (xe)latex sensibly. Create output.pdf from input.tex, rerunning as
necessary, silencing chatter, and hiding intermediate files in the directory
.latexwork/.

-x      - Use xetex.
-b      - Use bibtex.
-l      - Add "-papersize letter" argument.
-eSTYLE - Use 'bib' tool with bibtex style STYLE.
"""

default_args = ['-interaction', 'nonstopmode',
                '-halt-on-error',
                '-file-line-error']

max_iterations = 10


def logrun (command, boring_args, interesting_arg, logpath):
    if len (boring_args):
        print ('+', command, '...', interesting_arg)
    else:
        print ('+', command, interesting_arg)

    argv = [command] + boring_args + [interesting_arg]

    try:
        with io.open (logpath, 'wb') as f:
            print ('## running:', ' '.join (argv), file=f)
            f.flush ()
            subprocess.check_call (argv, stdout=f, stderr=f)
    except subprocess.CalledProcessError as e:
        with io.open (logpath, 'rt') as f:
            for line in f:
                print (line, end='', file=sys.stderr)
        print (file=sys.stderr)
        die ('command "%s" failed with exit status %d',
             ' '.join (argv), e.returncode)


def bib_export (style, auxpath, bibpath):
    args = ['bib', 'btexport', style, auxpath]
    print ('+', ' '.join (args), '>' + bibpath)

    try:
        with io.open (bibpath, 'wb') as f:
            subprocess.check_call (args, stdout=f)
    except subprocess.CalledProcessError as e:
        die ('command "%s >%s" failed with exit status %d',
             ' '.join (args), bibpath, e.returncode)



def commandline (argv=None):
    if argv is None:
        argv = sys.argv
        unicode_stdio ()

    check_usage (usage, argv, usageifnoargs='long')

    bib_style = None
    engine_args = default_args
    engine = 'pdflatex'
    workdir = '.latexwork'

    do_bibtex = pop_option ('b', argv)
    do_xetex = pop_option ('x', argv)
    do_letterpaper = pop_option ('l', argv)

    for i in xrange (1, len (argv)):
        if argv[i].startswith ('-e'):
            bib_style = argv[i][2:]
            del argv[i]
            break

    if len (argv) != 3:
        wrong_usage (usage, 'expect exactly 2 non-option arguments')

    input = argv[1]
    output = argv[2]

    if bib_style is not None:
        do_bibtex = True
    if do_xetex:
        engine = 'xelatex'
    if do_letterpaper:
        engine_args += ['-papersize', 'letter']

    if not os.path.exists (input):
        die ('input "%s" does not exist', input)

    base = os.path.splitext (os.path.basename (input))[0]
    if not len (base):
        die ('failed to strip extension from input path "%s"', input)

    # I stash the annoying LaTeX output files in a hidden directory called
    # .latexwork. However, some LaTeX distributions refuse to write to hidden
    # paths by default. I figured out how to hack the configuration, but
    # that's not a scalable solution. Instead I just create a temporary
    # symlink with an acceptable name -- good jorb security.
    workalias = '_' + workdir

    ensure_dir (workdir)
    ensure_symlink (workdir, workalias)

    job = os.path.join (workalias, base)
    tlog = os.path.join (workalias, base + '.hllog')
    blog = os.path.join (workalias, base + '.hlblg')
    engine_args += ['-jobname', job]

    try:
        logrun (engine, engine_args, base, tlog)

        if do_bibtex:
            if bib_style is not None:
                bib_export (bib_style, job + '.aux', base + '.bib')

            logrun ('bibtex', [], job, blog)

            with io.open (blog, 'rt') as f:
                for line in f:
                    if 'Warning' in line:
                        print (line, end='', file=sys.stderr)

            # force at least one extra run:
            logrun (engine, engine_args, base, tlog)

        for _ in xrange (max_iterations):
            keepgoing = False

            # longtables seem to always tell you to rerun latex. Stripping out
            # lines containing "longtable" makes us ignore these prompts.
            with io.open (tlog, 'rt') as f:
                for line in f:
                    if 'longtable' in line:
                        continue
                    if 'Rerun' in line:
                        keepgoing = True
                        break

            if not keepgoing:
                break

            logrun (engine, engine_args, base, tlog)
        else:
            # we didn't break out of the loop -- ie hit max_iterations
            die ('too many iterations; check "%s"', tlog)

        os.rename (job + '.pdf', output)
    finally:
        os.unlink (workalias)
