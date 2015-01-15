# -*- mode: python; coding: utf-8 -*-
# Copyright 2012-2015 Peter Williams <peter@newton.cx> and collaborators.
# Licensed under the MIT License.

"""pwkit.io - Utilities for input and output.

Functions:

  djoin          - "Dotless" version of os.path.join.
  ensure_dir     - Ensure that a directory exists.
  ensure_symlink - Ensure that a symbolic link exists.
  make_path_func - Make a function that conveniently constructs paths.
  pathlines      - Yield lines from a text file at a specified path.
  pathwords      - Yield lists of words from a text file at a specified path.
  rellink        - Create a symbolic link to a relative path.
  try_open       - Try opening a file; no exception if it doesn't exist.
  words          - Split a list of lines into individual words.

"""

from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = (b'djoin ensure_dir ensure_symlink make_path_func rellink pathlines '
           b'pathwords try_open words').split ()

import io, os


# Reading text.

def try_open (*args, **kwargs):
    """Simply a wrapper for io.open(), unless an IOError with errno=2 (ENOENT) is
    raised, in which case None is retured.

    """
    try:
        return io.open (*args, **kwargs)
    except IOError as e:
        if e.errno == 2:
            return None
        raise


def words (linegen):
    for line in linegen:
        a = line.split ('#', 1)[0].strip ().split ()
        if len (a):
            yield a


def pathwords (path, mode='rt', noexistok=False, **kwargs):
    try:
        with io.open (path, mode, **kwargs) as f:
            for line in f:
                a = line.split ('#', 1)[0].strip ().split ()
                if len (a):
                    yield a
    except IOError as e:
        if e.errno != 2 or not noexistok:
            raise


def pathlines (path, mode='rt', noexistok=False, **kwargs):
    try:
        with io.open (path, mode, **kwargs) as f:
            for line in f:
                yield line
    except IOError as e:
        if e.errno != 2 or not noexistok:
            raise


# Path manipulations.

def make_path_func (*baseparts):
    """Return a function that joins paths onto some base directory."""
    from os.path import join
    base = join (*baseparts)
    def path_func (*args):
        return join (base, *args)
    return path_func


def djoin (*args):
    """'dotless' join, for nicer paths."""
    from os.path import join

    i = 0
    alen = len (args)

    while i < alen and (args[i] == '' or args[i] == '.'):
        i += 1

    if i == alen:
        return '.'

    return join (*args[i:])


# Doing stuff on the filesystem.

def rellink (source, dest):
    """Create a symbolic link to path *source* from path *dest*. If either
    *source* or *dest* is an absolute path, the link from *dest* will point to
    the absolute path of *source*. Otherwise, the link to *source* from *dest*
    will be a relative link.

    """
    from os.path import isabs, dirname, relpath, abspath

    if isabs (source):
        os.symlink (source, dest)
    elif isabs (dest):
        os.symlink (abspath (source), dest)
    else:
        os.symlink (relpath (source, dirname (dest)), dest)


def ensure_dir (path, parents=False):
    """Returns a boolean indicating whether the directory already existed. Will
    attempt to create parent directories if *parents* is True.

    """
    if parents:
        from os.path import dirname
        parent = dirname (path)
        if len (parent) and parent != path:
            ensure_dir (parent, True)

    try:
        os.mkdir (path)
    except OSError, e:
        if e.errno == 17: # EEXIST
            return True
        raise
    return False


def ensure_symlink (src, dst):
    """Ensure the existence of a symbolic link pointing to src named dst. Returns
    a boolean indicating whether the symlink already existed.

    """
    try:
        os.symlink (src, dst)
    except OSError, e:
        if e.errno == 17: # EEXIST
            return True
        raise
    return False
