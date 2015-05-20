# -*- mode: python; coding: utf-8 -*-
# Copyright 2012-2015 Peter Williams <peter@newton.cx> and collaborators.
# Licensed under the MIT License.

"""pwkit.io - Utilities for input and output.

Classes:

  Path           - Augmented Path object from pathlib.

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

__all__ = b'''djoin ensure_dir ensure_symlink make_path_func rellink pathlines pathwords
           try_open words Path'''.split ()

import io, os, pathlib


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


# Path manipulations -- should largely be superseded by the Path object

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


# Extended Path object. pathlib.Path objects have fancy __new__ semantics that
# we need to jump through some hoops for.

_ParentPath = pathlib.WindowsPath if os.name == 'nt' else pathlib.PosixPath

class Path (_ParentPath):
    """Extended version of the pathlib.Path object. Methods (asterisk denotes
    pwkit-specific methods):

    as_uri            - Return as a file:/// URI.
    chmod             - Change file mode.
    exists            - Test whether path exists.
    glob              - Glob for files at this path (assumes it's a directory).
    is_absolute       - Test whether the path is absolute.
    is_*              - for: block_device, char_device, dir, fifo, file, socket symlink.
    iterdir           - Generate list of paths in a directory.
    joinpath(*rest)   - Create sub-paths.
    match(glob)       - Test whether this path matches a given glob.
    mkdir             - Create directory here; set parents=True to create parents.
    open              - Open as a file.
    pickle_many (*)   - Pickle multiple objects into this path.
    pickle_one (*)    - Pickle an object into this path.
    readlines (*)     - Generate lines from a text file.
    read_pandas (*)   - Read this path into a Pandas DataFrame.
    relative_to       - Compute a version of this path relative to another.
    rename(targ)      - Rename this file or directory into `targ`.
    resolve           - Make path absolute and remove symlinks.
    rglob             - Glob for files at this path, recursively.
    rmdir             - Remove this directory.
    scandir (*)       - Generate information about directory contents.
    stat              - Stat this path and return the result.
    symlink_to(targ)  - Make this path a symbolic link to to `targ`.
    touch             - Touch this file.
    try_open (*)      - Like open(), but return None if the file doesn't exist.
    unlink            - Remove this file or symbolic link.
    unpickle_one (*)  - Unpickle an object from this path.
    unpickle_many (*) - Generate a series of objects unpickled from this path.
    with_name         - Return a new path with a different "basename".
    with_suffix       - Return a new path with a different suffix on the basename.

    Properties:

    anchor            - The concatenation of drive and root.
    drive             - The Windows drive, if any; '' on POSIX.
    name              - The final path component.
    parts             - A tuple of the path components; '/a/b' -> ('/', 'a', 'b').
    parent            - The logical parent: '/a/b' -> '/a'; 'foo/..' -> 'foo'.
    parents           - An immutable sequence giving logical ancestors of the path.
    root              - The root: '/' or ''
    stem              - The final component without its suffix; 'foo.tar.gz' -> 'foo.tar'.
    suffix            - The final '.' extension of the final component. 'foo.tar.gz' -> '.gz'.
    suffixes          - A list of all extensions; 'foo.tar.gz' -> ['.tar', '.gz'].

    """
    def scandir (self):
        """This uses the `scandir` module or `os.scandir` to generate a listing of
        this path's contents, assuming it's a directory.

        `scandir` is different than `iterdir` because it generates `DirEntry`
        items rather than Path instances. DirEntry objects have their
        properties filled from the directory info itself, so querying them
        avoids syscalls that would be necessary with iterdir().

        DirEntry objects have these methods on POSIX systems: inode(),
        is_dir(), is_file(), is_symlink(), stat(). They have these attributes:
        `name` (the basename of the item), `path` (the concatenation of its
        name and this path).

        """
        if hasattr (os, 'scandir'):
            scandir = os.scandir
        else:
            from scandir import scandir

        from . import binary_type
        return scandir (binary_type (self))

    def try_open (self, **kwargs):
        try:
            return self.open (**kwargs)
        except IOError as e:
            if e.errno == 2:
                return None
            raise

    def readlines (self, mode='rt', noexistok=False, **kwargs):
        try:
            with self.open (mode=mode, **kwargs) as f:
                for line in f:
                    yield line
        except IOError as e:
            if e.errno != 2 or not noexistok:
                raise

    def unpickle_one (self):
        gen = self.unpickle_many ()
        value = gen.next ()
        gen.close ()
        return value

    def unpickle_many (self):
        import cPickle as pickle
        with self.open (mode='rb') as f:
            while True:
                try:
                    obj = pickle.load (f)
                except EOFError:
                    break
                yield obj

    def pickle_one (self, obj):
        self.pickle_many ((obj, ))

    def pickle_many (self, objs):
        import cPickle as pickle
        with self.open (mode='wb') as f:
            for obj in objs:
                pickle.dump (obj, f)

    def read_pandas (self, format='table', **kwargs):
        import pandas

        reader = getattr (pandas, 'read_' + format, None)
        if not callable (reader):
            raise PKError ('unrecognized Pandas format %r: no function pandas.read_%s',
                           format, format)

        with self.open ('rb') as f:
            return reader (f, **kwargs)

del _ParentPath
