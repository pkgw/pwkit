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

__all__ = str ('''djoin ensure_dir ensure_symlink make_path_func rellink
                  pathlines pathwords try_open words Path''').split ()

import io, os, pathlib

from . import PKError, text_type


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
    except OSError as e:
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
    except OSError as e:
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

    as_hdf_store      - Open as a Pandas HDFStore.
    as_uri            - Return as a file:/// URI.
    chmod             - Change file mode.
    ensure_parent (*) - Ensure the path's parent directory exists.
    exists            - Test whether path exists.
    expand            - Expand constructions like "~" or $VAR.
    glob              - Glob for files at this path (assumes it's a directory).
    is_absolute       - Test whether the path is absolute.
    is_*              - for: block_device, char_device, dir, fifo, file, socket symlink.
    iterdir           - Generate list of paths in a directory.
    joinpath(*rest)   - Create sub-paths.
    make_relative (*) - Generate a relative path.
    match(glob)       - Test whether this path matches a given glob.
    mkdir             - Create directory here; set parents=True to create parents.
    open              - Open as a file.
    read_fits (*)     - Open as a FITS file with Astropy.
    read_fits_bintable (*)
                      - Read a FITS-format binary table into a Pandas DataFrame.
    read_hdf (*)      - Read as HDF format into a Pandas DataFrame.
    read_lines (*)    - Generate lines from a text file.
    read_pandas (*)   - Read this path into a Pandas DataFrame.
    read_pickle (*)   - Read a single pickled object.
    read_pickles (*)  - Read a series of pickled objects.
    read_tabfile (*)  - Read this path as a pwkit typed data table.
    read_numpy_text (*)
                      - Read this path into a Numpy array.
    relative_to       - Compute a version of this path relative to another.
    rellink_to (*)    - Make a relative symlink.
    rename(targ)      - Rename this file or directory into `targ`.
    resolve           - Make path absolute and remove symlinks.
    rglob             - Glob for files at this path, recursively.
    rmdir             - Remove this directory.
    rmtree (*)        - Remove this directory and its contents recursively.
    scandir (*)       - Generate information about directory contents.
    stat              - Stat this path and return the result.
    symlink_to(targ)  - Make this path a symbolic link to to `targ`.
    touch             - Touch this file.
    try_open (*)      - Like open(), but return None if the file doesn't exist.
    try_unlink (*)    - Attempt to remove the file or symlink; no error if nonexistent.
    unlink            - Remove this file or symbolic link.
    unpickle_one (*)  - Unpickle an object from this path.
    unpickle_many (*) - Generate a series of objects unpickled from this path.
    with_name         - Return a new path with a different "basename".
    with_suffix       - Return a new path with a different suffix on the basename.
    write_pickle (*)  - Pickle an object into this path.
    write_pickles (*) - Pickle multiple objects into this path.

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

    # Manipulations

    def expand (self, user=False, vars=False, glob=False, resolve=False):
        from os import path
        from glob import glob

        text = text_type (self)
        if user:
            text = path.expanduser (text)
        if vars:
            text = path.expandvars (text)
        if glob:
            results = glob (text)
            if len (results) == 1:
                text = results[0]
            elif len (results) > 1:
                raise IOError ('glob of %r should\'ve returned 0 or 1 matches; got %d'
                               % (text, len (results)))

        other = self.__class__ (text)
        if resolve:
            other = other.resolve ()

        return other


    def make_relative (self, other):
        """A variant on relative_to() that allows computation of, e.g., "a" relative
        to "b", yielding "../a". This can technically give improper results if
        "b" is a directory symlink. If `self` is absolute, it is just
        returned unmodified.

        This might not work on Windows?

        """
        if self.is_absolute ():
            return self

        from os.path import relpath
        other = self.__class__ (other)
        return self.__class__ (relpath (text_type (self), text_type (other)))


    # Filesystem I/O operations

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


    def ensure_parent (self, mode=0o777, parents=False):
        """Ensure that this path's *parent* directory exists. Returns a boolean
        indicating whether the directory already existed. Will attempt to
        create superior parent directories if *parents* is True.

        """
        p = self.parent
        if p == self:
            return True # can never create root; avoids loop when parents=True

        if parents:
            p.ensure_parent (mode, True)

        try:
            p.mkdir (mode)
        except OSError as e:
            if e.errno == 17: # EEXIST?
                return True # that's fine
            raise # other exceptions are not fine
        return False


    def rmtree (self):
        import shutil
        from pwkit.cli import warn

        def on_error (func, path, exc_info):
            warn ('couldn\'t rmtree %s: in %s of %s: %s', self, func.__name__,
                  path, exc_info[1])

        shutil.rmtree (text_type (self), ignore_errors=False, onerror=on_error)
        return self


    def try_unlink (self):
        """Attempt to unlink this path, but do not raise an exception if it
        didn't exist. Returns a boolean indicating whether it was really
        unlinked.

        """
        try:
            self.unlink ()
            return True
        except OSError as e:
            if e.errno == 2:
                return False # ENOENT
            raise


    def rellink_to (self, target, force=False):
        """Like symlink_to(), but modify `target` to be relative to wherever
        `self` points.

        Path('a/b').symlink_to ('c') makes 'a/b' point to 'c', while
        Path('a/b').rellink_to ('c') makes 'a/b' point to '../c'.

        If `force`, the symlink will be forcibly created: if the `self`
        already exists as a path, it will be unlinked, and ENOENT will
        be ignored.

        """
        target = self.__class__ (target)

        if force:
            self.try_unlink ()

        if self.is_absolute ():
            target = target.absolute () # force absolute link

        return self.symlink_to (target.make_relative (self.parent))


    # Data I/O

    def try_open (self, null_if_noexist=False, **kwargs):
        try:
            return self.open (**kwargs)
        except IOError as e:
            if e.errno == 2:
                if null_if_noexist:
                    import io, os
                    return io.open (os.devnull, **kwargs)
                return None
            raise


    def read_inifile (self, noexistok=False, typed=False):
        if typed:
            from .tinifile import read_stream
        else:
            from .inifile import read_stream

        try:
            with self.open ('rb') as f:
                for item in read_stream (f):
                    yield item
        except IOError as e:
            if e.errno != 2 or not noexistok:
                raise


    def read_lines (self, mode='rt', noexistok=False, **kwargs):
        try:
            with self.open (mode=mode, **kwargs) as f:
                for line in f:
                    yield line
        except IOError as e:
            if e.errno != 2 or not noexistok:
                raise


    def read_pickle (self):
        gen = self.read_pickles ()
        value = gen.next ()
        gen.close ()
        return value


    def read_pickles (self):
        import cPickle as pickle
        with self.open (mode='rb') as f:
            while True:
                try:
                    obj = pickle.load (f)
                except EOFError:
                    break
                yield obj


    def write_pickle (self, obj):
        self.write_pickles ((obj, ))


    def write_pickles (self, objs):
        import cPickle as pickle
        with self.open (mode='wb') as f:
            for obj in objs:
                pickle.dump (obj, f)


    def as_hdf_store (self, mode='r', **kwargs):
        from pandas import HDFStore
        return HDFStore (text_type (self), mode=mode, **kwargs)


    def read_pandas (self, format='table', **kwargs):
        import pandas

        reader = getattr (pandas, 'read_' + format, None)
        if not callable (reader):
            raise PKError ('unrecognized Pandas format %r: no function pandas.read_%s',
                           format, format)

        with self.open ('rb') as f:
            return reader (f, **kwargs)


    def read_hdf (self, key, **kwargs):
        # This one needs special handling because of the "key" and path input.
        import pandas
        return pandas.read_hdf (text_type (self), key, **kwargs)


    def read_fits (self, **kwargs):
        """Open this path as a FITS file with Astropy. Keywords:

        mode='readonly' ('update', 'append', 'denywrite', 'ostream')
        memmap=None (boolean)
        save_backup=False
        cache=True
        uint=False or uint16=False
        ignore_missing_end=False
        checksum=False (boolean or 'remove')
        disable_image_compression=False
        do_not_scale_image_data=False
        ignore_blank=False
        scale_back=False

        """
        from astropy.io import fits
        return fits.open (text_type (self), **kwargs)


    def read_fits_bintable (self, hdu=1, drop_nonscalar_ok=True, **kwargs):
        from astropy.io import fits
        from .numutil import fits_recarray_to_data_frame as frtdf

        with fits.open (text_type (self), mode='readonly', **kwargs) as hdulist:
            return frtdf (hdulist[hdu].data, drop_nonscalar_ok=drop_nonscalar_ok)


    def read_tabfile (self, **kwargs):
        from .tabfile import read
        return read (text_type (self), **kwargs)


    def read_numpy_text (self, **kwargs):
        import numpy as np
        return np.loadtxt (text_type (self), **kwargs)


del _ParentPath
