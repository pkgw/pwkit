# -*- mode: python; coding: utf-8 -*-
# Copyright 2012-2016 Peter Williams <peter@newton.cx> and collaborators.
# Licensed under the MIT License.

"""The ``pwkit`` package provides many tools to ease reading and writing data
files. The most generic such tools are located in this module. The most
important tool is the :class:`Path` class for object-oriented navigation of
the filesystem.

There are also some :ref:`free functions <other-functions>` in the
:mod:`pwkit.io` module, but they are generally being superseded by operations
on :class:`Path` objects.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = str ('''djoin ensure_dir ensure_symlink make_path_func rellink
                  pathlines pathwords try_open words Path''').split ()

import io, os, pathlib
import six

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
    """This is an extended version of the :class:`pathlib.Path` class.
    (:mod:`pathlib` is built into Python 3.x and is available as a backport to
    Python 2.x.) It represents a path on the filesystem.

    """
    # Manipulations

    def expand (self, user=False, vars=False, glob=False, resolve=False):
        """Return a new :class:`Path` with various expansions performed. All
	expansions are disabled by default but can be enabled by passing in
	true values in the keyword arguments.

	user : bool (default False)
	  Expand ``~`` and ``~user`` home-directory constructs. If a username is
	  unmatched or ``$HOME`` is unset, no change is made. Calls
	  :func:`os.path.expanduser`.
	vars : bool (default False)
	  Expand ``$var`` and ``${var}`` environment variable constructs. Unknown
	  variables are not substituted. Calls :func:`os.path.expandvars`.
	glob : bool (default False)
	  Evaluate the path as a :mod:`glob` expression and use the matched path.
	  If the glob does not match anything, do not change anything. If the
	  glob matches more than one path, raise an :exc:`IOError`.
	resolve : bool (default False)
	  Call :meth:`resolve` on the return value before returning it.

        """
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


    def format (self, *args, **kwargs):
        """Return a new path formed by calling :meth:`str.format` on the
        textualization of this path.

        """
        return self.__class__ (str (self).format (*args, **kwargs))


    def get_parent (self, mode='naive'):
        """Get the path of this path’s parent directory.

        Unlike the :attr:`parent` attribute, this function can correctly
        ascend into parent directories if *self* is ``"."`` or a sequence of
        ``".."``. The precise way in which it handles these kinds of paths,
        however, depends on the *mode* parameter:

        ``"textual"``
          Return the same thing as the :attr:`parent` attribute.
        ``"resolved"``
          As *textual*, but on the :meth:`resolve`-d version of the path. This
          will always return the physical parent directory in the filesystem.
          The path pointed to by *self* must exist for this call to succeed.
        ``"naive"``
          As *textual*, but the parent of ``"."`` is ``".."``, and the parent of
          a sequence of ``".."`` is the same sequence with another ``".."``. Note
          that this manipulation is still strictly textual, so results when called
          on paths like ``"foo/../bar/../other"`` will likely not be what you want.
          Furthermore, ``p.get_parent(mode="naive")`` never yields a path equal to
          ``p``, so some kinds of loops will execute infinitely.

        """
        if mode == 'textual':
            return self.parent

        if mode == 'resolved':
            return self.resolve ().parent

        if mode == 'naive':
            from os.path import pardir

            if not len (self.parts):
                return self.__class__ (pardir)
            if all (p == pardir for p in self.parts):
                return self / pardir
            return self.parent

        raise ValueError ('unhandled get_parent() mode %r' % (mode, ))


    def make_relative (self, other):
        """Return a new path that is the equivalent of this one relative to the path
        *other*. Unlike :meth:`relative_to`, this will not throw an error if *self* is
        not a sub-path of *other*; instead, it will use ``..`` to build a relative
        path. This can result in invalid relative paths if *other* contains a
        directory symbolic link.

        If *self* is an absolute path, it is returned unmodified.

        """
        if self.is_absolute ():
            return self

        from os.path import relpath
        other = self.__class__ (other)
        return self.__class__ (relpath (text_type (self), text_type (other)))


    # Filesystem interrogation

    def readlink (self):
        """Assuming that this path is a symbolic link, read its contents and
        return them as another :class:`Path` object. An "invalid argument"
        OSError will be raised if this path does not point to a symbolic link.

        """
        from . import binary_type
        return self.__class__ (os.readlink (binary_type (self)))


    def scandir (self):
        """Iteratively scan this path, assuming it’s a directory. This requires and
        uses the :mod:`scandir` module.

        `scandir` is different than `iterdir` because it generates `DirEntry`
        items rather than Path instances. DirEntry objects have their
        properties filled from the directory info itself, so querying them
        avoids syscalls that would be necessary with iterdir().

        The generated values are :class:`scandir.DirEntry` objects which have
        some information pre-filled. These objects have methods ``inode()``,
        ``is_dir()``, ``is_file()``, ``is_symlink()``, and ``stat()``. They
        have attributes ``name`` (the basename of the entry) and ``path`` (its
        full path).

        """
        if hasattr (os, 'scandir'):
            scandir = os.scandir
        else:
            from scandir import scandir

        from . import binary_type
        return scandir (binary_type (self))


    # Filesystem modification

    def copy_to (self, dest, preserve='mode'):
        """Copy this path — as a file — to another *dest*.

        The *preserve* argument specifies which meta-properties of the file
        should be preserved:

        ``none``
          Only copy the file data.
        ``mode``
          Copy the data and the file mode (permissions, etc).
        ``all``
          Preserve as much as possible: mode, modification times, etc.

        The destination *dest* may be a directory.

        Returns the final destination path.

        """
        # shutil.copyfile() doesn't let the destination be a directory, so we
        # have to manage that possibility ourselves.

        import shutil
        dest = Path (dest)

        if dest.is_dir ():
            dest = dest / self.name

        if preserve == 'none':
            shutil.copyfile (str(self), str(dest))
        elif preserve == 'mode':
            shutil.copy (str(self), str(dest))
        elif preserve == 'all':
            shutil.copy2 (str(self), str(dest))
        else:
            raise ValueError ('unrecognized "preserve" value %r' % (preserve,))

        return dest


    def ensure_dir (self, mode=0o777, parents=False):
        """Ensure that this path exists as a directory.

        This function calls :meth:`mkdir` on this path, but does not raise an
        exception if it already exists. It does raise an exception if this
        path exists but is not a directory. If the directory is created,
        *mode* is used to set the permissions of the resulting directory, with
        the important caveat that the current :func:`os.umask` is applied.

        It returns a boolean indicating if the directory was actually created.

        If *parents* is true, parent directories will be created in the same
        manner.

        """
        if parents:
            p = self.parent
            if p == self:
                return False # can never create root; avoids loop when parents=True
            p.ensure_dir (mode, True)

        made_it = False

        try:
            self.mkdir (mode)
            made_it = True
        except OSError as e:
            if e.errno == 17: # EEXIST?
                return False # that's fine
            raise # other exceptions are not fine

        if not self.is_dir ():
            import errno
            raise OSError (errno.ENOTDIR, 'Not a directory', str(self))

        return made_it


    def ensure_parent (self, mode=0o777, parents=False):
        """Ensure that this path's *parent* directory exists.

        Returns a boolean whether the parent directory was created. Will
        attempt to create superior parent directories if *parents* is true.

        """
        return self.parent.ensure_dir (mode, parents)


    class _PathTempfileContextManager (object):
        def __init__ (self, reference, want, resolution, suffix, kwargs):
            self.reference = reference
            self.want = want
            self.resolution = resolution
            self.suffix = suffix
            self.kwargs = kwargs

        def __enter__ (self):
            from tempfile import NamedTemporaryFile

            # Pretty hacky: we know that we've been called from
            # create_tempfile() when `reference` is a class.

            if isinstance (self.reference, type):
                dir = None
                prefix = 'tmp'
                refcls = self.reference
            else:
                dir = str(self.reference.parent)
                prefix = (self.reference.name + '.')
                refcls = self.reference.__class__

            self.handle = NamedTemporaryFile (
                dir = dir,
                prefix = prefix,
                suffix = self.suffix,
                delete = False,
                **self.kwargs
            )

            self.temppath = refcls (self.handle.name)
            self.handle.path = self.temppath

            if self.want == 'handle':
                return self.handle

            if self.want == 'path':
                self.handle.close ()
                self.handle = None
                return self.temppath

            assert False, 'not reached'

        def __exit__ (self, etype, evalue, etb):
            if self.handle is not None:
                self.handle.close ()

            if etype is not None:
                # On error, keep the tempfile
                return False

            if self.resolution == 'unlink':
                self.temppath.unlink ()
            elif self.resolution == 'try_unlink':
                self.temppath.try_unlink ()
            elif self.resolution == 'keep':
                pass
            elif self.resolution == 'overwrite':
                self.temppath.rename (self.reference)
            else:
                assert False, 'not reached'

            return False


    def make_tempfile (self, want='handle', resolution='try_unlink', suffix='', **kwargs):
        """Get a context manager that creates and cleans up a uniquely-named temporary
        file with a name similar to this path.

        This function returns a context manager that creates a secure
        temporary file with a path similar to *self*. In particular, if
        ``str(self)`` is something like ``foo/bar``, the path of the temporary
        file will be something like ``foo/bar.ame8_2``.

        The object returned by the context manager depends on the *want* argument:

        ``"handle"``
          An open file-like object is returned. This is the object returned by
          :class:`tempfile.NamedTemporaryFile`. Its name on the filesystem is
          accessible as a string as its `name` attribute, or (a customization here)
          as a :class:`Path` instance as its `path` attribute.
        ``"path"``
          The temporary file is created as in ``"handle"``, but is then immediately
          closed. A :class:`Path` instance pointing to the path of the temporary file is
          instead returned.

        If an exception occurs inside the context manager block, the temporary file is
        left lying around. Otherwise, what happens to it upon exit from the context
        manager depends on the *resolution* argument:

        ``"try_unlink"``
          Call :meth:`try_unlink` on the temporary file — no exception is raised if
          the file did not exist.
        ``"unlink"``
          Call :meth:`unlink` on the temporary file — an exception is raised if
          the file did not exist.
        ``"keep"``
          The temporary file is left lying around.
        ``"overwrite"``
          The temporary file is :meth:`rename`-d to overwrite *self*.

        For instance, when rewriting important files, it’s typical to write
        the new data to a temporary file, and only rename the temporary file
        to the final destination at the end — that way, if a problem happens
        while writing the new data, the original file is left unmodified;
        otherwise you’d be stuck with a partially-written version of the file.
        This pattern can be accomplished with::

          p = Path ('path/to/important/file')
          with p.make_tempfile (resolution='overwrite', mode='wt') as h:
              print ('important stuff goes here', file=h)

        The *suffix* argument is appended to the temporary file name after the
        random portion. It defaults to the empty string. If you want it to
        operate as a typical filename suffix, include a leading ``"."``.

        Other **kwargs** are passed to :class:`tempfile.NamedTemporaryFile`.

        """
        if want not in ('handle', 'path'):
            raise ValueError ('unrecognized make_tempfile() "want" mode %r' % (want,))
        if resolution not in ('unlink', 'try_unlink', 'keep', 'overwrite'):
            raise ValueError ('unrecognized make_tempfile() "resolution" mode %r' % (resolution,))
        return Path._PathTempfileContextManager (self, want, resolution, suffix, kwargs)


    @classmethod
    def create_tempfile (cls, want='handle', resolution='try_unlink', suffix='', **kwargs):
        if want not in ('handle', 'path'):
            raise ValueError ('unrecognized create_tempfile() "want" mode %r' % (want,))
        if resolution not in ('unlink', 'try_unlink', 'keep'):
            raise ValueError ('unrecognized create_tempfile() "resolution" mode %r' % (resolution,))
        return cls._PathTempfileContextManager (cls, want, resolution, suffix, kwargs)


    def rellink_to (self, target, force=False):
        """Make this path a symlink pointing to the given *target*, generating the
	proper relative path using :meth:`make_relative`. This gives different
	behavior than :meth:`symlink_to`. For instance, ``Path
	('a/b').symlink_to ('c')`` results in ``a/b`` pointing to the path
	``c``, whereas :meth:`rellink_to` results in it pointing to ``../c``.
	This can result in broken relative paths if (continuing the example)
	``a`` is a symbolic link to a directory.

	If either *target* or *self* is absolute, the symlink will point at
	the absolute path to *target*. The intention is that if you’re trying
	to link ``/foo/bar`` to ``bee/boo``, it probably makes more sense for
	the link to point to ``/path/to/.../bee/boo`` rather than
	``../../../../bee/boo``.

	If *force* is true, :meth:`try_unlink` will be called on *self* before
	the link is made, forcing its re-creation.

        """
        target = self.__class__ (target)

        if force:
            self.try_unlink ()

        if self.is_absolute ():
            target = target.absolute () # force absolute link

        return self.symlink_to (target.make_relative (self.parent))


    def rmtree (self, errors='warn'):
        """Recursively delete this directory and its contents. The *errors* keyword
        specifies how errors are handled:

        "warn" (the default)
          Print a warning to standard error.
        "ignore"
          Ignore errors.

        """
        import shutil

        if errors == 'ignore':
            ignore_errors = True
            onerror = None
        elif errors == 'warn':
            ignore_errors = False
            from .cli import warn

            def onerror (func, path, exc_info):
                warn ('couldn\'t rmtree %s: in %s of %s: %s', self, func.__name__,
                      path, exc_info[1])
        else:
            raise ValueError ('unexpected "errors" keyword %r' % (errors,))

        shutil.rmtree (text_type (self), ignore_errors=ignore_errors, onerror=onerror)
        return self


    def try_unlink (self):
        """Try to unlink this path. If it doesn't exist, no error is returned. Returns
        a boolean indicating whether the path was really unlinked.

        """
        try:
            self.unlink ()
            return True
        except OSError as e:
            if e.errno == 2:
                return False # ENOENT
            raise


    # Data I/O

    def try_open (self, null_if_noexist=False, **kwargs):
        """Call :meth:`Path.open` on this path (passing *kwargs*) and return the
        result. If the file doesn't exist, the behavior depends on
        *null_if_noexist*. If it is false (the default), ``None`` is returned.
        Otherwise, :data:`os.devnull` is opened and returned.

        """
        try:
            return self.open (**kwargs)
        except IOError as e:
            if e.errno == 2:
                if null_if_noexist:
                    import io, os
                    return io.open (os.devnull, **kwargs)
                return None
            raise


    def as_hdf_store (self, mode='r', **kwargs):
        """Return the path as an opened :class:`pandas.HDFStore` object. Note that the
        :class:`HDFStore` constructor unconditionally prints messages to
        standard output when opening and closing files, so use of this
        function will pollute your program’s standard output. The *kwargs* are
        forwarded to the :class:`HDFStore` constructor.

        """
        from pandas import HDFStore
        return HDFStore (text_type (self), mode=mode, **kwargs)


    def read_astropy_ascii (self, **kwargs):
        """Open as an ASCII table, returning a :class:`astropy.table.Table` object.
        Keyword arguments are passed to :func:`astropy.io.ascii.open`; valid
        ones likely include:

        - ``names = <list>`` (column names)
        - ``format`` ('basic', 'cds', 'csv', 'ipac', ...)
        - ``guess = True`` (guess table format)
        - ``delimiter`` (column delimiter)
        - ``comment = <regex>``
        - ``header_start = <int>`` (line number of header, ignoring blank and comment lines)
        - ``data_start = <int>``
        - ``data_end = <int>``
        - ``converters = <dict>``
        - ``include_names = <list>`` (names of columns to include)
        - ``exclude_names = <list>`` (names of columns to exclude; applied after include)
        - ``fill_values = <dict>`` (filler values)

        """
        from astropy.io import ascii
        return ascii.read (text_type (self), **kwargs)


    def read_fits (self, **kwargs):
        """Open as a FITS file, returning a :class:`astropy.io.fits.HDUList` object.
        Keyword arguments are passed to :func:`astropy.io.fits.open`; valid
        ones likely include:

        - ``mode = 'readonly'`` (or "update", "append", "denywrite", "ostream")
        - ``memmap = None``
        - ``save_backup = False``
        - ``cache = True``
        - ``uint = False``
        - ``ignore_missing_end = False``
        - ``checksum = False``
        - ``disable_image_compression = False``
        - ``do_not_scale_image_data = False``
        - ``ignore_blank = False``
        - ``scale_back = False``

        """
        from astropy.io import fits
        return fits.open (text_type (self), **kwargs)


    def read_fits_bintable (self, hdu=1, drop_nonscalar_ok=True, **kwargs):
        """Open as a FITS file, read in a binary table, and return it as a
        :class:`pandas.DataFrame`, converted with
        :func:`pkwit.numutil.fits_recarray_to_data_frame`. The *hdu* argument
        specifies which HDU to read, with its default 1 indicating the first
        FITS extension. The *drop_nonscalar_ok* argument specifies if
        non-scalar table values (which are inexpressible in
        :class:`pandas.DataFrame`s) should be silently ignored (``True``) or
        cause a :exc:`ValueError` to be raised (``False``). Other **kwargs**
        are passed to :func:`astropy.io.fits.open`, (see
        :meth:`Path.read_fits`) although the open mode is hardcoded to be
        ``"readonly"``.

        """
        from astropy.io import fits
        from .numutil import fits_recarray_to_data_frame as frtdf

        with fits.open (text_type (self), mode='readonly', **kwargs) as hdulist:
            return frtdf (hdulist[hdu].data, drop_nonscalar_ok=drop_nonscalar_ok)


    def read_hdf (self, key, **kwargs):
        """Open as an HDF5 file using :mod:`pandas` and return the item stored under
        the key *key*. *kwargs* are passed to :func:`pandas.read_hdf`.

        """
        # This one needs special handling because of the "key" and path input.
        import pandas
        return pandas.read_hdf (text_type (self), key, **kwargs)


    def read_inifile (self, noexistok=False, typed=False):
        """Open assuming an “ini-file” format and return a generator yielding data
        records using either :func:`pwkit.inifile.read_stream` (if *typed* is
        false) or :func:`pwkit.tinifile.read_stream` (if it’s true). The
        latter version is designed to work with numerical data using the
        :mod:`pwkit.msmt` subsystem. If *noexistok* is true, a nonexistent
        file will result in no items being generated rather than an
        :exc:`IOError` being raised.

        """
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


    def read_json (self, mode='rt', **kwargs):
        """Use the :mod:`json` module to read in this file as a JSON-formatted data
        structure. Keyword arguments are passed to :func:`json.load`. Returns the
        read-in data structure.

        """
        import json

        with self.open (mode=mode) as f:
            return json.load (f, **kwargs)


    def read_lines (self, mode='rt', noexistok=False, **kwargs):
        """Generate a sequence of lines from the file pointed to by this path, by
        opening as a regular file and iterating over it. The lines therefore
        contain their newline characters. If *noexistok*, a nonexistent file
        will result in an empty sequence rather than an exception. *kwargs*
        are passed to :meth:`Path.open`.

        """
        try:
            with self.open (mode=mode, **kwargs) as f:
                for line in f:
                    yield line
        except IOError as e:
            if e.errno != 2 or not noexistok:
                raise


    def read_numpy (self, **kwargs):
        """Read this path into a :class:`numpy.ndarray` using :func:`numpy.load`.
        *kwargs* are passed to :func:`numpy.load`; they likely are:

	mmap_mode : None, 'r+', 'r', 'w+', 'c'
          Load the array using memory-mapping
	allow_pickle : bool = True
          Whether Pickle-format data are allowed; potential security hazard.
	fix_imports : bool = True
	  Try to fix Python 2->3 import renames when loading Pickle-format data.
	encoding : 'ASCII', 'latin1', 'bytes'
	  The encoding to use when reading Python 2 strings in Pickle-format data.

        """
        import numpy as np
        with self.open ('rb') as f:
            return np.load (f, **kwargs)


    def read_numpy_text (self, dfcols=None, **kwargs):
        """Read this path into a :class:`numpy.ndarray` as a text file using
        :func:`numpy.loadtxt`. In normal conditions the returned array is
        two-dimensional, with the first axis spanning the rows in the file and
        the second axis columns (but see the *unpack* and *dfcols* keywords).

        If *dfcols* is not None, the return value is a
        :class:`pandas.DataFrame` constructed from the array. *dfcols* should
        be an iterable of column names, one for each of the columns returned
        by the :func:`numpy.loadtxt` call. For convenience, if *dfcols* is a
        single string, it will by turned into an iterable by a call to
        :func:`str.split`.

        The remaining *kwargs* are passed to :func:`numpy.loadtxt`; they likely are:

	dtype : data type
	  The data type of the resulting array.
	comments : str
	  If specific, a character indicating the start of a comment.
	delimiter : str
	  The string that separates values. If unspecified, any span of whitespace works.
	converters : dict
	  A dictionary mapping zero-based column *number* to a function that will
	  turn the cell text into a number.
	skiprows : int (default=0)
	  Skip this many lines at the top of the file
	usecols : sequence
	  Which columns keep, by number, starting at zero.
	unpack : bool (default=False)
	  If true, the return value is transposed to be of shape ``(cols, rows)``.
	ndmin : int (default=0)
	  The returned array will have at least this many dimensions; otherwise
	  mono-dimensional axes will be squeezed.

        """
        import numpy as np

        if dfcols is not None:
            kwargs['unpack'] = True

        retval = np.loadtxt (text_type (self), **kwargs)

        if dfcols is not None:
            import pandas as pd
            if isinstance (dfcols, six.string_types):
                dfcols = dfcols.split ()
            retval = pd.DataFrame (dict (zip (dfcols, retval)))

        return retval


    def read_pandas (self, format='table', **kwargs):
        """Read using :mod:`pandas`. The function ``pandas.read_FORMAT`` is called
        where ``FORMAT`` is set from the argument *format*. *kwargs* are
        passed to this function. Supported formats likely include
        ``clipboard``, ``csv``, ``excel``, ``fwf``, ``gbq``, ``html``,
        ``json``, ``msgpack``, ``pickle``, ``sql``, ``sql_query``,
        ``sql_table``, ``stata``, ``table``. Note that ``hdf`` is not
        supported because it requires a non-keyword argument; see
        :meth:`Path.read_hdf`.

        """
        import pandas

        reader = getattr (pandas, 'read_' + format, None)
        if not callable (reader):
            raise PKError ('unrecognized Pandas format %r: no function pandas.read_%s',
                           format, format)

        with self.open ('rb') as f:
            return reader (f, **kwargs)


    def read_pickle (self):
        """Open the file, unpickle one object from it using :mod:`cPickle`, and return
        it.

        """
        gen = self.read_pickles ()
        value = gen.next ()
        gen.close ()
        return value


    def read_pickles (self):
        """Generate a sequence of objects by opening the path and unpickling items
        until EOF is reached.

        """
        import cPickle as pickle
        with self.open (mode='rb') as f:
            while True:
                try:
                    obj = pickle.load (f)
                except EOFError:
                    break
                yield obj


    def read_tabfile (self, **kwargs):
        """Read this path as a table of typed measurements via
        :func:`pwkit.tabfile.read`. Returns a generator for a sequence of
        :class:`pwkit.Holder` objects, one for each row in the table, with
        attributes for each of the columns.

        tabwidth : int (default=8)
            The tab width to assume. Defaults to 8 and should not be changed unless
            absolutely necessary.
        mode : str (default='rt')
            The file open mode, passed to :func:`io.open`.
        noexistok : bool (default=False)
            If true, a nonexistent file will result in no items being generated, as
            opposed to an :exc:`IOError`.
        kwargs : keywords
            Additional arguments are passed to :func:`io.open`.

        """
        from .tabfile import read
        return read (text_type (self), **kwargs)


    def read_yaml (self, encoding=None, errors=None, newline=None, **kwargs):
        """Read this path as a YAML document.

        The *encoding*, *errors*, and *newline* keywords are passed to
        :meth:`open`. The remaining *kwargs* are passed to :meth:`yaml.load`.

        """
        import yaml

        with self.open (mode='rt', encoding=encoding, errors=errors, newline=newline) as f:
            return yaml.load (f, **kwargs)


    def write_pickle (self, obj):
        """Dump *obj* to this path using :mod:`cPickle`."""
        self.write_pickles ((obj, ))


    def write_pickles (self, objs):
        """*objs* must be iterable. Write each of its values to this path in sequence
        using :mod:`cPickle`.

        """
        import cPickle as pickle
        with self.open (mode='wb') as f:
            for obj in objs:
                pickle.dump (obj, f)


    def write_yaml (self, data, encoding=None, errors=None, newline=None, **kwargs):
        """Read *data* to this path as a YAML document.

        The *encoding*, *errors*, and *newline* keywords are passed to
        :meth:`open`. The remaining *kwargs* are passed to :meth:`yaml.dump`.

        """
        import yaml

        with self.open (mode='wt', encoding=encoding, errors=errors, newline=newline) as f:
            return yaml.dump (data, stream=f, **kwargs)


del _ParentPath
