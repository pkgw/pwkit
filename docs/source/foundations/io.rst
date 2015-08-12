.. Copyright 2015 Peter K. G. Williams <peter@newton.cx> and collaborators.
   This file licensed under the Creative Commons Attribution-ShareAlike 3.0
   Unported License (CC-BY-SA).

Convenient file input and output (:mod:`pwkit.io`)
========================================================================

.. module:: pwkit.io
   :synopsis: basic utilities for file input and output

The ``pwkit`` package provides many tools to ease reading and writing data
files. The most generic such tools are located in the :mod:`pwkit.io` module.
The most important tool is the :class:`Path` class for object-oriented
navigation of the filesystem.


.. class:: Path(path)

   This is an extended version of the :class:`pathlib.Path` class.
   (:mod:`pathlib` is built into Python 3.x and is available as a backport to
   Python 2.x.) It represents a path on the filesystem.

   The key methods on :class:`Path` instances are:

   - :meth:`absolute` — see also :meth:`resolve`
   - :meth:`as_hdf_store`
   - :meth:`as_uri`
   - :meth:`chmod`
   - :meth:`cwd`
   - :meth:`ensure_parent`
   - :meth:`exists`
   - :meth:`expand`
   - :meth:`glob`
   - :meth:`is_absolute`
   - :meth:`is_block_device`
   - :meth:`is_char_device`
   - :meth:`is_dir`
   - :meth:`is_fifo`
   - :meth:`is_file`
   - :meth:`is_socket`
   - :meth:`is_symlink`
   - :meth:`iterdir` — see also :meth:`scandir`
   - :meth:`joinpath`
   - :meth:`make_relative`
   - :meth:`match`
   - :meth:`mkdir`
   - :meth:`open` — see also :meth:`try_open`
   - :meth:`read_lines`
   - :meth:`read_fits` — see also :meth:`read_fits_bintable`
   - :meth:`read_fits_bintable` — see also :meth:`read_fits`
   - :meth:`read_hdf`
   - :meth:`read_inifile`
   - :meth:`read_numpy_text`
   - :meth:`read_pandas`
   - :meth:`read_pickle`
   - :meth:`read_pickles`
   - :meth:`read_tabfile`
   - :meth:`relative_to` — see also :meth:`make_relative`
   - :meth:`rellink_to` — see also :meth:`symlink_to`
   - :meth:`rename`
   - :meth:`resolve` — see also :meth:`absolute`
   - :meth:`rglob`
   - :meth:`rmdir` — see also :meth:`rmtree`
   - :meth:`rmtree` — see also :meth:`rmdir`
   - :meth:`scandir` — see also :meth:`iterdir`
   - :meth:`stat`
   - :meth:`symlink_to` — see also :meth:`rellink_to`
   - :meth:`touch`
   - :meth:`try_open` — see also :meth:`open`
   - :meth:`try_unlink` — see also :meth:`unlink`
   - :meth:`unlink` — see also :meth:`try_unlink`
   - :meth:`with_name`
   - :meth:`with_suffix`
   - :meth:`write_pickle`
   - :meth:`write_pickles`

   Attributes are:

   - :attr:`anchor`
   - :attr:`drive`
   - :attr:`name`
   - :attr:`parts`
   - :attr:`parent`
   - :attr:`parents`
   - :attr:`root`
   - :attr:`stem`
   - :attr:`suffix`
   - :attr:`suffixes`


There are also some :ref:`free functions <other-functions>` in the
:mod:`pwkit.io` module, but they are generally being superseded by operations
on :class:`Path` objects.


:class:`Path` methods
------------------------------------------------------------------------

.. method:: Path.absolute()

   Return an absolute version of the path. Unlike :meth:`resolve`, does not
   normalize the path or resolve symlinks.


.. method:: Path.as_hdf_store(mode='r', **kwargs)

   Return the path as an opened :class:`pandas.HDFStore` object. Note that the
   :class:`HDFStore` constructor unconditionally prints messages to standard
   output when opening and closing files, so use of this function will pollute
   your program’s standard output. The *kwargs* are forwarded to the
   :class:`HDFStore` constructor.


.. method:: Path.as_uri()

   Return the path stringified as a `file:///` URI.


.. method:: Path.chmod(mode)

   Change the mode of the named path. Remember to use octal ``0o755``
   notation!


.. method:: Path.ensure_parent(mode=0o777, parents=False)

   Ensure that this path's *parent* directory exists. Returns a boolean
   indicating whether the parent directory already existed. Will attempt to
   create superior parent directories if *parents* is true. Unlike
   :meth:`Path.mkdir`, will not raise an exception if parents already exist.


.. method:: Path.exists()

   Returns whether the path exists.


.. method:: Path.expand(user=False, vars=False, glob=False, resolve=False)

   Return a new :class:`Path` with various expansions performed. All
   expansions are disabled by default but can be enabled by passing in true
   values in the keyword arguments.

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


.. method:: Path.glob(pattern)

   Assuming that the path is a directory, iterate over its contents and return
   sub-paths matching the given shell-style glob pattern.


.. method:: Path.is_absolute()

   Returns whether the path is absolute.


.. method:: Path.is_block_device()

   Returns whether the path resolves to a block device file.


.. method:: Path.is_char_device()

   Returns whether the path resolves to a character device file.


.. method:: Path.is_dir()

   Returns whether the path resolves to a directory.


.. method:: Path.is_fifo()

   Returns whether the path resolves to a Unix FIFO.


.. method:: Path.is_file()

   Returns whether the path resolves to a regular file.


.. method:: Path.is_socket()

   Returns whether the path resolves to a Unix socket.


.. method:: Path.is_symlink()

   Returns whether the path resolves to a symbolic link.


.. method:: Path.iterdir()

   Assuming the path is a directory, generate a sequence of sub-paths
   corresponding to its contents.


.. method:: Path.joinpath(*args)

   Combine this path with several new components. If one of the arguments is
   absolute, all previous components are discarded.


.. method:: Path.make_relative(other)

   Return a new path that is the equivalent of this one relative to the path
   *other*. Unlike :meth:`relative_to`, this will not throw an error if *self*
   is not a sub-path of *other*; instead, it will use ``..`` to build a
   relative path. This can result in invalid relative paths if *other* contains
   a directory symbolic link.

   If *self* is an absolute path, it is returned unmodified.


.. method:: Path.match(pattern)

   Test whether this path matches the given shell glob pattern.


.. method:: Path.mkdir(mode=0o777, parents=False)

   Create a directory at this path location. Creates parent directories if
   *parents* is true. Raises :class:`OSError` if the path already exists, even
   if *parents* is true.


.. method:: Path.open(mode='r', buffering=-1, encoding=None, errors=None, newline=None)

   Open the file pointed at by the path and return a :class:`file` object.
   **TODO**: verify whether semantics correspond to :func:`io.open` or plain
   builtin :func:`open`.


.. method:: Path.read_lines(mode='rt', noexistok=False, **kwargs)

   Generate a sequence of lines from the file pointed to by this path, by
   opening as a regular file and iterating over it. The lines therefore
   contain their newline characters. If *noexistok*, a nonexistent file will
   result in an empty sequence rather than an exception. *kwargs* are passed
   to :meth:`Path.open`.


.. method:: Path.read_fits(**kwargs)

   Open as a FITS file, returning a :class:`astropy.io.fits.HDUList` object.
   Keyword arguments are passed to :func:`astropy.io.fits.open`; valid ones
   likely include:

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


.. method:: Path.read_fits_bintable(hdu=1, drop_nonscalar_ok=True, **kwargs)

   Open as a FITS file, read in a binary table, and return it as a
   :class:`pandas.DataFrame`, converted with
   :func:`pkwit.numutil.fits_recarray_to_data_frame`. The *hdu* argument
   specifies which HDU to read, with its default 1 indicating the first FITS
   extension. The *drop_nonscalar_ok* argument specifies if non-scalar table
   values (which are inexpressible in :class:`pandas.DataFrame`s) should be
   silently ignored (``True``) or cause a :exc:`ValueError` to be raised
   (``False``). Other **kwargs** are passed to :func:`astropy.io.fits.open`,
   (see :meth:`Path.read_fits`) although the open mode is hardcoded to be
   ``"readonly"``.


.. method:: Path.read_hdf(key, **kwargs)

   Open as an HDF5 file using :mod:`pandas` and return the item stored under
   the key *key*. *kwargs* are passed to :func:`pandas.read_hdf`.


.. method:: Path.read_inifile(noexistok=False, typed=False)

   Open assuming an “ini-file” format and return a generator yielding data
   records using either :func:`pwkit.inifile.read_stream` (if *typed* is
   false) or :func:`pwkit.tinifile.read_stream` (if it’s true). The latter
   version is designed to work with numerical data using the :mod:`pwkit.msmt`
   subsystem. If *noexistok* is true, a nonexistent file will result in no
   items being generated rather than an :exc:`IOError` being raised.


.. method:: Path.read_numpy_text(**kwargs)

   Read this path into a :class:`numpy.ndarray` as a text file using
   :func:`numpy.loadtxt`. In normal conditions the returned array is
   two-dimensional, with the first axis spanning the rows in the file and the
   second axis columns (but see the *unpack* keyword). *kwargs* are passed to
   :func:`numpy.loadtxt`; they likely are:

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


.. method:: Path.read_pandas(format='table', **kwargs)

   Read using :mod:`pandas`. The function ``pandas.read_FORMAT`` is called
   where ``FORMAT`` is set from the argument *format*. *kwargs* are passed to
   this function. Supported formats likely include ``clipboard``, ``csv``,
   ``excel``, ``fwf``, ``gbq``, ``html``, ``json``, ``msgpack``, ``pickle``,
   ``sql``, ``sql_query``, ``sql_table``, ``stata``, ``table``. Note that
   ``hdf`` is not supported because it requires a non-keyword argument; see
   :meth:`Path.read_hdf`.


.. method:: Path.read_pickle()

   Open the file, unpickle one object from it using :mod:`cPickle`, and return
   it.


.. method:: Path.read_pickles()

   Generate a sequence of objects by opening the path and unpickling items
   until EOF is reached.


.. method:: Path.read_tabfile(tabwidth=8, mode='rt', noexistok=False, **kwargs)

   Read this path as a table of typed measurements via
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


.. method:: Path.relative_to(*other)

   Return this path as made relative to another path identified by *other*. If
   this is not possible, raise :exc:`ValueError`.


.. method:: Path.rellink_to(target, force=False)

   Make this path a symlink pointing to the given *target*, generating the
   proper relative path using :meth:`make_relative`. This gives different
   behavior than :meth:`symlink_to`. For instance, ``Path ('a/b').symlink_to
   ('c')`` results in ``a/b`` pointing to the path ``c``, whereas
   :meth:`rellink_to` results in it pointing to ``../c``. This can result in
   broken relative paths if (continuing the example) ``a`` is a symbolic link
   to a directory.

   If either *target* or *self* is absolute, the symlink will point at the
   absolute path to *target*. The intention is that if you’re trying to link
   ``/foo/bar`` to ``bee/boo``, it probably makes more sense for the link to
   point to ``/path/to/.../bee/boo`` rather than ``../../../../bee/boo``.

   If *force* is true, :meth:`try_unlink` will be called on *self* before the
   link is made, forcing its re-creation.


.. method:: Path.rename(target)

   Rename this path to *target*.


.. method:: Path.resolve()

   Make this path absolute, resolving all symlinks and normalizing.


.. method:: Path.rglob(pattern)

   Recursively yield all files and directories matching the shell glob pattern
   *pattern* below this path.


.. method:: Path.rmdir()

   Delete this path, if it is an empty directory.


.. method:: Path.rmtree()

   Recursively delete this directory and its contents. If any errors are
   encountered, they will be printed to standard error.


.. method:: Path.scandir()

   Iteratively scan this path, assuming it’s a directory. This requires and
   uses the :mod:`scandir` module. The generated values are
   :class:`scandir.DirEntry` objects which have some information pre-filled.
   These objects have methods ``inode()``, ``is_dir()``, ``is_file()``,
   ``is_symlink()``, and ``stat()``. They have attributes ``name`` (the
   basename of the entry) and ``path`` (its full path).


.. method:: Path.stat()

   Run :func:`os.stat` on the path and return the result.


.. method:: Path.symlink_to(target, target_is_directory=False)

   Make this path a symlink pointing to the given target.


.. method:: Path.touch(mode=0o666, exist_ok=True)

   Create a file at this path with the given mode, if needed.


.. method:: Path.try_open(null_if_noexist=False, **kwargs)

   Call :meth:`Path.open` on this path (passing *kwargs*) and return the
   result. If the file doesn't exist, the behavior depends on
   *null_if_noexist*. If it is false (the default), ``None`` is returned.
   Otherwise, :data:`os.devnull` is opened and returned.


.. method:: Path.try_unlink()

   Try to unlink this path. If it doesn't exist, no error is returned. Returns
   a boolean indicating whether the path was really unlinked.


.. method:: Path.unlink()

   Unlink this file or symbolic link.


.. method:: Path.with_name(name)

   Return a new path with the file name changed.


.. method:: Path.with_suffix(suffix)

   Return a new path with the file suffix changed, or a new suffix added if
   there was none before. *suffix* should start with a ``"."``.


.. method:: Path.write_pickle(obj)

   Dump *obj* to this path using :mod:`cPickle`.


.. method:: Path.write_pickles(objs)

   *objs* must be iterable. Write each of its values to this path in sequence
   using :mod:`cPickle`.


.. staticmethod:: Path.cwd()

   Returns a new path containing the absolute path of the current working
   directory.


:class:`Path` attributes
------------------------------------------------------------------------

.. attribute:: Path.anchor

   The concatenation of :attr:`Path.drive` and :attr:`Path.root`.


.. attribute:: Path.drive

   The Windows or network drive of the path. The empty string on POSIX.


.. attribute:: Path.name

   The final path component.


.. attribute:: Path.parts

   A tuple of the path components. The path ``/a/b`` maps to ``("/", "a",
   "b")``.


.. attribute:: Path.parent

   The path’s logical parent; that is, the path with the final component
   removed. The parent of ``foo`` is ``.``; the parent of ``.`` is ``.``; the
   parent of ``/`` is ``/``.


.. attribute:: Path.parents

   An immutable sequence giving the logical ancestors of the path. Given a
   :class:`Path` ``p``, ``p.parents[0]`` is the same as ``p.parent``,
   ``p.parents[1]`` matches ``p.parent.parent``, and so on. This item is of
   finite size, however, so going too far (e.g. ``p.parents[17]``) will yield
   an :exc:`IndexError`.


.. attribute:: Path.stem

   The final component without its suffix. The stem of ``"foo.tar.gz"`` is
   ``"foo.tar"``.


.. attribute:: Path.suffix

   The suffix of the final path component. The suffix of ``"foo.tar.gz"`` is
   ``".gz"``.


.. attribute:: Path.suffixes

   A list of all suffixes on the final component. The suffixes of
   ``"foo.tar.gz"`` are ``[".tar", ".gz"]``.


.. _other-functions:

Other functions in :mod:`pwkit.io`
------------------------------------------------------------------------

These are generally superseded by operations on :class:`Path`.


.. function:: try_open(*args, **kwargs)

   Placeholder.


.. function:: words(linegen)

   Placeholder.


.. function:: pathwords(path, mode='rt', noexistok=False, **kwargs)

   Placeholder.


.. function:: pathlines(path, mode='rt', noexistok=False, **kwargs)

   Placeholder.


.. function:: make_path_func(*baseparts)

   Placeholder.


.. function:: djoin(*args)

   Placeholder.


.. function:: rellink(source, dest)

   Placeholder.


.. function:: ensure_dir(path, parents=False)

   Placeholder.


.. function:: ensure_symlink(src, dst)

   Placeholder.
