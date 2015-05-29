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

   - :meth:`as_hdf_store`
   - :meth:`as_uri`
   - :meth:`chmod`
   - :meth:`ensure_parent`
   - :meth:`exists`
   - :meth:`glob`
   - :meth:`is_absolute`
   - :meth:`is_block_device`
   - :meth:`is_char_device`
   - :meth:`is_dir`
   - :meth:`is_fifo`
   - :meth:`is_file`
   - :meth:`is_socket`
   - :meth:`is_symlink`
   - :meth:`iterdir`
   - :meth:`joinpath`
   - :meth:`match`
   - :meth:`mkdir`
   - :meth:`open`
   - :meth:`read_lines`
   - :meth:`read_fits`
   - :meth:`read_hdf`
   - :meth:`read_pandas`
   - :meth:`read_pickle`
   - :meth:`read_pickles`
   - :meth:`relative_to`
   - :meth:`rename`
   - :meth:`resolve`
   - :meth:`rglob`
   - :meth:`rmdir`
   - :meth:`rmtree`
   - :meth:`scandir`
   - :meth:`stat`
   - :meth:`symlink_to`
   - :meth:`touch`
   - :meth:`try_open`
   - :meth:`try_unlink`
   - :meth:`unlink`
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


.. method:: Path.read_hdf(key, **kwargs)

   Open as an HDF5 file using :mod:`pandas` and return the item stored under
   the key *key*. *kwargs* are passed to :func:`pandas.read_hdf`.


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


.. method:: Path.relative_to(*other)

   Return this path as made relative to another path identified by *other*. If
   this is not possible, raise :exc:`ValueError`.


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


.. method:: Path.try_open(**kwargs)

   Call :meth:`Path.open` on this path and return the result. If the file
   doesn't exist, ``None`` is returned instead.


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
