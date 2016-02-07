.. Copyright 2015 Peter K. G. Williams <peter@newton.cx> and collaborators.
   This file licensed under the Creative Commons Attribution-ShareAlike 3.0
   Unported License (CC-BY-SA).

Convenient file input and output (:mod:`pwkit.io`)
========================================================================

.. automodule:: pwkit.io
   :synopsis: basic utilities for file input and output

.. currentmodule:: pwkit.io


The :class:`Path` object
------------------------------------------------------------------------

.. autoclass:: Path

   .. autosummary::

      absolute
      as_hdf_store
      as_uri
      chmod
      ensure_parent
      exists
      expand
      glob

.. method:: Path.absolute()

   Return an absolute version of the path. Unlike :meth:`resolve`,
   does not normalize the path or resolve symlinks.

.. automethod:: Path.as_hdf_store

.. automethod:: Path.as_uri

   Return the path stringified as a ``file:///`` URI.

.. method:: Path.chmod(mode)

   Change the mode of the named path. Remember to use octal
   ``0o755`` notation!

.. automethod:: Path.ensure_parent
.. method:: Path.exists()

   Returns whether the path exists.

.. automethod:: Path.expand
.. method:: Path.glob(pattern)

   Assuming that the path is a directory, iterate over its
   contents and return sub-paths matching the given shell-style
   glob pattern.

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

   Assuming the path is a directory, generate a sequence of
   sub-paths corresponding to its contents.

.. method:: Path.joinpath(*args)

   Combine this path with several new components. If one of the
   arguments is absolute, all previous components are discarded.

.. automethod:: Path.make_relative
.. method:: Path.match(pattern)

   Test whether this path matches the given shell glob pattern.

.. method:: Path.mkdir(mode=0o777, parents=False)

   Create a directory at this path location. Creates parent
   directories if *parents* is true. Raises :class:`OSError` if
   the path already exists, even if *parents* is true.

.. method:: Path.open(mode='r', buffering=-1, encoding=None, errors=None, newline=None)

   Open the file pointed at by the path and return a :class:`file`
   object. **TODO**: verify whether semantics correspond to
   :func:`io.open` or plain builtin :func:`open`.

.. automethod:: Path.read_lines
.. automethod:: Path.read_fits
.. automethod:: Path.read_fits_bintable
.. automethod:: Path.read_hdf
.. automethod:: Path.read_inifile
.. automethod:: Path.read_numpy_text
.. automethod:: Path.read_pandas
.. automethod:: Path.read_pickle
.. automethod:: Path.read_pickles
.. automethod:: Path.read_tabfile
.. method:: Path.relative_to(*other)

   Return this path as made relative to another path identified by
   *other*. If this is not possible, raise :exc:`ValueError`.

.. automethod:: Path.rellink_to
.. method:: Path.rename(target)

   Rename this path to *target*.

.. method:: Path.resolve()

   Make this path absolute, resolving all symlinks and normalizing.

.. method:: Path.rglob(pattern)

   Recursively yield all files and directories matching the shell
   glob pattern *pattern* below this path.

.. method:: Path.rmdir()

   Delete this path, if it is an empty directory.

.. automethod:: Path.rmtree
.. automethod:: Path.scandir
.. method:: Path.stat()

   Run :func:`os.stat` on the path and return the result.

.. method:: Path.symlink_to(target, target_is_directory=False)

   Make this path a symlink pointing to the given target.

.. method:: Path.touch(mode=0o666, exist_ok=True)

   Create a file at this path with the given mode, if needed.

.. automethod:: Path.try_open
.. automethod:: Path.try_unlink
.. method:: Path.unlink()

   Unlink this file or symbolic link.

.. method:: Path.with_name(name)

   Return a new path with the file name changed.

.. method:: Path.with_suffix(suffix)

   Return a new path with the file suffix changed, or a new suffix added if
   there was none before. *suffix* should start with a ``"."``.

.. automethod:: Path.write_pickle
.. automethod:: Path.write_pickles
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
