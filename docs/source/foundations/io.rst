.. Copyright 2015 Peter K. G. Williams <peter@newton.cx> and collaborators.
   This file licensed under the Creative Commons Attribution-ShareAlike 3.0
   Unported License (CC-BY-SA).

Convenient file input and output (:mod:`pwkit.io`)
========================================================================

.. automodule:: pwkit.io
   :synopsis: basic utilities for file input and output

.. currentmodule:: pwkit.io

The functionality in this module can be grouped into these categories:

 - :ref:`the-path-object`
 - :ref:`unicode-safety`
 - :ref:`other-functions` (generally being superseded by :class:`Path`)


.. _the-path-object:

The :class:`Path` object
------------------------------------------------------------------------

.. autoclass:: Path

   The methods and attributes on :class:`Path` objects fall into several broad categories:

   - :ref:`path-manipulations`
   - :ref:`path-filesystem-interrogation`
   - :ref:`path-filesystem-modifications`
   - :ref:`path-input-and-output`

   Constructors are:

   .. method:: Path(part, *more)

      Returns a new path equivalent to ``os.path.join (part, *more)``, except
      the arguments may be either strings or other :class:`Path` instances.

   .. classmethod:: Path.cwd()

      Returns a new path containing the absolute path of the current working
      directory.

   .. classmethod:: Path.create_tempfile(want='handle', resolution='try_unlink', suffix='', **kwargs)

      Returns a context manager managing the creation and destruction of a
      named temporary file. The operation of this function is exactly like
      that of the bound method :meth:`Path.make_tempfile`, except that instead
      of creating a temporary file with a name similar to an existing path,
      this function creates one with a name selected using the standard
      OS-dependent methods for choosing names of temporary files.

      The ``overwrite`` resolution is not allowed here since there is no
      original path to overwrite.

      Note that by default the returned context manager returns a file-like
      object and not an actual :class:`Path` instance; use ``want="path"`` to
      get a :class:`Path`.


.. _path-manipulations:

Manipulating and dissecting paths
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Child paths can be created by using the division operator, that is::

  parent = Path ('directory')
  child = parent / 'subdirectory'

Combining a relative path with an absolute path in this way will just yield the
absolute path::

  >>> foo = Path ('relative') / Path ('/a/absolute')
  >>> print (foo)
  <<< /a/absolute

Paths should be converted to text by calling :func:`str` or :func:`unicode` on
them.

Instances of :class:`Path` have the following attributes that help you create
new paths or break them into their components:

.. autosummary::

   ~Path.anchor
   ~Path.drive
   ~Path.name
   ~Path.parent
   ~Path.parents
   ~Path.parts
   ~Path.stem
   ~Path.suffix
   ~Path.suffixes

And they have the following related methods:

.. autosummary::

   ~Path.absolute
   ~Path.as_uri
   ~Path.expand
   ~Path.format
   ~Path.get_parent
   ~Path.is_absolute
   ~Path.joinpath
   ~Path.make_relative
   ~Path.relative_to
   ~Path.resolve
   ~Path.with_name
   ~Path.with_suffix

.. rubric:: Detailed descriptions of attributes

.. attribute:: Path.anchor

   The concatenation of :attr:`~Path.drive` and :attr:`~Path.root`.

.. attribute:: Path.drive

   The Windows or network drive of the path. The empty string on POSIX.

.. attribute:: Path.name

   The final path component. The *name* of ``/foo/`` is ``"foo"``. The *name*
   of ``/foo/.`` is ``"foo"`` as well. The *name* of ``/foo/..`` is ``".."``.

.. attribute:: Path.parent

   This path's parent, in a textual sense: the *parent* of ``foo`` is ``.``,
   but the parent of ``.`` is also ``.``. The parent of ``/bar`` is ``/``; the
   parent of ``/`` is also ``/``.

   .. seealso:: :meth:`Path.get_parent`

.. attribute:: Path.parents

   An immutable, indexable sequence of this path’s parents. Here are some
   examples showing the semantics::

     >>> list(Path("/foo/bar").parents)
     <<< [Path("/foo"), Path("/")]
     >>> list(Path("/foo/bar/").parents)
     <<< [Path("/foo"), Path("/")]
     >>> list(Path("/foo/bar/.").parents)
     <<< [Path("/foo"), Path("/")]
     >>> list(Path("/foo/./bar/.").parents)
     <<< [Path("/foo"), Path("/")]
     >>> list(Path("wib/wob").parents)
     <<< [Path("wib"), Path(".")]
     >>> list(Path("wib/../wob/.").parents)
     <<< [Path("wib/.."), Path("wib"), Path(".")]

   .. seealso:: :meth:`Path.get_parent`

.. attribute:: Path.parts

   A tuple of the path components. Examples::

     >>> Path('/a/b').parts
     <<< ('/', 'a', 'b')
     >>> Path('a/b').parts
     <<< ('a', 'b')
     >>> Path('/a/b/').parts
     <<< ('/', 'a', 'b')
     >>> Path('a/b/.').parts
     <<< ('a', 'b')
     >>> Path('/a/../b/./c').parts
     <<< ('/', 'a', '..', 'b', 'c')
     >>> Path('.').parts
     <<< ()
     >>> Path('').parts
     <<< ()

.. attribute:: Path.stem

   The :attr:`name` without its suffix. The stem of ``"foo.tar.gz"`` is
   ``"foo.tar"``. The stem of ``"noext"`` is ``"noext"``. It is an invariant
   that ``name = stem + suffix``.

.. attribute:: Path.suffix

   The suffix of the :attr:`name`, including the period. If there is no
   period, the empty string is returned::

     >>> print (Path("foo.tar.gz").suffix)
     <<< .gz
     >>> print (Path("foo.dir/.").suffix)
     <<< .dir
     >>> print (repr (Path("noextension").suffix))
     <<< ''

.. attribute:: Path.suffixes

   A list of all suffixes on :attr:`name`, including the periods. The suffixes
   of ``"foo.tar.gz"`` are ``[".tar", ".gz"]``. If :attr:`name` contains no
   periods, the empty list is returned.

.. rubric:: Detailed descriptions of methods

.. method:: Path.absolute()

   Return an absolute version of the path. Unlike :meth:`resolve`,
   does not normalize the path or resolve symlinks.

.. method:: Path.as_uri()

   Return the path stringified as a ``file:///`` URI.

.. automethod:: Path.expand
.. automethod:: Path.format
.. automethod:: Path.get_parent

.. method:: Path.is_absolute()

   Returns whether the path is absolute.

.. method:: Path.joinpath(*args)

   Combine this path with several new components. If one of the
   arguments is absolute, all previous components are discarded.

.. automethod:: Path.make_relative

.. method:: Path.relative_to(*other)

   Return this path as made relative to another path identified by
   *other*. If this is not possible, raise :exc:`ValueError`.

.. method:: Path.resolve()

   Make this path absolute, resolving all symlinks and normalizing away
   ``".."`` and ``"."`` components. The path must exist for this function to
   work.

.. method:: Path.with_name(name)

   Return a new path with the file name changed.

.. method:: Path.with_suffix(suffix)

   Return a new path with the file :attr:`suffix` changed, or a new suffix
   added if there was none before. *suffix* must start with a ``"."``. The
   semantics of the :attr:`suffix` attribute are maintained, so::

     >>> print (Path ('foo.tar.gz').with_suffix ('.new'))
     <<< foo.tar.new


.. _path-filesystem-interrogation:

Filesystem interrogation
~~~~~~~~~~~~~~~~~~~~~~~~

These methods probe the actual filesystem to test whether the given path, for
example, is a directory; but they do not modify the filesystem.

.. autosummary::
   ~Path.exists
   ~Path.glob
   ~Path.is_block_device
   ~Path.is_char_device
   ~Path.is_dir
   ~Path.is_fifo
   ~Path.is_file
   ~Path.is_socket
   ~Path.is_symlink
   ~Path.iterdir
   ~Path.match
   ~Path.readlink
   ~Path.rglob
   ~Path.scandir
   ~Path.stat

.. rubric:: Detailed descriptions

.. method:: Path.exists()

   Returns whether the path exists.

.. method:: Path.glob(pattern)

   Assuming that the path is a directory, iterate over its
   contents and return sub-paths matching the given shell-style
   glob pattern.

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

.. automethod:: Path.iterdir

   Assuming the path is a directory, generate a sequence of
   sub-paths corresponding to its contents.

.. method:: Path.match(pattern)

   Test whether this path matches the given shell glob pattern.

.. automethod:: Path.readlink

.. method:: Path.rglob(pattern)

   Recursively yield all files and directories matching the shell
   glob pattern *pattern* below this path.

.. automethod:: Path.scandir

.. method:: Path.stat()

   Run :func:`os.stat` on the path and return the result.


.. _path-filesystem-modifications:

Filesystem modifications
~~~~~~~~~~~~~~~~~~~~~~~~

These functions actually modify the filesystem.

.. autosummary::

   ~Path.chmod
   ~Path.copy_to
   ~Path.ensure_dir
   ~Path.ensure_parent
   ~Path.make_tempfile
   ~Path.mkdir
   ~Path.rellink_to
   ~Path.rename
   ~Path.rmdir
   ~Path.rmtree
   ~Path.symlink_to
   ~Path.touch
   ~Path.unlink
   ~Path.try_unlink

.. rubric:: Detailed descriptions

.. method:: Path.chmod(mode)

   Change the mode of the named path. Remember to use octal
   ``0o755`` notation!

.. automethod:: Path.copy_to
.. automethod:: Path.ensure_dir
.. automethod:: Path.ensure_parent
.. automethod:: Path.make_tempfile

.. method:: Path.mkdir(mode=0o777, parents=False)

   Create a directory at this path location. Creates parent directories if
   *parents* is true. Raises :class:`OSError` if the path already exists, even
   if *parents* is true.

.. automethod:: Path.rellink_to

.. method:: Path.rename(target)

   Rename this path to *target*.

.. method:: Path.rmdir()

   Delete this path, if it is an empty directory.

.. automethod:: Path.rmtree

.. method:: Path.symlink_to(target, target_is_directory=False)

   Make this path a symlink pointing to the given target.

.. method:: Path.touch(mode=0o666, exist_ok=True)

   Create a file at this path with the given mode, if needed.

.. method:: Path.unlink()

   Unlink this file or symbolic link.

.. automethod:: Path.try_unlink


.. _path-input-and-output:

Data input and output
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~Path.open
   ~Path.try_open
   ~Path.as_hdf_store
   ~Path.read_astropy_ascii
   ~Path.read_fits
   ~Path.read_fits_bintable
   ~Path.read_hdf
   ~Path.read_inifile
   ~Path.read_json
   ~Path.read_lines
   ~Path.read_numpy
   ~Path.read_numpy_text
   ~Path.read_pandas
   ~Path.read_pickle
   ~Path.read_pickles
   ~Path.read_tabfile
   ~Path.read_text
   ~Path.read_toml
   ~Path.read_yaml
   ~Path.write_pickle
   ~Path.write_pickles
   ~Path.write_yaml

.. rubric:: Detailed descriptions

.. method:: Path.open(mode='r', buffering=-1, encoding=None, errors=None, newline=None)

   Open the file pointed at by the path and return a :class:`file` object.
   This delegates to the modern :func:`io.open` function, not the global
   builtin :func:`open`.

.. automethod:: Path.try_open
.. automethod:: Path.as_hdf_store
.. automethod:: Path.read_astropy_ascii
.. automethod:: Path.read_fits
.. automethod:: Path.read_fits_bintable
.. automethod:: Path.read_hdf
.. automethod:: Path.read_inifile
.. automethod:: Path.read_json
.. automethod:: Path.read_lines
.. automethod:: Path.read_numpy
.. automethod:: Path.read_numpy_text
.. automethod:: Path.read_pandas
.. automethod:: Path.read_pickle
.. automethod:: Path.read_pickles
.. automethod:: Path.read_tabfile
.. automethod:: Path.read_text
.. automethod:: Path.read_toml
.. automethod:: Path.read_yaml
.. automethod:: Path.write_pickle
.. automethod:: Path.write_pickles
.. automethod:: Path.write_yaml



.. _unicode-safety:

Functions helping with Unicode safety
------------------------------------------------------------------------

.. autosummary::
   get_stdout_bytes
   get_stderr_bytes

.. autofunction:: get_stdout_bytes
.. autofunction:: get_stderr_bytes


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
