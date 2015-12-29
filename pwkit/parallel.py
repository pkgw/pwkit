# -*- mode: python; coding: utf-8 -*-
# Copyright 2014 Peter Williams <peter@newton.cx> and collaborators.
# Licensed under the MIT License.

"""A framework making it easy to write functions that can perform computations
in parallel.

Use this framework if you are writing a function that you would like to
perform some of its work in parallel, using multiple CPUs at once. First, you
must design the parallel part of the function's operation to be implementable
in terms of the standard library :func:`map` function. Then, give your
function an optional ``parallel=True`` keyword argument and use the
:func:`make_parallel_helper` function from this module like so::

  from pwkit.parallel import make_parallel_helper

  def my_parallelizable_function (arg1, arg1, parallel=True):
      # Get a "parallel helper" object that can provide us with a parallelized
      # "map" function. The caller specifies how the parallelization is done;
      # we don't have to know the details.
      phelp = make_parallel_helper (parallel)
      ...

      # When used as a context manager, the helper provides a function that
      # acts like the standard library function "map", except it may
      # parallelize its operation.
      with phelp.get_map () as map:
         results1 = map (my_subfunc1, subargs1)
         ...
         results2 = map (my_subfunc2, subargs2)

      ... do stuff with results1 and results2 ...

Passing ``parallel=True`` to a function defined this way will cause it to
parallelize ``map`` calls across all cores. Passing ``parallel=0.5`` will
cause it to use about half your machine. Passing ``parallel=False`` will cause
it to use serial processing. The helper must be used as a context manager (via
the ``with`` statement) because the parallel computation may involve creating
and destroying heavyweight resources (namely, child processes).

Along with standard :meth:`ParallelHelper.get_map`, :class:`ParallelHelper`
instances support a "partially-Pickling" `map`-like function
:meth:`ParallelHelper.get_ppmap` that works around Pickle-related limitations
in the :mod:`multiprocessing` library.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = str ('make_parallel_helper').split ()

import functools, signal
from multiprocessing.pool import Pool
from multiprocessing import Process, Queue, TimeoutError
from six.moves import range


def _initializer_wrapper (actual_initializer, *rest):
    """We ignore SIGINT. It's up to our parent to kill us in the typical condition
    of this arising from ``^C`` on a terminal. If someone is manually killing
    us with that signal, well... nothing will happen.

    """
    signal.signal (signal.SIGINT, signal.SIG_IGN)
    if actual_initializer is not None:
        actual_initializer (*rest)


class InterruptiblePool (Pool):
    """A modified version of `multiprocessing.pool.Pool` that has better
    behavior with regard to KeyboardInterrupts in the `map` method. Parameters:

    processes
      The number of worker processes to use; defaults to the number of CPUs.
    initializer
      Either None, or a callable that will be invoked by each worker
      process when it starts.
    initargs
      Arguments for `initializer`.
    kwargs
      Extra arguments. Python 2.7 supports a `maxtasksperchild` parameter.

    Python's multiprocessing.Pool class doesn't interact well with
    KeyboardInterrupt signals, as documented in places such as:

    - `<http://stackoverflow.com/questions/1408356/>`_
    - `<http://stackoverflow.com/questions/11312525/>`_
    - `<http://noswap.com/blog/python-multiprocessing-keyboardinterrupt>`_

    Various workarounds have been shared. Here, we adapt the one proposed in
    the last link above, by John Reese, and shared as

    - `<https://github.com/jreese/multiprocessing-keyboardinterrupt/>`_

    This version is a drop-in replacement for multiprocessing.Pool ... as long
    as the map() method is the only one that needs to be interrupt-friendly.

    """
    wait_timeout = 3600

    def __init__ (self, processes=None, initializer=None, initargs=(), **kwargs):
        new_initializer = functools.partial (_initializer_wrapper, initializer)
        super (InterruptiblePool, self).__init__ (processes, new_initializer,
                                                  initargs, **kwargs)


    def map (self, func, iterable, chunksize=None):
        """Equivalent of `map` built-in, without swallowing KeyboardInterrupt.

        func
          The function to apply to the items.
        iterable
          An iterable of items that will have `func` applied to them.

        """
        # The key magic is that we must call r.get() with a timeout, because a
        # Condition.wait() without a timeout swallows KeyboardInterrupts.
        r = self.map_async (func, iterable, chunksize)

        while True:
            try:
                return r.get (self.wait_timeout)
            except TimeoutError:
                pass
            except KeyboardInterrupt:
                self.terminate ()
                self.join ()
                raise
            # Other exceptions propagate up.


class ParallelHelper (object):
    """Object that helps genericize the setup needed for parallel computations.
    Each method returns a context manager that wraps up any resource
    allocation and deallocation that may need to occur to make the
    parallelization happen under the hood.

    :class:`ParallelHelper` objects should be obtained by calling
    :func:`make_parallel_helper`, not direct construction, unless you have
    special needs. See the documentation of that function for an example of
    the general usage pattern.

    Once you have a :class:`ParallelHelper` instance, usage should be
    something like::

        with phelp.get_map () as map:
            results_arr = map (my_function, my_args)

    The partially-Pickling map works around a limitation in the
    multiprocessing library. This library spawns subprocesses and executes
    parallel tasks by sending them to the subprocesses, which means that the
    data describing the task must be pickle-able. There are hacks so that you
    can pass functions defined in the global namespace but they're pretty much
    useless in production code. The "partially-Pickling map" works around this
    by using a different method that allows some arguments to the map
    operation to avoid being pickled. (Instead, they are directly inherited by
    :func:`os.fork`-ed subprocesses.) See the docs for :func:`serial_ppmap` for
    usage information.

    """
    def get_map (self):
        """Get a *context manager* that yields a function with the same call signature
        as the standard library function :func:`map`. Its results are the
        same, but it may evaluate the mapped function in parallel across
        multiple threads or processes --- the calling function should not have
        to particularly care about the details. Example usage is::

            with phelp.get_map () as map:
                results_arr = map (my_function, my_args)

        The passed function and its arguments must be Pickle-able. The alternate
        method :meth:`get_ppmap` relaxes this restriction somewhat.

        """
        raise NotImplementedError ('get_map() not available')

    def get_ppmap (self):
        """Get a *context manager* that yields a "partially-pickling map function". It
        can be used to perform a parallelized :func:`map` operation with some
        un-pickle-able arguments.

        The yielded function has the signature of :func:`serial_ppmap`. Its
        behavior is functionally equivalent to the following code, except that
        the calls to ``func`` may happen in parallel::

            def ppmap (func, fixed_arg, var_arg_iter):
                return [func (i, fixed_arg, x) for i, x in enumerate (var_arg_iter)]

        The arguments to the ``ppmap`` function are:

        *func*
          A callable taking three arguments and returning a Pickle-able value.
        *fixed_arg*
          Any value, even one that is not pickle-able.
        *var_arg_iter*
          An iterable that generates Pickle-able values.

        The arguments to your ``func`` function, which actually does the
        interesting computations, are:

        *index*
          The 0-based index number of the item being processed; often this can
          be ignored.
        *fixed_arg*
          The same *fixed_arg* that was passed to ``ppmap``.
        *var_arg*
          The *index*'th item in the *var_arg_iter* iterable passed to
          ``ppmap``.

        This variant of the standard :func:`map` function exists to allow the
        parallel-processing system to work around :mod:`pickle`-related
        limitations in the :mod:`multiprocessing` library.

        """
        raise NotImplementedError ('get_ppmap() not available')


class VacuousContextManager (object):
    """A context manager that just returns a static value and doesn't do anything
    clever with exceptions.

    """
    def __init__ (self, value):
        self.value = value
    def __enter__ (self):
        return self.value
    def __exit__ (self, etype, evalue, etb):
        return False


def serial_ppmap (func, fixed_arg, var_arg_iter):
    """A serial implementation of the "partially-pickling map" function returned
    by the :meth:`ParallelHelper.get_ppmap` interface. Its arguments are:

    *func*
      A callable taking three arguments and returning a Pickle-able value.
    *fixed_arg*
      Any value, even one that is not pickle-able.
    *var_arg_iter*
      An iterable that generates Pickle-able values.

    The functionality is::

        def serial_ppmap (func, fixed_arg, var_arg_iter):
            return [func (i, fixed_arg, x) for i, x in enumerate (var_arg_iter)]

    Therefore the arguments to your ``func`` function, which actually does the
    interesting computations, are:

    *index*
      The 0-based index number of the item being processed; often this can
      be ignored.
    *fixed_arg*
      The same *fixed_arg* that was passed to ``ppmap``.
    *var_arg*
      The *index*'th item in the *var_arg_iter* iterable passed to
      ``ppmap``.

    """
    return [func (i, fixed_arg, x) for i, x in enumerate (var_arg_iter)]


class SerialHelper (ParallelHelper):
    """A :class:`ParallelHelper` that actually does serial processing."""

    def __init__ (self, chunksize=None):
        # We accept and discard some of the multiprocessing kwargs that turn
        # into noops so that we can present a uniform API.
        pass

    def get_map (self):
        return VacuousContextManager (map)

    def get_ppmap (self):
        return VacuousContextManager (serial_ppmap)


def multiprocessing_ppmap_worker (in_queue, out_queue, func, fixed_arg):
    """Worker for the :mod:`multiprocessing` ppmap implementation. Strongly
    derived from code posted on StackExchange by "klaus se":
    `<http://stackoverflow.com/a/16071616/3760486>`_.

    """
    while True:
        i, var_arg = in_queue.get ()
        if i is None:
            break
        out_queue.put ((i, func (i, fixed_arg, var_arg)))


class MultiprocessingPoolHelper (ParallelHelper):
    """A :class:`ParallelHelper` that parallelizes computations using Python's
    :class:`multiprocessing.Pool` with a configurable number of processes.
    Actually, we use a wrapped version of :class:`multiprocessing.Pool` that
    handles :exc:`KeyboardInterrupt` exceptions more helpfully.

    """
    class InterruptiblePoolContextManager (object):
        def __init__ (self, methodname, methodkwargs={}, **kwargs):
            self.methodname = methodname
            self.methodkwargs = methodkwargs
            self.kwargs = kwargs

        def __enter__ (self):
            from functools import partial
            self.pool = InterruptiblePool (**self.kwargs)
            func = getattr (self.pool, self.methodname)
            return partial (func, **self.methodkwargs)

        def __exit__ (self, etype, evalue, etb):
            self.pool.terminate ()
            self.pool.join ()
            return False


    def __init__ (self, chunksize=None, **pool_kwargs):
        self.chunksize = chunksize
        self.pool_kwargs = pool_kwargs

    def get_map (self):
        return self.InterruptiblePoolContextManager ('map',
                                                     {'chunksize': self.chunksize},
                                                     **self.pool_kwargs)


    def _ppmap (self, func, fixed_arg, var_arg_iter):
        """The multiprocessing implementation of the partially-Pickling "ppmap"
        function. This doesn't use a Pool like map() does, because the whole
        problem is that Pool chokes on un-Pickle-able values. Strongly derived
        from code posted on StackExchange by "klaus se":
        `<http://stackoverflow.com/a/16071616/3760486>`_.

        This implementation could definitely be improved -- that's basically
        what the Pool class is all about -- but this gets us off the ground
        for those cases where the Pickle limitation is important.

        XXX This deadlocks if a child process crashes!!! XXX
        """
        n_procs = self.pool_kwargs.get ('processes')
        if n_procs is None:
            # Logic copied from multiprocessing.pool.Pool.__init__()
            try:
                from multiprocessing import cpu_count
                n_procs = cpu_count ()
            except NotImplementedError:
                n_procs = 1

        in_queue = Queue (1)
        out_queue = Queue ()
        procs = [Process (target=multiprocessing_ppmap_worker,
                          args=(in_queue, out_queue, func, fixed_arg))
                 for _ in range (n_procs)]

        for p in procs:
            p.daemon = True
            p.start ()

        i = -1

        for i, var_arg in enumerate (var_arg_iter):
            in_queue.put ((i, var_arg))

        n_items = i + 1
        result = [None] * n_items

        for p in procs:
            in_queue.put ((None, None))

        for _ in range (n_items):
            i, value = out_queue.get ()
            result[i] = value

        for p in procs:
            p.join ()

        return result

    def get_ppmap (self):
        return VacuousContextManager (self._ppmap)


def make_parallel_helper (parallel_arg, **kwargs):
    """Return a :class:`ParallelHelper` object that can be used for easy
    parallelization of computations. *parallel_arg* is an object that lets the
    caller easily specify the kind of parallelization they are interested in.
    Allowed values are:

    False
      Serial processing only.
    True
      Parallel processing using all available cores.
    1
      Equivalent to ``False``.
    Other positive integer
      Parallel processing using the specified number of cores.
    x, 0 < x < 1
      Parallel processing using about ``x * N`` cores, where N is the total
      number of cores in the system. Note that the meanings of ``0.99`` and ``1``
      as arguments are very different.
    :class:`ParallelHelper` instance
      Returns the instance.

    The ``**kwargs`` are passed on to the appropriate :class:`ParallelHelper`
    constructor, if the caller wants to do something tricky.

    Expected usage is::

        from pwkit.parallel import make_parallel_helper

        def sub_operation (arg):
            ... do some computation ...
            return result

        def my_parallelizable_function (arg1, arg2, parallel=True):
            phelp = make_parallel_helper (parallel)

            with phelp.get_map () as map:
                op_results = map (sub_operation, args)

            ... reduce "op_results" in some way ...
            return final_result

    This means that ``my_parallelizable_function`` doesn't have to worry about
    all of the various fancy things the caller might want to do in terms of
    special parallel magic.

    Note that ``sub_operation`` above must be defined in a stand-alone fashion
    because of the way Python's :mod:`multiprocessing` module works. This can
    be worked around somewhat with the special
    :meth:`ParallelHelper.get_ppmap` variant. This returns a
    "partially-Pickling" map operation --- with a different calling signature
    --- that allows un-Pickle-able values to be used. See the documentation
    for :func:`serial_ppmap` for usage information.

    """
    if parallel_arg is True: # note: (True == 1) is True
        return MultiprocessingPoolHelper (**kwargs)

    if parallel_arg is False or parallel_arg == 1:
        return SerialHelper (**kwargs)

    if parallel_arg > 0 and parallel_arg < 1:
        from multiprocessing import cpu_count
        n = int (round (parallel_arg * cpu_count ()))
        return MultiprocessingPoolHelper (processes=n, **kwargs)

    if isinstance (parallel_arg, ParallelHelper):
        return parallel_arg

    if isinstance (parallel_arg, (int, long)):
        return MultiprocessingPoolHelper (processes=parallel_arg, **kwargs)

    raise ValueError ('don\'t understand make_parallel_helper() argument %r'
                      % parallel_arg)
