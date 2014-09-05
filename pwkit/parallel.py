# -*- mode: python; coding: utf-8 -*-
# Copyright 2014 Peter Williams <peter@newton.cx> and collaborators.
# Licensed under the MIT License.

"""parallel - Tools for parallel processing.

Functions:

make_parallel_helper - Return an object that sets up parallel computations.

See the make_parallel_helper() documentation for more details, but in short:

```
from pwkit.parallel import make_parallel_helper

def my_parallelizable_function (arg1, arg1, parallel=True):
    phelp = make_parallel_helper (parallel)
    ...

    with phelp.get_map () as map:
       results1 = map (my_subfunc1, subargs1)
       ...
       results2 = map (my_subfunc2, subargs2)

    ... do stuff with results1 and results2 ...
```

Setting `parallel=True` will use all cores. `parallel=0.5` will use about half
your machine. `parallel=False` will use serial processing. The helper must be
used as a context manager (the "with" statement) because the parallel
computation may involve creating and destroying heavyweight resources (namely,
child processes).

"""
from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = (b'make_parallel_helper').split ()

import functools, signal
from multiprocessing.pool import Pool
from multiprocessing import TimeoutError


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

    processes   - The number of worker processes to use; defaults to the number of CPUs.
    initializer - Either None, or a callable that will be invoked by each worker
                  process when it starts.
    initargs    - Arguments for `initializer`.
    kwargs      - Extra arguments. Python 2.7 supports a `maxtasksperchild` parameter.

    Python's multiprocessing.Pool class doesn't interact well with
    KeyboardInterrupt signals, as documented in places such as:

    * `<http://stackoverflow.com/questions/1408356/>`_
    * `<http://stackoverflow.com/questions/11312525/>`_
    * `<http://noswap.com/blog/python-multiprocessing-keyboardinterrupt>`_

    Various workarounds have been shared. Here, we adapt the one proposed in
    the last link above, by John Reese, and shared as

    * `<https://github.com/jreese/multiprocessing-keyboardinterrupt/>`_

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

        func     - The function to apply to the items.
        iterable - An iterable of items that will have `func` applied to them.

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

    ParallelHelper objects should be obtained by calling make_helper(), not
    direct construction, unless you have special needs. See the documentation
    of that function for an example of the general usage pattern.

    Once you have a ParallelHelper instance, usage should be something like:

    ```python
    with phelp.get_map () as map:
        results_arr = map (my_function, my_args)
    ```

    """
    def get_map (self):
        raise NotImplementedError ('get_map() not available')


class SerialHelper (ParallelHelper):
    """A `ParallelHelper` that actually does serial processing."""

    class VacuousContextManager (object):
        def __init__ (self, value):
            self.value = value
        def __enter__ (self):
            return self.value
        def __exit__ (self, etype, evalue, etb):
            return False

    def __init__ (self, chunksize=None):
        # We accept and discard some of the multiprocessing kwargs that turn
        # into noops so that we can present a uniform API.
        pass

    def get_map (self):
        return self.VacuousContextManager (map)


class MultiprocessingPoolHelper (ParallelHelper):
    """A `ParallelHelper` that parallelizes computations using Python's
    `multiprocessing.Pool` with a configurable number of processes. Actually,
    we use a wrapped version of `Pool` that handles KeyboardInterrupts more
    helpfully.

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


def make_parallel_helper (parallel_arg, **kwargs):
    """Return a `ParallelHelper` object that can be used for easy parallelization
    of computations. `parallel_arg` is an object that lets the caller easily
    specify the kind of parallelization they are interested in. Allowed values
    are:

    False                     - Serial processing only.
    True                      - Parallel processing using all available cores.
    1                         - Equivalent to `False`.
    (other positive integer)  - Parallel processing using the specified number of cores.
    x, 0 < x < 1              - Parallel processing using about (x*N) cores,
                                where N is the total number of cores in the
                                system. Note that the meanings of `0.99` and
                                `1` as arguments are very different.
    (ParallelHelper instance) - Returns the instance.

    The **kwargs are passed on to the appropriate ParallelHelper constructor,
    if the caller wants to do something tricky.

    Expected usage is:

    ```python

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
    ```

    This means that `my_parallelizable_function` doesn't have to worry about
    all of the various fancy things the caller might want to do in terms of
    special parallel magic. Note that `sub_operation` must be defined in a
    stand-alone fashion because of the way Python's `multiprocessing` module
    works.

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
