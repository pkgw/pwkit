.. Copyright 2015 Peter K. G. Williams <peter@newton.cx> and collaborators.
   This file licensed under the Creative Commons Attribution-ShareAlike 3.0
   Unported License (CC-BY-SA).

Core utilities (:mod:`pwkit`)
========================================================================

.. automodule:: pwkit
   :synopsis: core abstractions for :mod:`pwkit`

.. currentmodule:: pwkit

The toplevel :mod:`pwkit` module includes a few basic abstractions that show
up throughout the rest of the codebase. These include:

 - :ref:`holder`
 - :ref:`exception-utilities`
 - :ref:`py3-abstractions`


.. _holder:

The :class:`Holder` namespace object
------------------------------------------------------------------------

:class:`Holder` is a “namespace object” that primarily exists so that you can
fill it with named attributes however you want. It’s essentially like a plain
:class:`dict`, but you can write the convenient form ``myholder.xcoord``
instead of ``mydict['xcoord']``. It has useful methods like
:meth:`~Holder.set` and :meth:`~Holder.to_pretty` also.

.. autoclass:: Holder

   .. autosummary::
      get
      set
      set_one
      has
      copy
      to_dict
      to_pretty

   Iterating over a :class:`Holder` yields its contents in the form of a
   sequence of ``(name, value)`` tuples. The stringification of a
   :class:`Holder` returns its representation in a dict-like format.
   :class:`Holder` objects implement ``__contains__`` so that boolean tests
   such as ``"myprop" in myholder`` act sensibly.

   .. automethod:: get
   .. automethod:: set
   .. automethod:: set_one
   .. automethod:: has
   .. automethod:: copy
   .. automethod:: to_dict
   .. automethod:: to_pretty

.. decorator:: Holder

   The :class:`Holder` class may also be used as a decorator on a class
   definition to transform its contents into a Holder instance. Writing::

     @Holder
     class mydata ():
         a = 1
         b = 'hello'

   creates a Holder instance named ``mydata`` containing names ``a`` and
   ``b``. This can be a convenient way to populate one-off data structures.


.. _exception-utilities:

Utilities for exceptions
------------------------------------------------------------------------

.. autoclass:: PKError

.. autofunction:: reraise_context



.. _py3-abstractions:

Abstractions between Python versions 2 and 3
------------------------------------------------------------------------

The toplevel :mod:`pwkit` module defines the following variables as a holdover
from the times when it was concerned with compatibility between Python 2 and
Python 3:

- :data:`binary_type`
- :data:`text_type`

.. function:: unicode_to_str(s)

   A function for implementing the ``__str__`` method of classes, the meaning
   of which differs between Python versions 2 and 3. In all cases, you should
   implement ``__unicode__`` on your classes. Setting the ``__str__`` property
   of a class to :func:`unicode_to_str` will cause it to Do The Right Thing™,
   which means returning the UTF-8 encoded version of its Unicode expression
   in Python 2, or returning the Unicode expression directly in Python 3::

     import pwkit

     class MyClass (object):
         def __unicode__ (self):
	     return u'my value'

	 __str__ = pwkit.unicode_to_str
