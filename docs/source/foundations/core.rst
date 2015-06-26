.. Copyright 2015 Peter K. G. Williams <peter@newton.cx> and collaborators.
   This file licensed under the Creative Commons Attribution-ShareAlike 3.0
   Unported License (CC-BY-SA).

Core utilities (:mod:`pwkit`)
========================================================================

.. module:: pwkit
   :synopsis: core abstractions for :mod:`pwkit`

The toplevel :mod:`pwkit` module includes a few basic abstractions that show
up throughout the rest of the codebase. These include:

 - :ref:`holder`
 - :ref:`exception-utilities`
 - :ref:`py3-abstractions`


.. _holder:

The :class:`Holder` namespace object
------------------------------------------------------------------------

:class:`Holder` is a “namespace object” that primarily exists so that you can
fill it with named attributes however you want. This is convenient for, say,
implementing functions that return complex data in a way that's amenable to
future extension.

.. class:: Holder(__decorating=None, **kwargs)

   Create a new :class:`Holder`. Any keyword arguments will be assigned as
   properties on the object itself, for instance, ``o = Holder (foo=1)``
   yields an object such that ``o.foo`` is 1.

   The *__decorating* keyword is used to implement the :class:`Holder`
   decorator functionality, described below.

While the :class:`Holder` is primarily meant for bare-bones namespace
management, it does provide several convenience functions: :meth:`Holder.get`,
:meth:`Holder.set`, :meth:`Holder.set_one`, :meth:`Holder.has`,
:meth:`Holder.copy`, :meth:`Holder.to_dict`, and :meth:`Holder.to_pretty`.


.. method:: Holder.__unicode__()

   Placeholder.


.. method:: Holder.__str__()

   Placeholder.


.. method:: Holder.__repr__()

   Placeholder.


.. method:: Holder.__iter__()

   Placeholder.


.. method:: Holder.__contains__(key)

   Placeholder.


.. method:: Holder.get(name, defval=None)

   Placeholder.


.. method:: Holder.set(**kwargs)

   Placeholder.


.. method:: Holder.set_one(name, value)

   Placeholder.


.. method:: Holder.has(name)

   Placeholder.


.. method:: Holder.copy()

   Placeholder.


.. method:: Holder.to_dict()

   Placeholder.


.. method:: Holder.to_pretty(format='str')

   Placeholder.


.. decorator:: Holder

   Placeholder decorator documentation.



.. _exception-utilities:

Utilities for exceptions
------------------------------------------------------------------------

.. exception:: PKError (fmt, *args):

   Placeholder.


.. function:: reraise_context (fmt, *args):

   Placeholder.



.. _py3-abstractions:

Abstractions between Python versions 2 and 3
------------------------------------------------------------------------

.. data:: text_type

   The builtin class corresponding to text in this Python interpreter: either
   :class:`unicode` in Python 2, or :class:`str` in Python 3.

.. data:: binary_type

   The builtin class corresponding to binary data in this Python interpreter:
   either :class:`str` in Python 2, or :class:`bytes` in Python 3.

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
