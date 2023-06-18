# -*- mode: python; coding: utf-8 -*-
# Copyright 2012-2018 Peter Williams <peter@newton.cx> and collaborators.
# Licensed under the MIT License.

"""Model data with least-squares fitting

This module provides tools for fitting models to data using least-squares
optimization.

"""
from __future__ import absolute_import, division, print_function

__all__ = "ModelBase Model ComposedModel PolynomialModel ScaleModel".split()

import numpy as np

try:
    # numpy 1.7
    import numpy.polynomial.polynomial as npoly
except ImportError:
    import numpy.polynomial as npoly


class Parameter(object):
    """Information about a parameter in a least-squares model.

    These data may only be obtained after solving least-squares problem. These
    objects reference information from their parent objects, so changing the
    parent will alter the apparent contents of these objects.

    """

    def __init__(self, owner, index):
        self._owner = owner
        self._index = index

    def __repr__(self):
        return '<Parameter "%s" (#%d) of %s>' % (self.name, self._index, self._owner)

    @property
    def index(self):  # make this read-only
        "The parameter's index in the Model's arrays."
        return self._index

    @property
    def name(self):
        "The parameter's name."
        return self._owner.pnames[self._index]

    @property
    def value(self):
        "The parameter's value."
        return self._owner.params[self._index]

    @property
    def uncert(self):
        "The uncertainty in :attr:`value`."
        return self._owner.puncerts[self._index]

    @property
    def uval(self):
        "Accesses :attr:`value` and :attr:`uncert` as a :class:`pwkit.msmt.Uval`."
        from .msmt import Uval

        return Uval.from_norm(self.value, self.uncert)


class ModelBase(object):
    """An abstract base class holding data and a model for least-squares fitting.

    The models implemented in this module all derive from this class and so
    inherit the attributes and methods described below.

    A :class:`Parameter` data structure may be obtained by indexing this
    object with either the parameter's numerical index or its name. I.e.::

      m = Model(...).solve(...)
      p = m['slope']
      print(p.name, p.value, p.uncert, p.uval)

    """

    data = None
    "The data to be modeled; an *n*-dimensional Numpy array."

    invsigma = None
    "Data weights: 1/σ for each data point."

    params = None
    "After fitting, a Numpy ndarray of solved model parameters."

    puncerts = None
    "After fitting, a Numpy ndarray of 1σ uncertainties on the model parameters."

    pnames = None
    "A list of textual names for the parameters."

    covar = None
    """After fitting, the variance-covariance matrix representing the parameter
    uncertainties.

    """
    mfunc = None
    """After fitting, a callable function evaluating the model fixed at best params.

    The resulting function may or may not take arguments depending on the particular
    kind of model being evaluated.

    """
    mdata = None
    "After fitting, the modeled data at the best parameters."

    chisq = None
    "After fitting, the χ² of the fit."

    rchisq = None
    "After fitting, the reduced χ² of the fit, or None if there are no degrees of freedom."

    resids = None
    "After fitting, the residuals: ``resids = data - mdata``."

    def __init__(self, data, invsigma=None):
        self.set_data(data, invsigma)

    def set_data(self, data, invsigma=None):
        """Set the data to be modeled.

        Returns *self*.

        """
        self.data = np.array(data, dtype=float, ndmin=1)

        if invsigma is None:
            self.invsigma = np.ones(self.data.shape)
        else:
            i = np.array(invsigma, dtype=float)
            self.invsigma = np.broadcast_arrays(self.data, i)[
                1
            ]  # allow scalar invsigma

        if self.invsigma.shape != self.data.shape:
            raise ValueError(
                "data values and inverse-sigma values must have same shape"
            )

        return self

    def print_soln(self):
        """Print information about the model solution."""
        lmax = reduce(max, (len(x) for x in self.pnames), len("r chi sq"))

        if self.puncerts is None:
            for pn, val in zip(self.pnames, self.params):
                print("%s: %14g" % (pn.rjust(lmax), val))
        else:
            for pn, val, err in zip(self.pnames, self.params, self.puncerts):
                frac = abs(100.0 * err / val)
                print("%s: %14g +/- %14g (%.2f%%)" % (pn.rjust(lmax), val, err, frac))

        if self.rchisq is not None:
            print("%s: %14g" % ("r chi sq".rjust(lmax), self.rchisq))
        elif self.chisq is not None:
            print("%s: %14g" % ("chi sq".rjust(lmax), self.chisq))
        else:
            print("%s: unknown/undefined" % ("r chi sq".rjust(lmax)))
        return self

    def make_frozen_func(self, params):
        """Return a data-generating model function frozen at the specified parameters.

        As with the :attr:`mfunc` attribute, the resulting function may or may
        not take arguments depending on the particular kind of model being
        evaluated.

        """
        raise NotImplementedError()

    def __getitem__(self, key):
        if isinstance(key, bytes):
            # If you're not using the unicode_literals __future__, things get
            # annoying really quickly without this.
            key = str(key)

        if isinstance(key, int):
            idx = key
            if idx < 0 or idx >= len(self.pnames):
                raise ValueError("illegal parameter number %d" % key)
        elif isinstance(key, str):
            try:
                idx = self.pnames.index(key)
            except ValueError:
                raise ValueError('no such parameter named "%s"' % key)
        else:
            raise ValueError("illegal parameter key %r" % key)

        return Parameter(self, idx)

    def plot(
        self, modelx, dlines=False, xmin=None, xmax=None, ymin=None, ymax=None, **kwargs
    ):
        """Plot the data and model (requires `omega`).

        This assumes that `data` is 1D and that `mfunc` takes one argument
        that should be treated as the X variable.

        """
        import omega as om

        modelx = np.asarray(modelx)
        if modelx.shape != self.data.shape:
            raise ValueError("modelx and data arrays must have same shape")

        modely = self.mfunc(modelx)
        sigmas = self.invsigma**-1  # TODO: handle invsigma = 0

        vb = om.layout.VBox(2)
        vb.pData = om.quickXYErr(
            modelx, self.data, sigmas, "Data", lines=dlines, **kwargs
        )

        vb[0] = vb.pData
        vb[0].addXY(modelx, modely, "Model")
        vb[0].setYLabel("Y")
        vb[0].rebound(False, True)
        vb[0].setBounds(xmin, xmax, ymin, ymax)

        vb[1] = vb.pResid = om.RectPlot()
        vb[1].defaultField.xaxis = vb[1].defaultField.xaxis
        vb[1].addXYErr(modelx, self.resids, sigmas, None, lines=False)
        vb[1].setLabels("X", "Residuals")
        vb[1].rebound(False, True)
        # ignore Y values since residuals are on different scale:
        vb[1].setBounds(xmin, xmax)

        vb.setWeight(0, 3)
        return vb

    def show_cov(self):
        "Show the parameter covariance matrix with `pwkit.ndshow_gtk3`."
        # would be nice: labels with parameter names (hard because this is
        # ndshow, not omegaplot)
        from .ndshow_gtk3 import view

        view(self.covar, title="Covariance Matrix")

    def show_corr(self):
        "Show the parameter correlation matrix with `pwkit.ndshow_gtk3`."
        from .ndshow_gtk3 import view

        d = np.diag(self.covar) ** -0.5
        corr = self.covar * d[np.newaxis, :] * d[:, np.newaxis]
        view(corr, title="Correlation Matrix")


class Model(ModelBase):
    """Models data with a generic nonlinear optimizer

    Basic usage is::

      def func(p1, p2, x):
          simulated_data = p1 * x + p2
          return simulated_data

      x = [1, 2, 3]
      data = [10, 14, 15.8]
      mdl = Model(func, data, args=(x,)).solve(guess).print_soln()

    The :class:`Model` constructor can take an optional argument ``invsigma``
    after ``data``; it specifies *inverse sigmas*, **not** inverse *variances*
    (the usual statistical weights), for the data points. Since most
    applications deal in sigmas, take care to write::

      m = Model(func, data, 1. / uncerts) # right!

    not::

      m = Model(func, data, uncerts) # WRONG

    If you have zero uncertainty on a measurement, you must wind a way to
    express that constraint without including that measurement as part of the
    ``data`` vector.

    """

    lm_prob = None
    """A :class:`pwkit.lmmin.Problem` instance describing the problem to be solved.

    After setting up the data-generating function, you can access this item to
    tune the solver.

    """

    def __init__(self, simple_func, data, invsigma=None, args=()):
        if simple_func is not None:
            self.set_simple_func(simple_func, args)
        if data is not None:
            self.set_data(data, invsigma)

    def set_func(self, func, pnames, args=()):
        """Set the model function to use an efficient but tedious calling convention.

        The function should obey the following convention::

            def func(param_vec, *args):
                modeled_data = { do something using param_vec }
                return modeled_data

        This function creates the :class:`pwkit.lmmin.Problem` so that the
        caller can futz with it before calling :meth:`solve`, if so desired.

        Returns *self*.

        """
        from .lmmin import Problem

        self.func = func
        self._args = args
        self.pnames = list(pnames)
        self.lm_prob = Problem(len(self.pnames))
        return self

    def set_simple_func(self, func, args=()):
        """Set the model function to use a simple but somewhat inefficient calling
        convention.

        The function should obey the following convention::

            def func(param0, param1, ..., paramN, *args):
                modeled_data = { do something using the parameters }
                return modeled_data

        Returns *self*.

        """
        code = func.__code__
        npar = code.co_argcount - len(args)
        pnames = code.co_varnames[:npar]

        def wrapper(params, *args):
            return func(*(tuple(params) + args))

        return self.set_func(wrapper, pnames, args)

    def make_frozen_func(self, params):
        """Returns a model function frozen to the specified parameter values.

        Any remaining arguments are left free and must be provided when the
        function is called.

        For this model, the returned function is the application of
        :func:`functools.partial` to the :attr:`func` property of this object.

        """
        params = np.array(params, dtype=float, ndmin=1)
        from functools import partial

        return partial(self.func, params)

    def solve(self, guess):
        """Solve for the parameters, using an initial guess.

        This uses the Levenberg-Marquardt optimizer described in
        :mod:`pwkit.lmmin`.

        Returns *self*.

        """
        guess = np.array(guess, dtype=float, ndmin=1)
        f = self.func
        args = self._args

        def lmfunc(params, vec):
            vec[:] = f(params, *args).flatten()

        self.lm_prob.set_residual_func(
            self.data.flatten(), self.invsigma.flatten(), lmfunc, None
        )
        self.lm_soln = soln = self.lm_prob.solve(guess)

        self.params = soln.params
        self.puncerts = soln.perror
        self.covar = soln.covar
        self.mfunc = self.make_frozen_func(soln.params)

        # fvec = resids * invsigma = (data - mdata) * invsigma
        self.resids = soln.fvec.reshape(self.data.shape) / self.invsigma
        self.mdata = self.data - self.resids

        # lm_soln.fnorm can be unreliable ("max(fnorm, fnorm1)" branch)
        self.chisq = (self.lm_soln.fvec**2).sum()
        if soln.ndof > 0:
            self.rchisq = self.chisq / soln.ndof

        return self


class PolynomialModel(ModelBase):
    """Least-squares polynomial fit.

    Because this is a very specialized kind of problem, we don't need an
    initial guess to solve, and we can use fast built-in numerical routines.

    The output parameters are named "a0", "a1", ... and are stored in that
    order in PolynomialModel.params[]. We have ``y = sum(x**i * a[i])``, so
    "a2" = "params[2]" is the quadratic term, etc.

    This model does *not* give uncertainties on the derived coefficients. The
    as_nonlinear() method can be use to get a `Model` instance with
    uncertainties.

    Methods:

    as_nonlinear - Return a (lmmin-based) `Model` equivalent to self.

    """

    def __init__(self, maxexponent, x, data, invsigma=None):
        self.maxexponent = maxexponent
        self.x = np.array(x, dtype=float, ndmin=1, copy=False, subok=True)
        self.set_data(data, invsigma)

    def make_frozen_func(self, params):
        return lambda x: npoly.polyval(x, params)

    def solve(self):
        self.pnames = ["a%d" % i for i in range(self.maxexponent + 1)]
        self.params = npoly.polyfit(
            self.x, self.data, self.maxexponent, w=self.invsigma
        )
        self.puncerts = None  # does anything provide this? could farm out to lmmin ...
        self.covar = None
        self.mfunc = self.make_frozen_func(self.params)
        self.mdata = self.mfunc(self.x)
        self.resids = self.data - self.mdata

        self.chisq = ((self.resids * self.invsigma) ** 2).sum()
        if self.x.size > self.maxexponent + 1:
            self.rchisq = self.chisq / (self.x.size - (self.maxexponent + 1))

        return self

    def as_nonlinear(self, params=None):
        """Return a `Model` equivalent to this object. The nonlinear solver is less
        efficient, but lets you freeze parameters, compute uncertainties, etc.

        If the `params` argument is provided, solve() will be called on the
        returned object with those parameters. If it is `None` and this object
        has parameters in `self.params`, those will be use. Otherwise, solve()
        will not be called on the returned object.

        """
        if params is None:
            params = self.params

        nlm = Model(None, self.data, self.invsigma)
        nlm.set_func(lambda p, x: npoly.polyval(x, p), self.pnames, args=(self.x,))

        if params is not None:
            nlm.solve(params)
        return nlm


class ScaleModel(ModelBase):
    """Solve `data = m * x` for `m`."""

    def __init__(self, x, data, invsigma=None):
        self.x = np.array(x, dtype=float, ndmin=1, copy=False, subok=True)
        self.set_data(data, invsigma)

    def make_frozen_func(self, params):
        return lambda x: params[0] * x

    def solve(self):
        w2 = self.invsigma**2
        sxx = np.dot(self.x**2, w2)
        sxy = np.dot(self.x * self.data, w2)
        m = sxy / sxx
        uc_m = 1.0 / np.sqrt(sxx)

        self.pnames = ["m"]
        self.params = np.asarray([m])
        self.puncerts = np.asarray([uc_m])
        self.covar = self.puncerts.reshape((1, 1))
        self.mfunc = lambda x: m * x
        self.mdata = m * self.x
        self.resids = self.data - self.mdata
        self.chisq = ((self.resids * self.invsigma) ** 2).sum()
        self.rchisq = self.chisq / (self.x.size - 1)
        return self


# lmmin-based model-fitting when the model is broken down into composable
# components.


class ModelComponent(object):
    npar = 0
    name = None
    pnames = ()
    nmodelargs = 0

    setguess = None
    setvalue = None
    setlimit = None
    _accum_mfunc = None

    def __init__(self, name=None):
        self.name = name

    def _param_names(self):
        """Overridable in case the list of parameter names needs to be
        generated on the fly."""
        return self.pnames

    def finalize_setup(self):
        """If the component has subcomponents, this should set their `name`,
        `setguess`, `setvalue`, and `setlimit` properties. It should also
        set `npar` (on self) to the final value."""
        pass

    def prep_params(self):
        """This should make any necessary calls to `setvalue` or `setlimit`,
        though in straightforward cases it should just be up to the user to
        do this. If the component has subcomponents, their `prep_params`
        functions should be called."""
        pass

    def model(self, pars, mdata):
        """Modify `mdata` based on `pars`."""
        pass

    def deriv(self, pars, jac):
        """Compute the Jacobian. `jac[i]` is d`mdata`/d`pars[i]`."""
        pass

    def extract(self, pars, perr, cov):
        """Extract fit results into the object for ease of inspection."""
        self.covar = cov

    def _outputshape(self, *args):
        """This is a helper for evaluating the model function at fixed parameters. To
        work in the ComposedModel paradigm, we have to allocate an empty array
        to hold the model output before we can fill it via the _accum_mfunc
        functions. We can't do that without knowing what size it will be. That
        size has to be a function of the "free" parameters to the model
        function that are implicit/fixed during the fitting process. Given these "free"
        parameters, _outputshape returns the shape that the output will have."""
        raise NotImplementedError()

    def mfunc(self, *args):
        if len(args) != self.nmodelargs:
            raise TypeError(
                "model function expected %d arguments, got %d"
                % (self.nmodelargs, len(args))
            )

        result = np.zeros(self._outputshape(*args))
        self._accum_mfunc(result, *args)
        return result


class ComposedModel(ModelBase):
    def __init__(self, component, data, invsigma=None):
        if component is not None:
            self.set_component(component)
        if data is not None:
            self.set_data(data, invsigma)

    def _component_setguess(self, vals, ofs=0):
        vals = np.asarray(vals)
        if ofs < 0 or ofs + vals.size > self.component.npar:
            raise ValueError(
                "ofs %d, vals.size %d, npar %d" % (ofs, vals.size, self.component.npar)
            )
        self.force_guess[ofs : ofs + vals.size] = vals

    def _component_setvalue(self, cidx, val, fixed=False):
        if cidx < 0 or cidx >= self.component.npar:
            raise ValueError("cidx %d, npar %d" % (cidx, self.component.npar))
        self.lm_prob.p_value(cidx, val, fixed=fixed)
        self.force_guess[cidx] = val

    def _component_setlimit(self, cidx, lower=-np.inf, upper=np.inf):
        if cidx < 0 or cidx >= self.component.npar:
            raise ValueError("cidx %d, npar %d" % (cidx, self.component.npar))
        self.lm_prob.p_limit(cidx, lower, upper)

    def set_component(self, component):
        self.component = component

        component.setguess = self._component_setguess
        component.setvalue = self._component_setvalue
        component.setlimit = self._component_setlimit
        component.finalize_setup()

        from .lmmin import Problem

        self.lm_prob = Problem(component.npar)
        self.force_guess = np.empty(component.npar)
        self.force_guess.fill(np.nan)
        self.pnames = list(component._param_names())

        component.prep_params()

    def solve(self, guess=None):
        if guess is None:
            guess = self.force_guess
        else:
            guess = np.array(guess, dtype=float, ndmin=1, copy=True)

            for i in range(self.force_guess.size):
                if np.isfinite(self.force_guess[i]):
                    guess[i] = self.force_guess[i]

        def model(pars, outputs):
            outputs.fill(0)
            self.component.model(pars, outputs)

        self.lm_model = model
        self.lm_deriv = self.component.deriv
        self.lm_prob.set_residual_func(
            self.data, self.invsigma, model, self.component.deriv
        )
        self.lm_soln = soln = self.lm_prob.solve(guess)

        self.params = soln.params
        self.puncerts = soln.perror
        self.covar = soln.covar

        # fvec = resids * invsigma = (data - mdata) * invsigma
        self.resids = self.lm_soln.fvec.reshape(self.data.shape) / self.invsigma
        self.mdata = self.data - self.resids

        # lm_soln.fnorm can be unreliable ("max(fnorm, fnorm1)" branch)
        self.chisq = (self.lm_soln.fvec**2).sum()
        if soln.ndof > 0:
            self.rchisq = self.chisq / soln.ndof

        self.component.extract(soln.params, soln.perror, soln.covar)
        return self

    def make_frozen_func(self):
        return self.component.mfunc

    def mfunc(self, *args):
        return self.component.mfunc(*args)

    def debug_derivative(self, guess):
        """returns (explicit, auto)"""
        from .lmmin import check_derivative

        return check_derivative(
            self.component.npar, self.data.size, self.lm_model, self.lm_deriv, guess
        )


# Now specific components useful in the above framework. The general strategy
# is to err on the side of having additional parameters in the individual
# classes, and the user can call setvalue() to fix them if they're not needed.


class AddConstantComponent(ModelComponent):
    npar = 1
    pnames = ("value",)
    nmodelargs = 0

    def model(self, pars, mdata):
        mdata += pars[0]

    def deriv(self, pars, jac):
        jac[0] = 1.0

    def _outputshape(self):
        return ()

    def extract(self, pars, perr, cov):
        def _accum_mfunc(res):
            res += pars[0]

        self._accum_mfunc = _accum_mfunc

        self.covar = cov
        self.f_value = pars[0]
        self.u_value = perr[0]


class AddValuesComponent(ModelComponent):
    """XXX terminology between this and AddConstant is mushy."""

    nmodelargs = 0

    def __init__(self, nvals, name=None):
        super(AddValuesComponent, self).__init__(name)
        self.npar = nvals

    def _param_names(self):
        for i in range(self.npar):
            yield "v%d" % i

    def model(self, pars, mdata):
        mdata += pars

    def deriv(self, pars, jac):
        jac[:, :] = np.eye(self.npar)

    def _outputshape(self):
        return (self.npar,)

    def extract(self, pars, perr, cov):
        def _accum_mfunc(res):
            res += pars

        self._accum_mfunc = _accum_mfunc

        self.covar = cov
        self.f_vals = pars
        self.u_vals = perr


class AddPolynomialComponent(ModelComponent):
    nmodelargs = 1

    def __init__(self, maxexponent, x, name=None):
        super(AddPolynomialComponent, self).__init__(name)
        self.npar = maxexponent + 1
        self.x = np.array(x, dtype=float, ndmin=1, copy=False, subok=True)

    def _param_names(self):
        for i in range(self.npar):
            yield "c%d" % i

    def model(self, pars, mdata):
        mdata += npoly.polyval(self.x, pars)

    def deriv(self, pars, jac):
        w = np.ones_like(self.x)

        for i in range(self.npar):
            jac[i] = w
            w *= self.x

    def _outputshape(self, x):
        return x.shape

    def extract(self, pars, perr, cov):
        def _accum_mfunc(res, x):
            res += npoly.polyval(x, pars)

        self._accum_mfunc = _accum_mfunc

        self.covar = cov
        self.f_coeffs = pars
        self.u_coeffs = perr


def _broadcast_shapes(s1, s2):
    """Given array shapes `s1` and `s2`, compute the shape of the array that would
    result from broadcasting them together."""

    n1 = len(s1)
    n2 = len(s2)
    n = max(n1, n2)
    res = [1] * n

    for i in range(n):
        if i >= n1:
            c1 = 1
        else:
            c1 = s1[n1 - 1 - i]

        if i >= n2:
            c2 = 1
        else:
            c2 = s2[n2 - 1 - i]

        if c1 == 1:
            rc = c2
        elif c2 == 1 or c1 == c2:
            rc = c1
        else:
            raise ValueError("array shapes %r and %r are not compatible" % (s1, s2))

        res[n - 1 - i] = rc

    return tuple(res)


class SeriesComponent(ModelComponent):
    """Apply a set of subcomponents in series, isolating each from the other. This
    is only valid if every subcomponent except the first is additive --
    otherwise, the Jacobian won't be right."""

    def __init__(self, components=(), name=None):
        super(SeriesComponent, self).__init__(name)
        self.components = list(components)

    def add(self, component):
        """This helps, but direct manipulation of self.components should be
        supported."""
        self.components.append(component)
        return self

    def _param_names(self):
        for c in self.components:
            pfx = c.name + "." if c.name is not None else ""
            for p in c._param_names():
                yield pfx + p

    def _offset_setguess(self, ofs, npar, vals, subofs=0):
        vals = np.asarray(vals)
        if subofs < 0 or subofs + vals.size > npar:
            raise ValueError(
                "subofs %d, vals.size %d, npar %d" % (subofs, vals.size, npar)
            )
        return self.setguess(vals, ofs + subofs)

    def _offset_setvalue(self, ofs, npar, cidx, value, fixed=False):
        if cidx < 0 or cidx >= npar:
            raise ValueError("cidx %d, npar %d" % (cidx, npar))
        return self.setvalue(ofs + cidx, value, fixed)

    def _offset_setlimit(self, ofs, npar, cidx, lower=-np.inf, upper=np.inf):
        if cidx < 0 or cidx >= npar:
            raise ValueError("cidx %d, npar %d" % (cidx, npar))
        return self.setlimit(ofs + cidx, lower, upper)

    def finalize_setup(self):
        from functools import partial

        ofs = 0
        self.nmodelargs = 0

        for i, c in enumerate(self.components):
            if c.name is None:
                c.name = "c%d" % i

            c.setguess = partial(self._offset_setguess, ofs, c.npar)
            c.setvalue = partial(self._offset_setvalue, ofs, c.npar)
            c.setlimit = partial(self._offset_setlimit, ofs, c.npar)
            c.finalize_setup()
            ofs += c.npar
            self.nmodelargs += c.nmodelargs

        self.npar = ofs

    def prep_params(self):
        for c in self.components:
            c.prep_params()

    def model(self, pars, mdata):
        ofs = 0

        for c in self.components:
            p = pars[ofs : ofs + c.npar]
            c.model(p, mdata)
            ofs += c.npar

    def deriv(self, pars, jac):
        ofs = 0

        for c in self.components:
            p = pars[ofs : ofs + c.npar]
            j = jac[ofs : ofs + c.npar]
            c.deriv(p, j)
            ofs += c.npar

    def extract(self, pars, perr, cov):
        ofs = 0

        for c in self.components:
            n = c.npar

            spar = pars[ofs : ofs + n]
            serr = perr[ofs : ofs + n]
            scov = cov[ofs : ofs + n, ofs : ofs + n]
            c.extract(spar, serr, scov)
            ofs += n

    def _outputshape(self, *args):
        s = ()
        ofs = 0

        for c in self.components:
            cargs = args[ofs : ofs + c.nmodelargs]
            s = _broadcast_shapes(s, c._outputshape(*cargs))
            ofs += c.nmodelargs

        return s

    def _accum_mfunc(self, res, *args):
        ofs = 0

        for c in self.components:
            cargs = args[ofs : ofs + c.nmodelargs]
            c._accum_mfunc(res, *cargs)
            ofs += c.nmodelargs


class MatMultComponent(ModelComponent):
    """Given a component yielding k**2 data points and k additional components,
    each yielding n data points. The result is [A]×[B], where A is the square
    matrix formed from the first component's output, and B is the (k, n)
    matrix of stacked output from the final k components.

    Parameters are ordered in same way as the components named above.
    """

    def __init__(self, k, name=None):
        super(MatMultComponent, self).__init__(name)
        self.k = k
        self.acomponent = None
        self.bcomponents = [None] * k

    def _param_names(self):
        pfx = self.acomponent.name + "." if self.acomponent.name is not None else ""
        for p in self.acomponent._param_names():
            yield pfx + p

        for c in self.bcomponents:
            pfx = c.name + "." if c.name is not None else ""
            for p in c._param_names():
                yield pfx + p

    def _offset_setguess(self, ofs, npar, vals, subofs=0):
        vals = np.asarray(vals)
        if subofs < 0 or subofs + vals.size > npar:
            raise ValueError(
                "subofs %d, vals.size %d, npar %d" % (subofs, vals.size, npar)
            )
        return self.setguess(vals, ofs + subofs)

    def _offset_setvalue(self, ofs, npar, cidx, value, fixed=False):
        if cidx < 0 or cidx >= npar:
            raise ValueError("cidx %d, npar %d" % (cidx, npar))
        return self.setvalue(ofs + cidx, value, fixed)

    def _offset_setlimit(self, ofs, npar, cidx, lower=-np.inf, upper=np.inf):
        if cidx < 0 or cidx >= npar:
            raise ValueError("cidx %d, npar %d" % (cidx, npar))
        return self.setlimit(ofs + cidx, lower, upper)

    def finalize_setup(self):
        from functools import partial

        c = self.acomponent

        if c.name is None:
            c.name = "a"

        c.setguess = partial(self._offset_setguess, 0, c.npar)
        c.setvalue = partial(self._offset_setvalue, 0, c.npar)
        c.setlimit = partial(self._offset_setlimit, 0, c.npar)
        c.finalize_setup()
        ofs = c.npar
        self.nmodelargs = c.nmodelargs

        for i, c in enumerate(self.bcomponents):
            if c.name is None:
                c.name = "b%d" % i

            c.setguess = partial(self._offset_setguess, ofs, c.npar)
            c.setvalue = partial(self._offset_setvalue, ofs, c.npar)
            c.setlimit = partial(self._offset_setlimit, ofs, c.npar)
            c.finalize_setup()
            ofs += c.npar
            self.nmodelargs += c.nmodelargs

        self.npar = ofs

    def prep_params(self):
        self.acomponent.prep_params()

        for c in self.bcomponents:
            c.prep_params()

    def _sep_model(self, pars, nd):
        k = self.k
        ma = np.zeros((k, k))
        mb = np.zeros((k, nd))

        c = self.acomponent
        c.model(pars[: c.npar], ma.reshape(k**2))

        pofs = c.npar

        for i, c in enumerate(self.bcomponents):
            p = pars[pofs : pofs + c.npar]
            c.model(p, mb[i])
            pofs += c.npar

        return ma, mb

    def model(self, pars, mdata):
        k = self.k
        nd = mdata.size // k
        ma, mb = self._sep_model(pars, nd)
        np.dot(ma, mb, mdata.reshape((k, nd)))

    def deriv(self, pars, jac):
        k = self.k
        nd = jac.shape[1] // k
        npar = self.npar

        ma, mb = self._sep_model(pars, nd)
        ja = np.zeros((npar, k, k))
        jb = np.zeros((npar, k, nd))

        c = self.acomponent
        c.deriv(pars[: c.npar], ja[: c.npar].reshape((c.npar, k**2)))
        pofs = c.npar

        for i, c in enumerate(self.bcomponents):
            p = pars[pofs : pofs + c.npar]
            c.deriv(p, jb[pofs : pofs + c.npar, i, :])
            pofs += c.npar

        for i in range(self.npar):
            jac[i] = (np.dot(ja[i], mb) + np.dot(ma, jb[i])).reshape(k * nd)

    def extract(self, pars, perr, cov):
        c = self.acomponent
        c.extract(pars[: c.npar], perr[: c.npar], cov[: c.npar, : c.npar])
        ofs = c.npar

        for c in self.bcomponents:
            n = c.npar

            spar = pars[ofs : ofs + n]
            serr = perr[ofs : ofs + n]
            scov = cov[ofs : ofs + n, ofs : ofs + n]
            c.extract(spar, serr, scov)
            ofs += n

    def _outputshape(self, *args):
        aofs = self.acomponent.nmodelargs
        sb = ()

        for c in self.bcomponents:
            a = args[aofs : aofs + c.nmodelargs]
            sb = _broadcast_shapes(sb, c._outputshape(*a))
            aofs += c.nmodelargs

        return (self.k,) + sb

    def _accum_mfunc(self, res, *args):
        k = self.k
        nd = res.shape[1]

        ma = np.zeros((k, k))
        mb = np.zeros((k, nd))

        c = self.acomponent
        c._accum_mfunc(ma.reshape(k**2), *(args[: c.nmodelargs]))
        aofs = c.nmodelargs

        for i, c in enumerate(self.bcomponents):
            a = args[aofs : aofs + c.nmodelargs]
            c._accum_mfunc(mb[i], *a)
            aofs += c.nmodelargs

        np.dot(ma, mb, res)


class ScaleComponent(ModelComponent):
    npar = 1

    def __init__(self, subcomp=None, name=None):
        super(ScaleComponent, self).__init__(name)
        self.setsubcomp(subcomp)

    def setsubcomp(self, subcomp):
        self.subcomp = subcomp
        return self

    def _param_names(self):
        yield "factor"

        pfx = self.subcomp.name + "." if self.subcomp.name is not None else ""
        for p in self.subcomp._param_names():
            yield pfx + p

    def _sub_setguess(self, npar, cidx, vals, ofs=0):
        vals = np.asarray(vals)
        if ofs < 0 or ofs + vals.size > npar:
            raise ValueError("ofs %d, vals.size %d, npar %d" % (ofs, vals.size, npar))
        return self.setguess(vals, ofs + 1)

    def _sub_setvalue(self, npar, cidx, value, fixed=False):
        if cidx < 0 or cidx >= npar:
            raise ValueError("cidx %d, npar %d" % (cidx, npar))
        return self.setvalue(1 + cidx, value, fixed)

    def _sub_setlimit(self, npar, cidx, lower=-np.inf, upper=np.inf):
        if cidx < 0 or cidx >= npar:
            raise ValueError("cidx %d, npar %d" % (cidx, npar))
        return self.setlimit(1 + cidx, lower, upper)

    def finalize_setup(self):
        if self.subcomp.name is None:
            self.subcomp.name = "c"

        from functools import partial

        self.subcomp.setvalue = partial(self._sub_setvalue, self.subcomp.npar)
        self.subcomp.setlimit = partial(self._sub_setvalue, self.subcomp.npar)
        self.subcomp.finalize_setup()

        self.npar = self.subcomp.npar + 1
        self.nmodelargs = self.subcomp.nmodelargs

    def prep_params(self):
        self.subcomp.prep_params()

    def model(self, pars, mdata):
        self.subcomp.model(pars[1:], mdata)
        mdata *= pars[0]

    def deriv(self, pars, jac):
        self.subcomp.model(pars[1:], jac[0])
        self.subcomp.deriv(pars[1:], jac[1:])
        jac[1:] *= pars[0]

    def extract(self, pars, perr, cov):
        self.f_factor = pars[0]
        self.u_factor = perr[0]
        self.c_factor = cov[0]

        self.subcomp.extract(pars[1:], perr[1:], cov[1:, 1:])

    def _outputshape(self, *args):
        return self.subcomp._outputshape(*args)

    def _accum_mfunc(self, res, *args):
        self.subcomp._accum_mfunc(res, *args)
