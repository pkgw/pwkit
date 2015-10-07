# -*- mode: python; coding: utf-8 -*-
# Copyright 2012-2014 Peter Williams <peter@newton.cx> and collaborators.
# Licensed under the MIT License.

"""pwkit.contours - Tracing contours in functions and data.

Uses my own homebrew algorithm. So far, it's only tested on extremely
well-behaved functions, so probably doesn't cope well with poorly-behaved
ones.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = str ('analytic_2d').split ()


import numpy as np

from . import PKError


def analytic_2d (f, df, x0, y0,
                 maxiters=5000,
                 defeta=0.05,
                 netastep=12,
                 vtol1=1e-3,
                 vtol2=1e-8,
                 maxnewt=20,
                 dorder=7,
                 goright=False):
    """Sample a contour in a 2D analytic function. Arguments:

    f
      A function, mapping (x, y) -> z.
    df
      The partial derivative: df (x, y) -> [dz/dx, dz/dy]. If None,
      the derivative of f is approximated numerically with
      scipy.derivative.
    x0
      Initial x value. Should be of "typical" size for the problem;
      avoid 0.
    y0
      Initial y value. Should be of "typical" size for the problem;
      avoid 0.

    Optional arguments:

    maxiters
      Maximum number of points to create. Default 5000.
    defeta
      Initially offset by distances of defeta*[df/dx, df/dy]
      Default 0.05.
    netastep
      Number of steps between defeta and the machine resolution
      in which we test eta values for goodness. (OMG FIXME doc).
      Default 12.
    vtol1
      Tolerance for constancy in the value of the function in the
      initial offset step. The value is only allowed to vary by
      ``f(x0,y0) * vtol1``. Default 1e-3.
    vtol2
      Tolerance for constancy in the value of the function in the
      along the contour. The value is only allowed to vary by
      ``f(x0,y0) * vtol2``. Default 1e-8.
    maxnewt
      Maximum number of Newton's method steps to take when
      attempting to hone in on the desired function value. Default 20.
    dorder
      Number of function evaluations to perform when evaluating
      the derivative of f numerically. Must be an odd integer greater
      than 1. Default 7.
    goright
      If True, trace the contour rightward (as looking uphill),
      rather than leftward (the default).

    """
    # Coerce argument types.

    if not callable (f):
        raise ValueError ('f')

    if df is not None and not callable (df):
        raise ValueError ('df')

    x0 = float (x0)
    if x0 == 0.:
        raise ValueError ('x0')

    y0 = float (y0)
    if y0 == 0.:
        raise ValueError ('y0')

    maxiters = int (maxiters)
    if maxiters < 3:
        raise ValueError ('maxiters')

    defeta = float (defeta)
    if defeta <= 0:
        raise ValueError ('defeta')

    netastep = int (netastep)
    if netastep < 2:
        raise ValueError ('netastep')

    vtol1 = float (vtol1)
    if vtol1 <= 0:
        raise ValueError ('vtol1')

    vtol2 = float (vtol2)
    if vtol2 >= vtol1:
        raise ValueError ('vtol2')

    maxnewt = int (maxnewt)
    if maxnewt < 1:
        raise ValueError ('maxnewt')

    # What value are we contouring?
    v = f (x0, y0)

    # If no derivative is given, use a numerical approximation.

    if df is None:
        derivx = abs (x0 * 0.025)
        derivy = abs (y0 * 0.025)
        from scipy import derivative

        if dorder == 2:
            # simple derivative
            def df (x1, y1):
                z0 = f (x1, y1)
                dx = max (abs (x1) * 1e-5, 1e-8)
                dy = max (abs (y1) * 1e-5, 1e-8)
                dzdx = (f (x1 + dx, y1) - z0) / dx
                dzdy = (f (x1, y1 + dy) - z0) / dy
                return [dzdx, dzdy]
        else:
            def df (x1, y1):
                dx = derivative (lambda x: f (x, y1), x1, derivx, order=dorder)
                dy = derivative (lambda y: f (x1, y), y1, derivy, order=dorder)
                return [dx, dy]

    # Init eta progression.
    rez = np.finfo (np.double).resolution
    if rez > defeta:
        raise PKError ('defeta below resolution!')
    eta_scale = np.exp ((np.log (rez) - np.log (defeta)) / netastep)

    # Init data storage
    n = 1
    pts = np.empty ((maxiters, 2))
    pts[0] = (x0, y0)
    x = x0
    y = y0

    # Quitflag: 0 if first iteration
    #           1 if inited but not yet ok to quit (definition of this below)
    #           2 if ok to quit
    # initquad: 0 if x > 0, y > 0
    #           1 if x < 0, y > 0
    #           2 if x < 0, y < 0
    #           3 if x > 0, y < 0
    # We invert these senses in the in-loop test to make comparison easy.

    quitflag = 0
    initquad = -1

    # Start finding contours.

    while n < maxiters:
        dfdx, dfdy = df (x, y)

        # If we're booting up, remember the quadrant that df/dx points in.
        # Once we've rotated around to the other direction, it is safe to quit
        # once we return close to the original point, since we must have
        # completed a circle.

        if quitflag == 0:
            if dfdx > 0:
                if dfdy > 0:
                    initquad = 0
                else:
                    initquad = 3
            else:
                if dfdy > 0:
                    initquad = 1
                else:
                    initquad = 2
            quitflag = 1
        elif quitflag == 1:
            if dfdx > 0:
                if dfdy > 0:
                    curquad = 2
                else:
                    curquad = 1
            else:
                if dfdy > 0:
                    curquad = 3
                else:
                    curquad = 0

            if curquad == initquad:
                quitflag = 2

        # We will move perpendicular to [df/dx, df/dy], rotating to the left
        # (arbitrarily) from that direction. We need to figure out how far we
        # can safely move in this direction.

        if goright:
            dx = dfdy * defeta
            dy = -dfdx * defeta
        else:
            dx = -dfdy * defeta
            dy = dfdx * defeta

        i = 0

        while i < netastep:
            nx = x + dx
            ny = y + dy
            nv = f (nx, ny)

            # Is the value of the function sufficently close to what
            # we're aiming for?
            if abs (nv / v - 1) < vtol1:
                break

            # No. Try a smaller dx/dy.
            dx *= eta_scale
            dy *= eta_scale
            i += 1
        else:
            # Failed to find a sufficiently small eta (did not break out of
            # loop)
            raise PKError ('failed to find sufficiently small eta: xy %g,%g; '
                           'dv %g; df %g,%g; dxy %g,%g; defeta %g; eta_scale '
                           '%g' % (x, y, nv - v, dfdx, dfdy, dx, dy, defeta,
                                   eta_scale))

        # Now compute a new [df/dx, df/dy], and move along it, finding our way
        # back to the desired value, 'v'. Newton's method should suffice. This
        # loop usually exits after one iteration.

        i = 0

        while i < maxnewt:
            dfdx, dfdy = df (nx, ny)
            df2 = dfdx**2 + dfdy**2
            dv = nv - v

            nx -= dv * dfdx / df2
            ny -= dv * dfdy / df2
            nv = f (nx, ny)

            if abs (nv/v - 1) < vtol2:
                break

            i += 1
        else:
            # Did not break out of loop.
            raise PKError ('failed to converge with Newton\'s method')

        # Ok, we found our next value.
        pts[n] = (nx, ny)
        x = nx
        y = ny
        n += 1

        # Time to stop? Make sure we've gone at least a half-turn so that we
        # don't just exit on the first iteration.
        if quitflag == 2:
            dist2 = (x/x0 - 1)**2 + (y/y0 - 1)**2
            if dist2 < 3 * (dx**2 + dy**2):
                break
    else:
        raise PKError ('needed too many points to close contour')

    # Woohoo! All done.
    return pts[:n]
