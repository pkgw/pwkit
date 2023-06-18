# -*- mode: python; coding: utf-8 -*-
# Copyright 2014 Peter Williams <peter@newton.cx> and collaborators.
# Licensed under the MIT License.

"""pwkit.bblocks - Bayesian Blocks analysis, with a few extensions.

Bayesian Blocks analysis for the "time tagged" case described by Scargle+
2013. Inspired by the bayesian_blocks implementation by Jake Vanderplas in the
AstroML package, but that turned out to have some limitations.

We have iterative determination of the best number of blocks (using an ad-hoc
routine described in Scargle+ 2013) and bootstrap-based determination of
uncertainties on the block heights (ditto).

Functions are:

:func:`bin_bblock`
  Bayesian Blocks analysis with counts and bins.
:func:`tt_bblock`
  BB analysis of time-tagged events.
:func:`bs_tt_bblock`
  Like :func:`tt_bblock` with bootstrap-based uncertainty assessment. NOTE:
  the uncertainties are not very reliable!

"""
from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = str("nlogn bin_bblock tt_bblock bs_tt_bblock").split()


import numpy as np

from . import Holder


def nlogn(n, dt):
    # I really feel like there must be a cleverer way to do this
    # scalar-or-vector possible-bad-value masking.

    if np.isscalar(n):
        if n == 0:
            return 0.0
        return n * (np.log(n) - np.log(dt))

    n = np.asarray(n)
    mask = n == 0
    r = n * (np.log(np.where(mask, 1, n)) - np.log(dt))
    return np.where(mask, 0, r)


def bin_bblock(widths, counts, p0=0.05):
    """Fundamental Bayesian Blocks algorithm. Arguments:

    widths  - Array of consecutive cell widths.
    counts  - Array of numbers of counts in each cell.
    p0=0.05 - Probability of preferring solutions with additional bins.

    Returns a Holder with:

    blockstarts - Start times of output blocks.
    counts      - Number of events in each output block.
    finalp0     - Final value of p0, after iteration to minimize `nblocks`.
    nblocks     - Number of output blocks.
    ncells      - Number of input cells/bins.
    origp0      - Original value of p0.
    rates       - Event rate associated with each block.
    widths      - Width of each output block.

    """
    widths = np.asarray(widths)
    counts = np.asarray(counts)
    ncells = widths.size
    origp0 = p0

    if np.any(widths <= 0):
        raise ValueError("bin widths must be positive")
    if widths.size != counts.size:
        raise ValueError("widths and counts must have same size")
    if p0 < 0 or p0 >= 1.0:
        raise ValueError("p0 must lie within [0, 1)")

    vedges = np.cumsum(np.concatenate(([0], widths)))  # size: ncells + 1
    block_remainders = vedges[-1] - vedges  # size: nedges = ncells + 1
    ccounts = np.cumsum(np.concatenate(([0], counts)))
    count_remainders = ccounts[-1] - ccounts

    prev_blockstarts = None
    best = np.zeros(ncells, dtype=float)
    last = np.zeros(ncells, dtype=int)

    for _ in range(10):
        # Pluggable num-change-points prior-weight expression:
        ncp_prior = 4 - np.log(p0 / (0.0136 * ncells**0.478))

        for r in range(ncells):
            tk = block_remainders[: r + 1] - block_remainders[r + 1]
            nk = count_remainders[: r + 1] - count_remainders[r + 1]

            # Pluggable fitness expression:
            fit_vec = nlogn(nk, tk)

            # This incrementally penalizes partitions with more blocks:
            tmp = fit_vec - ncp_prior
            tmp[1:] += best[:r]

            imax = np.argmax(tmp)
            last[r] = imax
            best[r] = tmp[imax]

        # different semantics than Scargle impl: our blockstarts is similar to
        # their changepoints, but we always finish with blockstarts[0] = 0.

        work = np.zeros(ncells, dtype=int)
        workidx = 0
        ind = last[-1]

        while True:
            work[workidx] = ind
            workidx += 1
            if ind == 0:
                break
            ind = last[ind - 1]

        blockstarts = work[:workidx][::-1]

        if prev_blockstarts is not None:
            if (
                blockstarts.size == prev_blockstarts.size
                and (blockstarts == prev_blockstarts).all()
            ):
                break  # converged

        if blockstarts.size == 1:
            break  # can't shrink any farther

        # Recommended ad-hoc iteration to favor fewer blocks above and beyond
        # the value of p0:
        p0 = 1.0 - (1.0 - p0) ** (1.0 / (blockstarts.size - 1))
        prev_blockstarts = blockstarts

    assert blockstarts[0] == 0
    nblocks = blockstarts.size

    info = Holder()
    info.ncells = ncells
    info.nblocks = nblocks
    info.origp0 = origp0
    info.finalp0 = p0
    info.blockstarts = blockstarts
    info.counts = np.empty(nblocks, dtype=int)
    info.widths = np.empty(nblocks)

    for iblk in range(nblocks):
        cellstart = blockstarts[iblk]
        if iblk == nblocks - 1:
            cellend = ncells - 1
        else:
            cellend = blockstarts[iblk + 1] - 1

        info.widths[iblk] = widths[cellstart : cellend + 1].sum()
        info.counts[iblk] = counts[cellstart : cellend + 1].sum()

    info.rates = info.counts / info.widths
    return info


def tt_bblock(tstarts, tstops, times, p0=0.05, intersect_with_bins=False):
    """Bayesian Blocks for time-tagged events. Arguments:

    *tstarts*
      Array of input bin start times.
    *tstops*
      Array of input bin stop times.
    *times*
      Array of event arrival times.
    *p0* = 0.05
      Probability of preferring solutions with additional bins.
    *intersect_with_bins* = False
      If true, intersect bblock bins with input bins; can result in more bins
      than bblocks wants; they will have the same rate values.

    Returns a Holder with:

    *counts*
      Number of events in each output block.
    *finalp0*
      Final value of p0, after iteration to minimize `nblocks`.
    *ledges*
      Times of left edges of output blocks.
    *midpoints*
      Times of midpoints of output blocks.
    *nblocks*
      Number of output blocks.
    *ncells*
      Number of input cells/bins.
    *origp0*
      Original value of p0.
    *rates*
      Event rate associated with each block.
    *redges*
      Times of right edges of output blocks.
    *widths*
      Width of each output block.

    Bin start/stop times are best derived from a 1D Voronoi tesselation of the
    event arrival times, with some kind of global observation start/stop time
    setting the extreme edges. Or they can be set from "good time intervals"
    if observations were toggled on or off as in an X-ray telescope.

    If *intersect_with_bins* is True, the true Bayesian Blocks bins (BBBs) are
    intersected with the "good time intervals" (GTIs) defined by the *tstarts*
    and *tstops* variables. One GTI may contain multiple BBBs if the event
    rate appears to change within the GTI, and one BBB may span multiple GTIs
    if the event date does *not* appear to change between the GTIs. The
    intersection will ensure that no BBB intervals cross the edge of a GTI. If
    this would happen, the BBB is split into multiple, partially redundant
    records. Each of these records will have the **same** value for the
    *counts*, *rates*, and *widths* values. However, the *ledges*, *redges*,
    and *midpoints* values will be recalculated. Note that in this mode, it is
    not necessarily true that ``widths = ledges - redges`` as is usually the
    case. When this flag is true, keep in mind that multiple bins are
    therefore *not* necessarily independent statistical samples.

    """
    tstarts = np.asarray(tstarts)
    tstops = np.asarray(tstops)
    times = np.asarray(times)

    if tstarts.size != tstops.size:
        raise ValueError("must have same number of starts and stops")

    ngti = tstarts.size

    if ngti < 1:
        raise ValueError("must have at least one goodtime interval")
    if np.any((tstarts[1:] - tstarts[:-1]) <= 0):
        raise ValueError("tstarts must be ordered and distinct")
    if np.any((tstops[1:] - tstops[:-1]) <= 0):
        raise ValueError("tstops must be ordered and distinct")
    if np.any(tstarts >= tstops):
        raise ValueError("tstarts must come before tstops")
    if np.any((times[1:] - times[:-1]) < 0):
        raise ValueError("times must be ordered")
    if times.min() < tstarts[0]:
        raise ValueError("no times may be smaller than first tstart")
    if times.max() > tstops[-1]:
        raise ValueError("no times may be larger than last tstop")
    for i in range(1, ngti):
        if np.where((times > tstops[i - 1]) & (times < tstarts[i]))[0].size:
            raise ValueError("no times may fall in goodtime gap #%d" % i)
    if p0 < 0 or p0 >= 1.0:
        raise ValueError("p0 must lie within [0, 1)")

    utimes, uidxs = np.unique(times, return_index=True)
    nunique = utimes.size

    counts = np.empty(nunique)
    counts[:-1] = uidxs[1:] - uidxs[:-1]
    counts[-1] = times.size - uidxs[-1]
    assert counts.sum() == times.size

    # we grow these arrays with concats, which will perform badly with lots of
    # GTIs. Not expected to be a big deal.
    widths = np.empty(0)
    ledges = np.empty(0)
    redges = np.empty(0)

    for i in range(ngti):
        tstart, tstop = tstarts[i], tstops[i]

        w = np.where((utimes >= tstart) & (utimes <= tstop))[0]

        if not w.size:
            # No events during this goodtime! We have to insert a zero-count
            # event block. This may break assumptions within bin_bblock()?

            # j = idx of first event after this GTI
            wafter = np.where(utimes > tstop)[0]
            if wafter.size:
                j = wafter[0]
            else:
                j = utimes.size
            assert j == 0 or np.where(utimes < tstart)[0][-1] == j - 1

            counts = np.concatenate((counts[:j], [0], counts[j:]))
            widths = np.concatenate((widths, [tstop - tstart]))
            ledges = np.concatenate((ledges, [tstart]))
            redges = np.concatenate((redges, [tstop]))
        else:
            gtutimes = utimes[w]
            midpoints = 0.5 * (gtutimes[1:] + gtutimes[:-1])  # size: n - 1
            gtedges = np.concatenate(([tstart], midpoints, [tstop]))  # size: n + 1
            gtwidths = gtedges[1:] - gtedges[:-1]  # size: n
            assert gtwidths.sum() == tstop - tstart
            widths = np.concatenate((widths, gtwidths))
            ledges = np.concatenate((ledges, gtedges[:-1]))
            redges = np.concatenate((redges, gtedges[1:]))

    assert counts.size == widths.size
    info = bin_bblock(widths, counts, p0=p0)
    info.ledges = ledges[info.blockstarts]
    # The right edge of the i'th block is the right edge of its rightmost
    # bin, which is the bin before the leftmost bin of the (i+1)'th block:
    info.redges = np.concatenate((redges[info.blockstarts[1:] - 1], [redges[-1]]))
    info.midpoints = 0.5 * (info.ledges + info.redges)
    del info.blockstarts

    if intersect_with_bins:
        # OK, we now need to intersect the bblock bins with the input bins.
        # This can fracture one bblock bin into multiple ones but shouldn't
        # make any of them disappear, since they're definitionally constrained
        # to contain events.
        #
        # First: sorted list of all timestamps at which *something* changes:
        # either a bblock edge, or a input bin edge. We drop the last entry,
        # giving is a list of left edges of bins in which everything is the
        # same.

        all_times = set(tstarts)
        all_times.update(tstops)
        all_times.update(info.ledges)
        all_times.update(info.redges)
        all_times = np.array(sorted(all_times))[:-1]

        # Now, construct a lookup table of which bblock number each of these
        # bins corresponds to. More than one bin may have the same bblock
        # number, if a GTI change slices a single bblock into more than one
        # piece. We do this in a somewhat non-obvious way since we know that
        # the bblocks completely cover the overall GTI span in order.

        bblock_ids = np.zeros(all_times.size)

        for i in range(1, info.nblocks):
            bblock_ids[all_times >= info.ledges[i]] = i

        # Now, a lookup table of which bins are within a good GTI span. Again,
        # we know that all bins are either entirely in a good GTI or entirely
        # outside, so the logic is simplified but not necessarily obvious.

        good_timeslot = np.zeros(all_times.size, dtype=bool)

        for t0, t1 in zip(tstarts, tstops):
            ok = (all_times >= t0) & (all_times < t1)
            good_timeslot[ok] = True

        # Finally, look for contiguous spans that are in a good timeslot *and*
        # have the same underlying bblock number. These are our intersected
        # blocks.

        old_bblock_ids = []
        ledges = []
        redges = []
        cur_bblock_id = -1

        for i in range(all_times.size):
            if bblock_ids[i] != cur_bblock_id or not good_timeslot[i]:
                if cur_bblock_id >= 0:
                    # Ending a previous span.
                    redges.append(all_times[i])
                    cur_bblock_id = -1

                if good_timeslot[i]:
                    # Starting a new span.
                    ledges.append(all_times[i])
                    old_bblock_ids.append(bblock_ids[i])
                    cur_bblock_id = bblock_ids[i]

        if cur_bblock_id >= 0:
            # End the last span.
            redges.append(tstops[-1])

        # Finally, rewrite all of the data as planned.

        old_bblock_ids = np.array(old_bblock_ids, dtype=int)
        info.counts = info.counts[old_bblock_ids]
        info.rates = info.rates[old_bblock_ids]
        info.widths = info.widths[old_bblock_ids]

        info.ledges = np.array(ledges)
        info.redges = np.array(redges)
        info.midpoints = 0.5 * (info.ledges + info.redges)
        info.nblocks = info.ledges.size

    return info


def bs_tt_bblock(times, tstarts, tstops, p0=0.05, nbootstrap=512):
    """Bayesian Blocks for time-tagged events with bootstrapping uncertainty
    assessment. THE UNCERTAINTIES ARE NOT VERY GOOD! Arguments:

    tstarts        - Array of input bin start times.
    tstops         - Array of input bin stop times.
    times          - Array of event arrival times.
    p0=0.05        - Probability of preferring solutions with additional bins.
    nbootstrap=512 - Number of bootstrap runs to perform.

    Returns a Holder with:

    blockstarts - Start times of output blocks.
    bsrates     - Mean event rate in each bin from bootstrap analysis.
    bsrstds     - ~Uncertainty: stddev of event rate in each bin from bootstrap analysis.
    counts      - Number of events in each output block.
    finalp0     - Final value of p0, after iteration to minimize `nblocks`.
    ledges      - Times of left edges of output blocks.
    midpoints   - Times of midpoints of output blocks.
    nblocks     - Number of output blocks.
    ncells      - Number of input cells/bins.
    origp0      - Original value of p0.
    rates       - Event rate associated with each block.
    redges      - Times of right edges of output blocks.
    widths      - Width of each output block.

    """
    times = np.asarray(times)
    tstarts = np.asarray(tstarts)
    tstops = np.asarray(tstops)

    nevents = times.size
    if nevents < 1:
        raise ValueError("must be given at least 1 event")

    info = tt_bblock(tstarts, tstops, times, p0)

    # Now bootstrap resample to assess uncertainties on the bin heights. This
    # is the approach recommended by Scargle+.

    bsrsums = np.zeros(info.nblocks)
    bsrsumsqs = np.zeros(info.nblocks)

    for _ in range(nbootstrap):
        bstimes = times[np.random.randint(0, times.size, times.size)]
        bstimes.sort()
        bsinfo = tt_bblock(tstarts, tstops, bstimes, p0)
        blocknums = np.minimum(
            np.searchsorted(bsinfo.redges, info.midpoints), bsinfo.nblocks - 1
        )
        samprates = bsinfo.rates[blocknums]
        bsrsums += samprates
        bsrsumsqs += samprates**2

    bsrmeans = bsrsums / nbootstrap
    mask = bsrsumsqs / nbootstrap <= bsrmeans**2
    bsrstds = np.sqrt(np.where(mask, 0, bsrsumsqs / nbootstrap - bsrmeans**2))
    info.bsrates = bsrmeans
    info.bsrstds = bsrstds
    return info
