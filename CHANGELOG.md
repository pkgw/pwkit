# rc: micro bump

- *Actually* fix `pwkit.astutil.get_2mass_epoch` (#26, @pkgw)
- In CASA, update `dftphotom`, `dftdynspec`, and `dftspect` to
  work with CASA 6.

The DOI of this release is [xx.xxxx/dev-build.pwkit.version][vdoi].

[vdoi]: https://doi.org/xx.xxxx/dev-build.pwkit.version


# pwkit 1.3.0 (2025-04-25)

- Add support for CASA 6! (#25, @pkgw). It's still incomplete, and there
  may be some lingering bugs, but I am currently able to run most of my
  processing pipeline on real VLA data.
- In `imtool show`, pressing `f` or `p` will now fit a source at the position of
  the mouse cursor. The `f` key will fit either a point or a Gaussian, depending
  on what fits better; `p` will force a point-source fit.
- Fix `pwkit.astutil.get_2mass_epoch` to use HTTPS to contact the Vizier API
- More Numpy 2 compatibility fixes, including the use of `np.asarray` with a
  `float` dtype instead of `np.asfarray`. This could potentially reduce
  efficiency if the input was a `float32` and we're upgrading it to a `float64`.

The DOI of this release is [10.5281/zenodo.15283107][vdoi].

[vdoi]: https://doi.org/10.5281/zenodo.15283107


# pwkit 1.2.2 (2024-10-11)

- Correct fixup processing of args with non-None defaults in `pwkit.kwargv` (#24)
- Correct the docs link in the README! (#23)

The version DOI of this release is [10.5281/zenodo.13920308][vdoi].

[vdoi]: https://doi.org/10.5281/zenodo.13920308


# pwkit 1.2.1 (2024-09-13)

- Fix `pwkit.parallel.SerialHelper.get_map()` for Python 3 (#22)
- Fix `pwkit.lmmin.Problem.set_npar()` for recent Numpys (#20). I'm not sure if
  we'll be compatible with Numpy 2.0, though.
- Get the ReadTheDocs docs building again, and make it so that we check RTD
  builds as part of pull request processing
- README.md: fix some links to the documentation (@AstroGenevieve)


# pwkit 1.2.0 (2023-06-18)

- Remove the dependency on `six`
- Remove the long-unused `pwkit.ndshow_gtk2` module
- A further attempt to fix the ReadTheDocs build.


# pwkit 1.1.1 (2023-06-18)

No code changes.

- The last release did not publish to PyPI due to a problem with
  the CI automation; attempt to fix that.
- Also attempt to fix the ReadTheDocs build.


# pwkit 1.1.0 (2023-06-18)

A test release, after many years of inactivity, because the previous release
won't even import noawadays due to the use of very old Numpy features that have
been removed.

- Switch to Cranko CI and release automation
- Smattering of modernizations
- Replace deprecated `np.{int,float,complex,bool}` with corresponding built-in types
- Remove uses of `np.asscalar()`, which has been removed in latest Numpys
- `pwkit/inifile.py`: untested hacks to get it working on Python 3
- `pwkit/astutil.py`: fix up `sastrom`
- `pwkit/colormaps.py`: update to use Gtk 3


# Version 1.0.0 (2019 Dec 19)

Call this version 1.0.0!

- Fixed `casatask bpplot` to work when not all spectral windows in the
  bandpass solution contain the same number of channels.
- Fixed `casatask bpplot` to work when an amplitude or phase solution is
  exactly constant.
