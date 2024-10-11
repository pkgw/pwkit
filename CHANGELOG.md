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
