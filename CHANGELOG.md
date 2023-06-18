# pwkit 1.1.0 (2023-06-18)

A test release, after many years of inactivity, because the previous release
won't even import noawadays due to the use of very old Numpy features that have
been removed.

- Switch to Cranko CI and release automation
- Smattering of modernizations
- Replace deprecated `np.{int,float,complex,bool}` with corresponding built-in types
- Remove uses of np.asscalar(), which has been removed in latest Numpys
- `pwkit/inifile.py`: untested hacks to get it working on Python 3
- `pwkit/astutil.py`: fix up `sastrom`
- `pwkit/colormaps.py`: update to use Gtk 3


# Version 1.0.0 (2019 Dec 19)

Call this version 1.0.0!

- Fixed `casatask bpplot` to work when not all spectral windows in the
  bandpass solution contain the same number of channels.
- Fixed `casatask bpplot` to work when an amplitude or phase solution is
  exactly constant.
