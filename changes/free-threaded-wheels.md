<!-- area: platform -->
### Added

- **Free-threaded CPython 3.14 installs a wheel.** Each platform gets a second
  wheel, `cp314t`, for the interpreter the abi3 wheel cannot load, so medh5
  installs there without a compiler. The module asks for the GIL, which
  CPython turns on as it imports it; running without it is planned for 2.2.
