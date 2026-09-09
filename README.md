srp_tracing
==============================

Shortest ray path model for austenitic welds

`srp_tracing` provides a fast ray tracing solver based on the shortest ray path (SRP) principle, belonging to the wider family of Dijkstra methods. The model was created with the intention to be used in ultrasonic tomography and a number of choices have been made to make that task simpler. Consequently, it may not be possible to model every setup straight out of the box.
Follow the jupyter notebook for further details.

Requirements
------------

`pip install -e .` installs the PyPI dependencies (numpy, scipy, tqdm, shapely, matplotlib) and is all `srp_tracing` itself needs -- its anisotropic wave-velocity solver (`srp_tracing/_wave_physics.py`) is self-contained, ported from the external `raytracer` package it used to depend on.

The demo scripts in `examples/` (and `tests/test_ogilvy_validation.py`) additionally depend on the unpublished `ogilvy_weld` sibling package, declared as an editable local install in `environment.yml` (`conda env create -f environment.yml`, run from this repo's root, assuming `ogilvy_weld/` is checked out as a sibling directory). A few of the `examples/` scripts further depend on the unpublished `mina` and `advise_support` packages and on local `../data/*.npy` files, which are not covered by `environment.yml`.

--------
