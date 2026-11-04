"""Lets a run start with `python -m galactic_tidal_constraints.simr`.

A separate file so that the code MPI workers call lives in the `simr` module, not in `__main__`.
"""

from galactic_tidal_constraints.simr.simr import main

if __name__ == "__main__":
    main()
