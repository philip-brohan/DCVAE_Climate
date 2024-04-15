Define a standard grid 
======================

Building models is much easier with standardized data. So define a standard grid (geographical projection, longitude range [-180,180], 0.25 degree resolution).
All data will be regridded to this grid before use.

There is one user-accessible variable in this file

* E5sCube - an iris cube on the standard grid (use as the target in regridding)

.. literalinclude:: ../../utilities/grids.py

