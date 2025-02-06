ERA5 data download and access
=============================

Functions to access downloaded ERA5 data:

.. literalinclude:: ../../get_data/ERA5/ERA5_monthly.py

Script to download a year of ERA5 data:
The `netCDF <https://www.unidata.ucar.edu/software/netcdf/>`_ files provided by the `Copernicus Climate Data Store <https://cds.climate.copernicus.eu/cdsapp#!/home>`_ are no-longer `CF-compliant <https://cfconventions.org/>`_. This means that `iris <https://scitools.org.uk/iris/docs/latest/>`_ (which we use to read in the data) can't read them. The fix for this is something of a fudge - we use the `netcdf operators <http://nco.sourceforge.net/>`_ to delete a problematic attribute from the files (all included in the script below). This in turn means that reading the files will generate a warning about missing variables, but this is harmless.

.. literalinclude:: ../../get_data/ERA5/get_year_of_monthlies_from_ERA5.py

Script to download all the ERA5 data:

.. literalinclude:: ../../get_data/ERA5/get_data_for_period_ERA5.py




