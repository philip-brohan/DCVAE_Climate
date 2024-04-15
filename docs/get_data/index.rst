Get the data to be used
=======================

We are going to use monthly averaged surface variables from ERA5.

Four variables:
 * 2m_temperature
 * sea_surface_temperature
 * mean_sea_level_pressure
 * total_precipitation

We also want a land-sea mask (for plotting only). Use a land-surface only variable from ERA5-land for this (we only need one month).

We can download all this from the awesome `Copernicus Climate Data Store <https://cds.climate.copernicus.eu/cdsapp#!/home>`_

Script to do the whole download (about 8Gb, will take a few hours). Only downloads data where it is not already on disc.

.. literalinclude:: ../../get_data/download_all_data.sh

.. toctree::
   :titlesonly:
   :maxdepth: 1

   Download and access ERA5 data <ERA5>
   Download the land mask <land_mask>



