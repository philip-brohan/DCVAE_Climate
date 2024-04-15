Normalizing training data
=========================

.. figure:: ../../normalize/ERA5/monthly.png
   :width: 95%
   :align: center
   :figwidth: 95%

   Raw temperature (top) and normalized equivalent (bottom) for a month.


.. toctree::
   :titlesonly:
   :maxdepth: 1

    Assemble the raw data <make_raw_tensors>
    Validating gamma fits for selected points <validate_for_points>
    Estimating normalization parameters from raw data <estimate_parameters>
    Validating normalization for a selected month <validate_for_month>
    Assemble the normalized data <make_normalized_tensors>

Script to make all the normalization parameters:

.. literalinclude:: ../../normalize/ERA5/make_all_fits.py

