Estimate normalization parameters for ERA5 data
===============================================

.. figure:: ../../normalize/ERA5/gamma_2m_temperature_m03.png
   :width: 65%
   :align: center
   :figwidth: 95%

   Fitted gamma distribution parameters for temperature data. Top\: location, centre\: shape, bottom\: scale (:doc:`details <validate_for_fields>`.

To normalize the data we are fitting `gamma distributions <https://statisticsbyjim.com/probability/gamma-distribution/>`_ to the training data. We fit a separate distribution at every grid-point, for each calendar month (so 1 distribution for each point, for all January data, and so on).

The scripts to estimate the fits are in the `normalize` directory. The script `normalize_all.sh` creates a set of commands to estimate all the fits. The script outputs a list of other scripts (one per year, month, variable). Running all the output scripts will do the estimation. (Use `GNU parallel` to run the scripts efficiently - or submit them as jobs to a cluster).

.. literalinclude:: ../../normalize/normalize_all.sh

Other scripts used by that main script:

Script to make normalization parameters for one calendar month. Takes arguments `--month`, `-variable`, `--startyear`, and `--endyear`:

.. literalinclude:: ../../normalize/ERA5/fit_for_month.py

The data are taken from the `tf.tensor` datasets of raw data created during the normalization process. Functions to present these as `tf.data.DataSets`:

.. literalinclude:: ../../normalize/ERA5/makeDataset.py

   