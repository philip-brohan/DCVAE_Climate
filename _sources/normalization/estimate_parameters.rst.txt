Estimate normalization parameters for ERA5 data
===============================================

.. figure:: ../../normalize/ERA5/gamma.png
   :width: 65%
   :align: center
   :figwidth: 95%

   Fitted gamma distribution parameters for temperature data. Top\: location, centre\: shape, bottom\: scale.

Script to make normalization parameters for one calendar month. Takes arguments `--month`, `-variable`, `--startyear`, and `--endyear`:

.. literalinclude:: ../../normalize/ERA5/fit_for_month.py

The data are taken from the `tf.tensor` datasets of raw data created during the normalization process. Functions to present these as `tf.data.DataSets`:

.. literalinclude:: ../../normalize/ERA5/makeDataset.py

Script to plot the fitted gamma parameters (produces figure at top of page):

.. literalinclude:: ../../normalize/ERA5/plot_gamma_fit.py
   