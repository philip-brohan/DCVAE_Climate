Check normalization of ERA5 data
================================

.. figure:: ../../normalize/ERA5/monthly_precip_1969_03.png
   :width: 95%
   :align: center
   :figwidth: 95%

   Raw precipitation (top) and normalized equivalent (bottom) for a March 1969.

If successful, the normalized data should be approximately normally distributed with mean=0.5 and standard deviation=0.2 (whatever the distribution of the input data is. The script `plot_distribution_monthly.py` compares normalized and original data (spatial distribution and histogram) for a month. (Takes `--variable`, `--year`, and `--month` as arguments.)


Script to make the plot:

.. literalinclude:: ../../normalize/ERA5/plot_distribution_monthly.py
   