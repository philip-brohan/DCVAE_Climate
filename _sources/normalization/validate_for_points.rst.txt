Check gamma distributions for ERA5 data
=======================================

.. figure:: ../../normalize/ERA5/samples_2m_temperature_m03.png
   :width: 95%
   :align: center
   :figwidth: 95%

   Data histograms and fitted gamma distributions for 25 random grid points of ERA5 March temperature.

It's worth checking that the gamma distributions fitted to the training data are sensible. (They don't have to be very good, we don't have much data to fit to, but they do have to be reasonable, and there's nothing like a quick graphical check to see if they are.) The script `plot_sample_fits.py` does this for 25 random grid points. It takes `--variable` and `--month` as arguments.

Script to make the plot:

.. literalinclude:: ../../normalize/ERA5/plot_sample_fits.py
   