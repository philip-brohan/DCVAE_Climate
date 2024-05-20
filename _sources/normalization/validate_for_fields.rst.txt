Check gamma parameters for ERA5 data
====================================

.. figure:: ../../normalize/ERA5/gamma_2m_temperature_m03.png
   :width: 95%
   :align: center
   :figwidth: 95%

   Fitted gamma distribution parameters for temperature data. Top: location, centre: shape, bottom: scale.

To check the gamma distribution fit worked, it's useful to plot the estimated parameters. The script `plot_gamma_fit.py` does this. (arguments are `--variable` and `--month``).

Script to make the plot:

.. literalinclude:: ../../normalize/ERA5/plot_gamma_fit.py
   