Default model - make time-series by assimilating all test months into trained model
===================================================================================

.. figure:: ../../ML_models/default/assimilate_multi_T+P.webp
   :width: 95%
   :align: center
   :figwidth: 95%

   Global mean series (black original, red DCVAE output) and scatterplots for each output variable. T2m and MSLP were assimilated, Precipitation wasn't - it's model output.

Script (`assimilate_multi.py`) to make the validation figure

By default, it will use the test set, but the `--training` argument will take months from the training set instead of the test set. By default, it won't assimilate anything, specify variables to assimilate as arguments (so the figure above has arguments ``--T2m`` and ``--MSLP``).

.. literalinclude:: ../../ML_models/default/assimilate_multi.py

Utility functions used in the plot

.. literalinclude:: ../../ML_models/default/gmUtils.py




