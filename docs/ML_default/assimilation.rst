Default model - assimilate test data for a single month into trained model
==========================================================================

.. figure:: ../../ML_models/default/assimilated.webp
   :width: 95%
   :align: center
   :figwidth: 95%

   Left-hand column is the target, next column is the DCVAE output assimilating T2m and MSLP, right-hand columns is a scatter plot of target and output.

Script (`assimilate.py`) to make the figure

By default, it will use a random month from the test set, but you can specify a month using the `--year` and `--month` arguments. The `--training` argument will take months from the training set instead of the test set. By default it won't assimilate anything, specify variables to assimilate as arguments (so the figure above has agguments ``--T2m`` and ``--MSLP``).

.. literalinclude:: ../../ML_models/default/assimilate.py

Utility functions used in the plot

.. literalinclude:: ../../ML_models/default/gmUtils.py




