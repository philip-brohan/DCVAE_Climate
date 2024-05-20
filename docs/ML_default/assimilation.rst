Default model - assimilate test data for a single month into trained model
==========================================================================

.. figure:: ../../ML_models/default/assimilated_T+P.webp
   :width: 95%
   :align: center
   :figwidth: 95%

   Left-hand column is the target, middle column is the DCVAE output assimilating T2m and MSLP, right-hand column is a scatter plot of target and output.

The objective is not just a good autoencoder, but a generator that is useful in making new output fields. The script `assimilate.py` generates new months using the trained model.

By default, it will use a random month from the test set, but you can specify a month using the `--year` and `--month` arguments. The `--training` argument will take months from the training set instead of the test set. By default, it won't assimilate anything, specify variables to assimilate as arguments (so the figure above has arguments ``--T2m`` and ``--MSLP``). Note that if you don't assimilate anything - the model output won't look like the model input. `--epoch` uses the model from a specific epoch.

.. literalinclude:: ../../ML_models/default/assimilate.py

Utility functions used in the plot

.. literalinclude:: ../../ML_models/default/gmUtils.py




