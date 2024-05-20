Default model - validate trained model on time-series of all test months
========================================================================

.. figure:: ../../ML_models/default/multi.webp
   :width: 95%
   :align: center
   :figwidth: 95%

   Global mean series (black original, red DCVAE output) and scatterplots for each output variable.

We can test the time-consistency of the trained autoencoder by running it on all months in the test dataset, and plotting the time-series of global means. The script `validate_multi.py` makes this validation figure

By default, it will use the test set, but the `--training` argument will take months from the training set instead of the test set. `--epoch` will specify the epoch to use.

.. literalinclude:: ../../ML_models/default/validate_multi.py

Utility functions used in the plot

.. literalinclude:: ../../ML_models/default/gmUtils.py




