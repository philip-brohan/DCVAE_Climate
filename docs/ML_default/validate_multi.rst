Default model - validate trained model on time-series of all test months
========================================================================

.. figure:: ../../ML_models/default/multi.webp
   :width: 95%
   :align: center
   :figwidth: 95%

   Global mean series (black original, red DCVAE output) and scatterplots for each output variable. Top panel shows regions used for training. Bottom panel shows regions masked from training.

Script (`validate_multi.py`) to make the validation figure

By default, it will use the test set, but the `--training` argument will take months from the training set instead of the test set.

.. literalinclude:: ../../ML_models/default/validate_multi.py

Utility functions used in the plot

.. literalinclude:: ../../ML_models/default/gmUtils.py




