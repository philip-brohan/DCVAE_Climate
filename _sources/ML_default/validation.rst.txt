Default model - validate trained model on test data for a single month
======================================================================

.. figure:: ../../ML_models/default/comparison.webp
   :width: 95%
   :align: center
   :figwidth: 95%

   Left-hand column is the target, next column is the DCVAE output, right-hand two columns are scatter plots of target and output - the first one is for regions with training data, the second for regions without.

Script (`validate.py`) to make the validation figure

By default, it will use a random month from the test set, but you can specify a month using the `--year` and `--month` arguments. The `--training` argument will take months from the training set instead of the test set.

.. literalinclude:: ../../ML_models/default/validate.py

Utility functions used in the plot

.. literalinclude:: ../../ML_models/default/gmUtils.py




