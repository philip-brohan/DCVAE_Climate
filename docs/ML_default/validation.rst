Default model - validate trained model on test data for a single month
======================================================================

.. figure:: ../../ML_models/default/comparison.webp
   :width: 95%
   :align: center
   :figwidth: 95%

   Left-hand column is the target, middle column is the DCVAE output, right-hand column is a scatter plot of target and output.

To validate the autoencoder, we can run the trained model using a month from the test dataset as input: The output fields should look like the inputs. Use script `validate.py` to make the validation figure

By default, it will use a random month from the test set, but you can specify a month using the `--year` and `--month` arguments. The `--training` argument will take months from the training set instead of the test set, and `--epoch` will use a specific epoch.

.. literalinclude:: ../../ML_models/default/validate.py

Utility functions used in the plot

.. literalinclude:: ../../ML_models/default/gmUtils.py




