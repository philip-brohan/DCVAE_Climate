Default model
=============

The model is a Deep Convolutional `Variational Autoencoder <https://en.wikipedia.org/wiki/Variational_autoencoder>`_. It takes :doc:`input data <make_dataset>` on a 721x1440 grid (same as ERA5), and uses a pair of convolutional neural nets to encode and decode the data - learning to both produce a target output on the same grid and make the encoded version (the embedding) distributed as a unit normal. The :doc:`structure of the model <VAE>` (12 convolutional layers in the encoder, 11 in the decoder) is fixed, but input and output data, and the hyperparameters (learning rate, batch size, beta, etc) can be changed. They are set in a :doc:`specification file <specify>`.

.. figure:: ../Illustrations/Slide3.PNG
   :width: 95%
   :align: center
   :figwidth: 95%

   The structure of the VAE used to train the generator



The model structure and data flows are defined in files that should not need modification:

.. toctree::
   :titlesonly:
   :maxdepth: 1

   Model structure <VAE>
   Input Datasets <make_dataset>

To run the model, we need to specify the model inputs, outputs and hyperparameters. This is done in a specification file. Copy the specification file and the model training script to a new directory, and edit the specification file to suit your needs. Then run the training script. 

.. toctree::
   :titlesonly:
   :maxdepth: 1

   Specification file <specify>
   Training script <training>

And to test the training success there are scripts for plotting the training history and for testing the model on a test dataset.

.. toctree::
   :titlesonly:
   :maxdepth: 1

   Plot training_history <plot_history>
   Validate on single month <validation>
   Validate on time-series <validate_multi>

We are using a `Variational Autoencoder <https://en.wikipedia.org/wiki/Variational_autoencoder>`_, because we want to be able to `assimilate data into it <https://brohan.org/Proxy_20CR>`_.

.. toctree::
   :titlesonly:
   :maxdepth: 1

   Assimilate a single month <assimilation>
   Assimilate a time-series <assimilate_multi>

