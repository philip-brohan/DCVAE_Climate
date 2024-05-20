Default model
=============

The purpose of the model, is to learn a generator function that makes climate states (monthly fields of temperature, pressure and precipitation) with a low-dimensional state vector (embedding) as input. This generator is the climate model that e will use for experiments. To learn that model (and embedding) a standard tool in ML - a factory for learning such functions from example climate states.

The factory is a Deep Convolutional `Variational Autoencoder <https://en.wikipedia.org/wiki/Variational_autoencoder>`_. It takes :doc:`input data <make_dataset>` on a 721x1440 grid (same as ERA5), and uses a pair of convolutional neural nets to encode and decode the data - learning to both produce a target output on the same grid and make the encoded version (the embedding) distributed as a unit normal. The :doc:`structure of the model <VAE>` (12 convolutional layers in the encoder, 11 in the decoder) is fixed, but input and output data, and the hyperparameters (learning rate, batch size, beta, etc) can be changed. They are set in a :doc:`specification file <specify>` (`specify.py`).

.. figure:: ../Illustrations/Slide3.PNG
   :width: 95%
   :align: center
   :figwidth: 95%

   The structure of the VAE used to train the generator


We will code the DCVAE using the `TensorFlow <https://www.tensorflow.org/>` platform. The big advantage of TensorFlow is that a lot of the work has been done for us: The model structure is a subclass of `tf.keras.Model <https://www.tensorflow.org/api_docs/python/tf/keras/Model>`_, and there is an `example of a VAE <https://www.tensorflow.org/tutorials/generative/cvae>`_ in the TensorFlow documentation which we can build upon. Building on that, we can specify a DCVAE class for our climate data:

.. toctree::
   :titlesonly:
   :maxdepth: 1

   Model structure <VAE>

The model is trained on monthly fields of temperature, pressure and precipitation. We have :doc:`taken these fields from ERA5 <../get_data/index>`, and :doc:`converted and normalised them <../normalization/index>` already. What remains is to package them for input to the DCVAE. Training the DCVAE will mean we need to access the fields repeatedly and fast, and TensorFlow has software tools optimized for exactly this use: We will use the `tf.data.Dataset <https://www.tensorflow.org/api_docs/python/tf/data/Dataset>`_ class to present the training data to the DCVAE.

.. toctree::
   :titlesonly:
   :maxdepth: 1

   Input Datasets <make_dataset>

To run the model, we need to specify the model inputs, outputs and hyperparameters. This is done in a specification file. Copy the specification file (`specify.py`) and the model training script to a new directory, and edit the specification file to suit your needs. Then run the training script (`autoencoder.py`). 

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

Those scripts train and test the AutoEncoder. We are using a `Variational Autoencoder <https://en.wikipedia.org/wiki/Variational_autoencoder>`_, because we want to be able to use its generator as a climate model, and particularly to `assimilate data into it <https://brohan.org/Proxy_20CR>`_.

.. toctree::
   :titlesonly:
   :maxdepth: 1

   Assimilate a single month <assimilation>
   Assimilate a time-series <assimilate_multi>

