A Machine Learning Climate Model based on a Deep Convolutional Variational AutoEncoder
======================================================================================

This is an example of how to build a climate model using Machine Learning (ML). It's designed to be used as a training tool for those new to ML, but to be powerful enough to be adapted into something of value for science work.

Background
----------

Climate Scientists are mostly used to using physical models (`GCMs <https://www.ipcc-data.org/guidelines/pages/gcm_guide.html>`_). It is important to understand that ML models are quite a different thing - don't think of an ML model as a new variant of GCM.

* When building a GCM we specify how the model behaves. The aim is to make the model as close to physical reality as possible. The results produced by the model are then an emergent property, and they will be correct if the model is an adequate representation of the real world. So, with a GCM, the focus is on the model itself.

* When building an ML model, we specify what the model learns. The aim is to make the model as good as possible at predicting the data we have given it. The results produced by the model are then a prediction, and they will be correct if the model has learned the patterns in the data. So, with an ML model, the focus is on the data used in training, and the model is an emergent property - ML models do have to correspond to reality, but they don't have to do so in any way we'd recognize as physics.

So physical modellers care primarily about the physical correctness of their model, and only secondarily about their validation statistics. ML modellers care primarily about their validation statistics, and the issue of physical correctness does not arise. (Of course this is a simplification - real science is much more complex than this, but it is nonetheless an important distinction.)

A result of this is that GCMs are, in a sense, all trying to do the same thing - they are all trying to represent the same set of physical equations. ML models can be much more diverse - each is trying to predict its training data, and the training data can be anything. So GCMs are generic: one GCM - many applications. ML models are specific: we should expect to train a new model for each new problem. (It is possible to train generic ML models, but this is difficult, expensive, and will be less accurate than training a specific model. The majority of ML models used in climate will be problem-specific)

So the first task is to choose a problem. This model is built to study monthly precipitation - in particular to represent the spatial distribution of precipitation (to what extent can we infer the precipitation in London given the precipitation in Birmingham), and the relationship between precipitation and other variables (how does precipitation relate to 2m-air temperature and mean-sea-level pressure). This is a convenient example problem, but note that the model may well be of interest even if you don't care about precipitation: The model we are training here is precipitation-specific, but the model code and structure are much more generic - you could take the same model specification, and the same software, and train it on a different dataset, to make an ML model of use for your own work.

Design Decisions
----------------

Our aim, with a climate model, is typically to generate an estimate of some future climate state, so the obvious tool is a `generative model <https://en.wikipedia.org/wiki/Generative_artificial_intelligence>`_. There are already several well established ML methods for making generative models, here we will use a `Variational AutoEncoder <https://en.wikipedia.org/wiki/Autoencoder#Variational_autoencoder_(VAE)>`_ (VAE). The main virtue of the VAE is that it provides a specifically-constrained model state vector, which offers additional ways to control the model output, effectively allowing us to do data assimilation. See `this detailed discussion of the model design <http://brohan.org/Proxy_20CR/>`_ for details.

Our model is an example of a `Deep Convolutional Variational AutoEncoder <https://en.wikipedia.org/wiki/Convolutional_neural_network>`_ (DCVAE). This is a VAE with a convolutional neural network (CNN) as the encoder and decoder. The CNN is a good choice for this problem because it is good at learning spatial patterns. We will build the model using the `TensorFlow <https://www.tensorflow.org/>`_ deep learning framework.

Appendices
----------

.. toctree::
   :titlesonly:
   :maxdepth: 1

   Get the training data <get_data/index>
   Normalize the data for model fitting <normalization/index>
   ML model specification <ML_default/index>
   Utility functions for plotting and re-gridding <utils/index>


Small print
-----------

.. toctree::
   :titlesonly:
   :maxdepth: 1

   How to reproduce or extend this work <how_to>
   Authors and acknowledgements <credits>


  
This document is crown copyright (2024). It is published under the terms of the `Open Government Licence <https://www.nationalarchives.gov.uk/doc/open-government-licence/version/2/>`_. Source code included is published under the terms of the `BSD licence <https://opensource.org/licenses/BSD-2-Clause>`_.
