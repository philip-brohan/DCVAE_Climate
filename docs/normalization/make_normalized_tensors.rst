Assemble ERA5 normalized data into a set of tf.tensors
======================================================

The data :doc:`download scripts <../get_data/ERA5>` assemble selected ERA5 data in netCDF files. To use that data efficiently in analysis and modelling it is necessary both to normalize it, and to reformat it as a set of `tf.tensors`. These have consistent format and resolution and can be reassembled into a `tf.data.Dataset`` for ML model training.

Script to make the set of tensors. Takes argument `--variable`, and uses :doc:`precalculated normalization parameters <estimate_parameters>`:

.. literalinclude:: ../../make_normalized_tensors/ERA5/make_all_tensors.py

Library functions to do the normalization:

.. literalinclude:: ../../normalize/ERA5/normalize.py

Library functions to convert between `tf.tensor`` and `iris.cube.cube`:

.. literalinclude:: ../../make_normalized_tensors/ERA5/tensor_utils.py
   