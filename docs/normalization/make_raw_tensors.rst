Assemble ERA5 raw data into a set of tf.tensors
===============================================

The data :doc:`download scripts <../get_data/ERA5>` assemble selected ERA5 data in netCDF files. To use that data efficiently in analysis and modelling it is necessary to reformat it as a set of `tf.tensors`. These have consistent format and resolution and can be reassembled into a `tf.data.Dataset` for ML model training.

So for each month in the training period, for each variable (2m_temperature, mean_sea_level_pressure, total_precipitation), we read in the data from netCDF, regrid it to a :doc:`common grid <../utils/grids>`, and save it as a `tf.tensor`. 

The script `make_all_raw_tensors.sh` creates a set of commands to make all the tensors. The script outputs a list of other scripts (one per year, month, variable). Running all the output scripts will create the set of tensors. (Use `GNU parallel` to run the scripts efficiently - or submit them as jobs to a cluster).

.. literalinclude:: ../../make_raw_tensors/make_all_raw_tensors.sh

When the main script has completed and all the raw tensors are made, we need to add some metadata to them (used by subsequent scripts to find out how much data is available). The script ```update_tensor_metadata.sh``` does this. Run this script to set the metadata and check that the tensors have been created successfully.

.. literalinclude:: ../../make_raw_tensors/update_tensor_metadata.sh

Other scripts used by that main script:

Script to make the set of tensors for one variable. Takes argument `--variable`:

.. literalinclude:: ../../make_raw_tensors/ERA5/make_all_tensors.py

Calls another script to make a single tensor:

.. literalinclude:: ../../make_raw_tensors/ERA5/make_training_tensor.py

Library functions to convert between `tf.tensor` and `iris.cube.cube`:

.. literalinclude:: ../../make_raw_tensors/ERA5/tensor_utils.py

Metadata update script for an ERA5 variable:

.. literalinclude:: ../../make_raw_tensors/ERA5/update_tensor_metadata.py
   