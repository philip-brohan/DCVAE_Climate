Setting up the software environment
===================================

To run the scripts in this repository, you will need to set up an environment with the necessary compute resources.

First, you will need to assign some disc space for the output files. The scripts in this repository will write a lot of data to disc, and you will need to have a few hundred gigabytes of free space available. They rely on an environment variable ``SCRATCH``.

* Set the ``SCRATCH`` environment variable to a directory with plenty of free disc space.

Then, you will need to install some software. The software is all open-source, and should be available for most operating systems (but it's only been tested on Linux-x86). By far the easiest way to do this is to use `conda <https://docs.conda.io/en/latest/>`_:

* Install `anaconda or miniconda <https://docs.conda.io/en/latest/>`_.

When you have conda installed, you can create an environment with all the necessary software by using the YML file in this repository:

* Edit the YML file ``DCVAE-Climate.yml`` to set the ``PYTHONPATH`` environment variable to the directory you have installed the code in.
* Create and activate the ``DCVAE-Climate`` environment specified in the YML file ``DCVAE-Climate.yml``. 

.. literalinclude:: ../environment/DCVAE-Climate.yml

