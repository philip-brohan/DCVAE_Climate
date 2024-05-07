How to reproduce and extend this work
=====================================

To reproduce and extent this work, go through the following steps.

#. Get a `GitHub <https://github.com/>`_ account (it's free).
#. `Fork <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo>`_ `this repository <https://github.com/philip-brohan/DCVAE_Climate>`_. That is - make your own copy of it in your GitHub account.
#. `Clone <https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository>`_ your fork to your local machine.
#. (You don't have to do the steps above if you just want to download the code and data to run it locally. You can download the whole thing as `a zip file <https://github.com/philip-brohan/DCVAE_Climate/archive/refs/heads/main.zip>`_. But I do strongly recommend them - it makes it much easier to keep track of what you've done, and to share your work with others.)
#. :doc:`Set up the necessary software environment <environment>`.
#. Run the scripts to reproduce the work:
    #. :doc:`Download the data <get_data/index>`
    #. :doc:`Normalise the data <normalization/index>`
    #. :doc:`Train the model <ML_default/training>`
    #. :doc:`Evaluate the model <ML_default/validation>`
    #. :doc:`Assimilate temperature and pressure to calculate precipitation <ML_default/assimilation>`
#. Now you are ready to extend the work. You can try:
    #. Retraining the model on different data.
    #. Changing the model hyperparameters - maybe training rate or beta.
    #. Changing the model: More layers? Different activation functions? Different loss function? More features?
    #. Whatever - be creative. And share your results!


..
   I Want the sub-pages of this page only accessible by inline links - so :hidden: here.

.. toctree::
   :hidden:

   environment
