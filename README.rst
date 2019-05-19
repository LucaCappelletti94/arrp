Active regulatory regions prediction dataset renderer
===================================================================
Simple python package to render the holdouts and training datasets of active regulatory regions for models with the task to predict them.

How do I get started?
----------------------------------------------
If you don't have it already, you will need to install the package bedtools_. A setup for both Linux and macOS can be found here_. 

Most commonly you just need to run, from within the repository:

.. code:: shell

    sudo apt install bedtools
    pip install .

How do I build the dataset?
---------------------------------------
Just run:

.. code:: python

    from arrp import build
    build(target="dataset")

Which genome does it use?
----------------------------------------
By default it uses hg19_, as it is the genome used in the labeled data currently available from the Wasserman team. This is one of the numerous settings available.


.. _hg19: https://www.ncbi.nlm.nih.gov/assembly/GCF_000001405.13/
.. _bedtools: https://bedtools.readthedocs.io/en/latest/
.. _here: https://github.com/LucaCappelletti94/wasserman/blob/master/info/bedtools.md
