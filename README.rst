Active regulatory regions prediction dataset renderer
===================================================================
Simple python tool to render dataset that can be used for training models for active regulatory regions prediction.

How to get the dataset?
--------------------------
Just clone the repo.

How to get the package?
---------------------------
Just type into your terminal from within the repository:

.. code:: shell

   pip install .

Which genome does it use by default?
----------------------------------------
By default it uses hg19_, as it is the genome used in the labeled data currently available from the Wasserman team.

Dependencies
------------------------------
This package will use the package bedtools_ to elaborate the bed files. A setup for the package is available here_.

Building the dataset
-----------------------------
Just type into your terminal:

.. code:: shell

   python build.py



.. _hg19: https://www.ncbi.nlm.nih.gov/assembly/GCF_000001405.13/
.. _bedtools: https://bedtools.readthedocs.io/en/latest/
.. _here: https://github.com/LucaCappelletti94/wasserman/blob/master/info/bedtools.md
