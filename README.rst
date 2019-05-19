Active regulatory regions prediction dataset renderer
===================================================================
|travis| |sonar_quality| |sonar_maintainability| |sonar_coverage| |code_climate_maintainability| |pip|

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

.. |travis| image:: https://travis-ci.org/LucaCappelletti94/arrp.png
   :target: https://travis-ci.org/LucaCappelletti94/arrp

.. |sonar_quality| image:: https://sonarcloud.io/api/project_badges/measure?project=LucaCappelletti94_arrp&metric=alert_status
    :target: https://sonarcloud.io/dashboard/index/LucaCappelletti94_arrp

.. |sonar_maintainability| image:: https://sonarcloud.io/api/project_badges/measure?project=LucaCappelletti94_arrp&metric=sqale_rating
    :target: https://sonarcloud.io/dashboard/index/LucaCappelletti94_arrp

.. |sonar_coverage| image:: https://sonarcloud.io/api/project_badges/measure?project=LucaCappelletti94_arrp&metric=coverage
    :target: https://sonarcloud.io/dashboard/index/LucaCappelletti94_arrp

.. |code_climate_maintainability| image:: https://api.codeclimate.com/v1/badges/25fb7c6119e188dbd12c/maintainability
   :target: https://codeclimate.com/github/LucaCappelletti94/arrp/maintainability
   :alt: Maintainability

.. |bases| image:: https://github.com/LucaCappelletti94/arrp/blob/master/bases.png?raw=true
   :alt: Bases

.. |kmers| image:: https://github.com/LucaCappelletti94/arrp/raw/master/kmers.png
   :alt: Kmers

.. |pip| image:: https://badge.fury.io/py/arrp.svg
    :target: https://badge.fury.io/py/arrp
