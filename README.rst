arrp
=========================================================================================
|travis| |sonar_quality| |sonar_maintainability| |codacy| |code_climate_maintainability| |pip| |downloads|

Simple python package to render the holdouts and training datasets of active regulatory regions for models with the task to predict them.

How do I install this package?
----------------------------------------------
As usual, just download it using pip:

.. code:: shell

    pip install arrp

Tests Coverage
----------------------------------------------
Since some software handling coverages sometime get slightly different results, here's three of them:

|coveralls| |sonar_coverage| |code_climate_coverage|

How do I get started?
----------------------------------------------
If you don't have it already, you will need to install the package bedtools_. 

Most commonly you just need to run the following:

.. code:: shell

    sudo apt install bedtools
    pip install arrp

How do I build the dataset?
---------------------------------------
From within the repo run:

.. code:: python

    from arrp import build
    build(target="dataset")

Where `"dataset"` is the path to your dataset. The default one is the one in the repository.

Which genome does it use?
----------------------------------------
By default it uses hg19_, as it is the genome used in the labeled data currently available from the Wasserman team. This is one of the numerous settings available.

Running the code
----------------------------
Once you have rendered the dataset you can run the following snippets:

CNN
----------

.. code:: python
   
   from arrp_NNs import cnn
   cnn(target="dataset")
  
MLP
----------

.. code:: python
   
   from arrp_NNs import mlp
   mlp(target="dataset")
   
Multi-modal neural network
------------------------------

.. code:: python
   
   from arrp_NNs import mmnn
   mmnn(target="dataset")



.. _hg19: https://www.ncbi.nlm.nih.gov/assembly/GCF_000001405.13/
.. _bedtools: https://bedtools.readthedocs.io/en/latest/
.. _here: https://github.com/LucaCappelletti94/wasserman/blob/master/info/bedtools.md

.. |travis| image:: https://travis-ci.org/LucaCappelletti94/arrp.png
   :target: https://travis-ci.org/LucaCappelletti94/arrp
   :alt: Travis CI build

.. |sonar_quality| image:: https://sonarcloud.io/api/project_badges/measure?project=LucaCappelletti94_arrp&metric=alert_status
    :target: https://sonarcloud.io/dashboard/index/LucaCappelletti94_arrp
    :alt: SonarCloud Quality

.. |sonar_maintainability| image:: https://sonarcloud.io/api/project_badges/measure?project=LucaCappelletti94_arrp&metric=sqale_rating
    :target: https://sonarcloud.io/dashboard/index/LucaCappelletti94_arrp
    :alt: SonarCloud Maintainability

.. |sonar_coverage| image:: https://sonarcloud.io/api/project_badges/measure?project=LucaCappelletti94_arrp&metric=coverage
    :target: https://sonarcloud.io/dashboard/index/LucaCappelletti94_arrp
    :alt: SonarCloud Coverage

.. |coveralls| image:: https://coveralls.io/repos/github/LucaCappelletti94/arrp/badge.svg?branch=master
    :target: https://coveralls.io/github/LucaCappelletti94/arrp?branch=master
    :alt: Coveralls Coverage

.. |pip| image:: https://badge.fury.io/py/arrp.svg
    :target: https://badge.fury.io/py/arrp
    :alt: Pypi project

.. |downloads| image:: https://pepy.tech/badge/arrp
    :target: https://pepy.tech/badge/arrp
    :alt: Pypi total project downloads 

.. |codacy|  image:: https://api.codacy.com/project/badge/Grade/4c74988d1fa84ab6a458ccba6eb0a19e
    :target: https://www.codacy.com/app/LucaCappelletti94/arrp?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=LucaCappelletti94/arrp&amp;utm_campaign=Badge_Grade
    :alt: Codacy Maintainability

.. |code_climate_maintainability| image:: https://api.codeclimate.com/v1/badges/a6be31e68bbb41d7de5a/maintainability
    :target: https://codeclimate.com/github/LucaCappelletti94/arrp/maintainability
    :alt: Maintainability

.. |code_climate_coverage| image:: https://api.codeclimate.com/v1/badges/a6be31e68bbb41d7de5a/test_coverage
    :target: https://codeclimate.com/github/LucaCappelletti94/arrp/test_coverage
    :alt: Code Climate Coverate
