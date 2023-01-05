Frequently asked questions
================================


Citing spacy-wrap
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you wish this library in your research, please cite it using (Changing the version if relevant):

.. code-block::

   @software{Enevoldsen_spaCy-wrap_For_Wrapping_2022,
      author = {Enevoldsen, Kenneth},
      doi = {10.5281/zenodo.6675315},
      month = {8},
      title = {{spaCy-wrap: For Wrapping fine-tuned transformers in spaCy pipelines}},
      url = {https://github.com/KennethEnevoldsen/spacy-wrap},
      version = {1.0.2},
      year = {2022}
   }


Or if you prefer APA:

.. code-block:: 

   Enevoldsen, K. (2022). spaCy-wrap: For Wrapping fine-tuned transformers in spaCy pipelines (Version 1.0.2) [Computer software]. https://doi.org/10.5281/zenodo.6675315



How do I test the code and run the test suite?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This package comes with an extensive test suite. In order to run the tests,
you'll usually want to clone the repository and build the package from the
source. This will also install the required development dependencies
and test utilities defined in the `requirements.txt <https://github.com/KennethEnevoldsen/spacy-wrap/blob/master/requirements.txt>`__.


.. code-block:: bash

   pip install -e ".[tests]"

   python -m pytest


which will run all the test in the `tests` folder.

Specific tests can be run using:

.. code-block:: bash

   python -m pytest tests/desired_test.py



How is the documentation generated?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

SpaCy-wrap uses `sphinx <https://www.sphinx-doc.org/en/master/index.html>`__ to generate
documentation. It uses the `Furo <https://github.com/pradyunsg/furo>`__ theme
with custom styling.

To make the documentation you can run:

.. code-block:: bash

   # install sphinx, themes and extensions
   pip install -e ".[docs]"

   # generate html from documentations

   make -C docs html
