Frequently asked questions
================================


Citing spacy-wrap
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you wish this library in your research, please cite it using:

.. code-block::

   @inproceedings{spacywrap2022,
      title={spaCy-wrap: Wrapper for including pre-trained transformers in spaCy},
      author={Enevoldsen, Kenneth},
      year={2022}
   }


How do I test the code and run the test suite?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This package comes with an extensive test suite. In order to run the tests,
you'll usually want to clone the repository and build the package from the
source. This will also install the required development dependencies
and test utilities defined in the `requirements.txt <https://github.com/KennethEnevoldsen/spacy-wrap/blob/master/requirements.txt>`__.


.. code-block:: bash

   pip install -r requirements.txt
   pip install pytest

   python -m pytest


which will run all the test in the `spacy-wrap/tests` folder.

Specific tests can be run using:

.. code-block:: bash

   python -m pytest spacy-wrap/tests/desired_test.py



How is the documentation generated?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

SpaCy-wrap uses `sphinx <https://www.sphinx-doc.org/en/master/index.html>`__ to generate
documentation. It uses the `Furo <https://github.com/pradyunsg/furo>`__ theme
with custom styling.

To make the documentation you can run:

.. code-block:: bash

   # install sphinx, themes and extensions
   pip install -r requirements.txt

   # generate html from documentations

   make -C docs html
