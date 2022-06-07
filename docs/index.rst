SpaCy-wrap
================================

.. image:: https://img.shields.io/github/stars/kennethenevoldsen/spacy-wrap.svg?style=social&label=Star&maxAge=2592000
   :target: https://github.com/kennethenevoldsen/spacy-wrap


spaCy-wrap is minimal library intended for wrapping fine-tuned transformers from the
`Huggingface model hub <https://huggingface.co/models?pipeline_tag=text-classification&sort=downloads>`__
in your spaCy pipeline allowing inclusion of existing models within `SpaCy <https://spacy.io>`__ workflows. 

As for as possible it follows a similar API as `spacy-transformers <https://github.com/explosion/spacy-transformers>`__.



Contents
---------------------------------
  
This documentation is intentionally minimal given the small codebase, for more on spacy-transformers,
architectures and pipeline components, do be sure to check out the `spaCy documentation <https://spacy.io>`__

- **Getting started** contains the installation instructions, guides, and tutorials on how to use spacy-wrap.
- **API references** contains the documentation of each function and public class.

.. toctree::
   :maxdepth: 3
   :caption: Getting started

   installation
   faq

.. toctree::
   :caption: News

   news

.. toctree::
   :caption: FAQ



.. toctree::
   :maxdepth: 3
   :caption: API references

   wrap.architectures
   wrap.layers
   wrap.pipeline_component   


.. toctree::
   :caption: GitHub

   GitHub Repository <https://github.com/kennethenevoldsen/spacy-wrap>


Indices and search
==================

* :ref:`genindex`
* :ref:`search`
