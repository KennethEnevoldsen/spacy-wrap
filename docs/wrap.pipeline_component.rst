pipeline Components
--------------------------------------------------

SpaCy-wrap currently includes only two pipeline component the
:code:`"sequence_classification_transformer"` for sequence classification and
the :code:`"token_classification_transformer"` for token classification or named
entity recognition. The components are implemented as a subclass of
:code:`spacy.pipeline.Pipe` and can be added to a spaCy pipeline using the
:code:`nlp.add_pipe` method. The components can be configured with a
:code:`config` dictionary.

Sequence Classification Transformer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. autofunction:: spacy_wrap.pipeline_component_seq_clf.make_sequence_classification_transformer


Sequence Classification Transformer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: spacy_wrap.pipeline_component_token_clf.make_token_classification_transformer

