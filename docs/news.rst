News and Changelog
==============================

* 1.2.0 (23/11/22)
  
- Automatically attempts to extract labels from HuggingFace model config if left unspecified.
- Added new pipeline "token_classification_transformer" for token classification, including NER, POS and other types of token predictions

  - Automatically infer NER or POS from labels and assign these to `doc.ents` or `doc.pos` if possible.

- Renamed "classification_transformer" to "sequence_classification_transformer" to avoid confusion with new pipelines.

  - Automatically assign sequence prediction to `docs.cats`. Can be toggled off by setting the `"assign_to_cats": False`

- Update to documentation


* 1.1.0 (15/08/22)

  - Update following this `Pull request <https://github.com/explosion/spacy-transformers/pull/332>`__ on spacy-transformers allowing for alternate model loaders. This makes the code-base more stable towards future changes and removes existing monkeypatch.


* 1.0.0 (29/01/22)

  - Stable version of spacy-wrap launches
  
    * extended the test suite.
    * Added docstring to ensure credit is given to the parts of the code developed by the team at Explosion AI.  
    * Minor implementations details leading the pipeline to behave more consistently with the rest of the spaCy framework, including:
    
      * Adding an attribute to indicate that the pipeline can't be trained.
      * Added functionality to read and write the component to disk 💾



* 0.0.1 (29/01/22)

  - First version of spacy-wrap launches
  
    * Including wrappers for turning any classification transformer into a spacy transformer 🎉
    * Functionality for easily including multiple of these transformers 🌟
    * it even comes with a (minimal) documentation 📖

