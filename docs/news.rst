News and Changelog
==============================

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

