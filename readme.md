<a href="https://github.com/kennethenevoldsen/spacy-wrap"><img src="https://github.com/KennethEnevoldsen/spacy-wrap/blob/main/docs/img/logo_black_font.png?raw=true" width="300" align="right" /></a>
# spaCy-wrap: For Wrapping fine-tuned transformers in spaCy pipelines


[![PyPI version](https://badge.fury.io/py/spacy-wrap.svg)](https://pypi.org/project/spacy-wrap/)
[![python version](https://img.shields.io/badge/Python-%3E=3.7-blue)](https://github.com/kennethenevoldsen/spacy-wrap)
[![Code style: black](https://img.shields.io/badge/Code%20Style-Black-black)](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html)
[![github actions pytest](https://github.com/kennethenevoldsen/spacy-wrap/actions/workflows/pytest-cov-comment.yml/badge.svg)](https://github.com/kennethenevoldsen/spacy-wrap/actions)
[![github actions docs](https://github.com/kennethenevoldsen/spacy-wrap/actions/workflows/documentation.yml/badge.svg)](https://kennethenevoldsen.github.io/spacy-wrap/)
![github coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/KennethEnevoldsen/33fb85a2c440013df494c1fce884633c/raw/5b4c0725d8bd63c8d76bb40c0d08a46077fa9d4e/badge-spacy-wrap-coverage.json)
[![CodeFactor](https://www.codefactor.io/repository/github/kennethenevoldsen/spacy-wrap/badge)](https://www.codefactor.io/repository/github/kennethenevoldsen/spacy-wrap)
<!-- [![pip downloads](https://img.shields.io/pypi/dm/spacy_wrap.svg)](https://pypi.org/project/spacy_wrap/) -->


## Installation

Installing spacy-wrap is simple using pip:

```
pip install spacy_wrap
```

There is no reason to update from GitHub as the version on PyPI should always be the same as on GitHub.

## Simple Example
The following shows a simple example of how you can quickly add a finetuned transformer model from the Huggingface model hub. In this case we will add the
Hate speech model for Danish to a blank Danish pipeline.

```python
import spacy
import spacy_wrap

nlp = spacy.blank("da")

config = {
    "doc_extension_trf_data": "clf_trf_data",  # document extention for the forward pass
    "doc_extension_prediction": "hate_speech",  # document extention for the prediction
    "labels": ["Not hate Speech", "Hate speech"],
    "model": {
        "name": "DaNLP/da-bert-hatespeech-detection",  # the model name or path of huggingface model
    },
}

transformer = nlp.add_pipe("classification_transformer", config=config)
transformer.model.initialize()

doc = nlp("Senile gamle idiot") # old senile idiot

doc._.clf_trf_data
# TransformerData(wordpieces=...
doc._.hate_speech
# "Hate speech"
doc._.hate_speech_prob
# {'prob': array([0.013, 0.987], dtype=float32), 'labels': ['Not hate Speech', 'Hate speech']}
```


# 📖 Documentation

| Documentation              |                                                                           |
| -------------------------- | ------------------------------------------------------------------------- |
| 🔧 **[Installation]**       | Installation instructions for spacy-wrap                                       |
| 📚 **[Usage Guides]**       | Guides and instructions on how to use spacy-wrap and its features.             |
| 📰 **[News and changelog]** | New additions, changes and version history.                               |
| 🎛 **[Documentation]**      | The detailed reference for spacy-wrap's API. Including function documentation |

[Documentation]: https://kennethenevoldsen.github.io/spacy-wrap/index.html
[Installation]: https://kennethenevoldsen.github.io/spacy-wrap/installation.html
[usage guides]: https://kennethenevoldsen.github.io/spacy-wrap/introduction.html
[News and changelog]: https://kennethenevoldsen.github.io/spacy-wrap/news.html

# 💬 Where to ask questions

| Type                           |                        |
| ------------------------------ | ---------------------- |
| 🚨 **FAQ**                      | [FAQ]                  |
| 🚨 **Bug Reports**              | [GitHub Issue Tracker] |
| 🎁 **Feature Requests & Ideas** | [GitHub Issue Tracker] |
| 👩‍💻 **Usage Questions**          | [GitHub Discussions]   |
| 🗯 **General Discussion**       | [GitHub Discussions]   |


[FAQ]: https://kennethenevoldsen.github.io/spacy-wrap/faq.html
[github issue tracker]: https://github.com/kennethenevoldsen/spacy-wrap/issues
[github discussions]: https://github.com/kennethenevoldsen/spacy-wrap/discussions

