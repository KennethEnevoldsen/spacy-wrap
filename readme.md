<a href="https://github.com/kennethenevoldsen/spacy-wrap"><img src="https://raw.githubusercontent.com/KennethEnevoldsen/spacy-wrap/main/docs/_static/icon.png" width="300" align="right" /></a>
# spaCy-wrap: For Wrapping fine-tuned transformers in spaCy pipelines


[![PyPI version](https://badge.fury.io/py/spacy-wrap.svg)](https://pypi.org/project/spacy-wrap/)
[![python version](https://img.shields.io/badge/Python-%3E=3.7-blue)](https://github.com/kennethenevoldsen/spacy-wrap)
[![Code style: black](https://img.shields.io/badge/Code%20Style-Black-black)](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html)
[![github actions pytest](https://github.com/kennethenevoldsen/spacy-wrap/actions/workflows/pytest-cov-comment.yml/badge.svg)](https://github.com/kennethenevoldsen/spacy-wrap/actions)
[![github actions docs](https://github.com/kennethenevoldsen/spacy-wrap/actions/workflows/documentation.yml/badge.svg)](https://kennethenevoldsen.github.io/spacy-wrap/)
![github coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/KennethEnevoldsen/33fb85a2c440013df494c1fce884633c/raw/3813a0369fdd61b39a806b7b91839ff405ef809a/badge-spacy-wrap-coverage.json)
[![CodeFactor](https://www.codefactor.io/repository/github/kennethenevoldsen/spacy-wrap/badge)](https://www.codefactor.io/repository/github/kennethenevoldsen/spacy-wrap)
<!-- [![pip downloads](https://img.shields.io/pypi/dm/spacy_wrap.svg)](https://pypi.org/project/spacy_wrap/) -->


## Installation

Installing spacy-wrap is simple using pip:

```
pip install spacy_wrap
```

There is no reason to update from GitHub as the version on PyPI should always be the same as on GitHub.

## Example
The following shows a simple example of how you can quickly add a fine-tuned transformer model from the [Huggingface model hub](https://huggingface.co/models?pipeline_tag=text-classification&sort=downloads).  In this example we will use the sentiment model by [Barbieri et al. (2020)](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment) for classifying whether a tweet is positive, negative or neutral. We will add this model to a blank English pipeline:

```python
import spacy
import spacy_wrap

nlp = spacy.blank("en")

config = {
    "doc_extension_trf_data": "clf_trf_data",  # document extention for the forward pass
    "doc_extension_prediction": "sentiment",  # document extention for the prediction
    "labels": ["negative", "neutral", "positive"],
    "model": {
        "name": "cardiffnlp/twitter-roberta-base-sentiment",  # the model name or path of huggingface model
    },
}

transformer = nlp.add_pipe("classification_transformer", config=config)
transformer.model.initialize()

doc = nlp("spaCy is a wonderful tool")

print(doc._.clf_trf_data)
# TransformerData(wordpieces=...
print(doc._.sentiment)
# 'positive'
print(doc._.sentiment_prob)
#{'prob': array([0.004, 0.028, 0.969], dtype=float32), 'labels': ['negative', 'neutral', 'positive']}
```

These pipelines can also easily be applied to multiple documents using the `nlp.pipe` as one would expect from a spaCy component:

```python
docs = nlp.pipe(
    [
        "I hate wrapping my own models",
        "Isn't there a tool for this?",
        "spacy-wrap is great for wrapping models",
    ]
)

for doc in docs:
    print(doc._.sentiment)
# 'negative'
# 'neutral'
# 'positive'
```

 <br /> 

<details>
  <summary> More Examples </summary>

It is always nice to have more than one example. Here is another one where we add the Hate speech model for Danish to a blank Danish pipeline:

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

</details>

 <br /> 



# 📖 Documentation

| Documentation              |                                                                               |
| -------------------------- | ----------------------------------------------------------------------------- |
| 🔧 **[Installation]**       | Installation instructions for spacy-wrap.                                  |
| 📰 **[News and changelog]** | New additions, changes and version history.                                   |
| 🎛 **[Documentation]**      | The reference for spacy-wrap's API. |

[Documentation]: https://kennethenevoldsen.github.io/spacy-wrap/index.html
[Installation]: https://kennethenevoldsen.github.io/spacy-wrap/installation.html
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

