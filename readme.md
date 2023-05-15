<a href="https://github.com/kennethenevoldsen/spacy-wrap"><img src="https://raw.githubusercontent.com/KennethEnevoldsen/spacy-wrap/main/docs/_static/icon.png" width="300" align="right" /></a>
# spaCy-wrap: For Wrapping fine-tuned transformers in spaCy pipelines

[![PyPI version](https://badge.fury.io/py/spacy-wrap.svg)](https://pypi.org/project/spacy-wrap/)
[![python version](https://img.shields.io/badge/Python-%3E=3.8-blue)](https://github.com/kennethenevoldsen/spacy-wrap)
[![Code style: black](https://img.shields.io/badge/Code%20Style-Black-black)](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html)
[![github actions pytest](https://github.com/kennethenevoldsen/spacy-wrap/actions/workflows/tests.yml/badge.svg)](https://github.com/kennethenevoldsen/spacy-wrap/actions)
[![github actions docs](https://github.com/kennethenevoldsen/spacy-wrap/actions/workflows/documentation.yml/badge.svg)](https://kennethenevoldsen.github.io/spacy-wrap/)
![github coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/KennethEnevoldsen/33fb85a2c440013df494c1fce884633c/raw/3813a0369fdd61b39a806b7b91839ff405ef809a/badge-spacy-wrap-coverage.json)


spaCy-wrap is a minimal library intended for wrapping fine-tuned transformers from the [Huggingface model hub](https://huggingface.co/models?pipeline_tag=text-classification&sort=downloads) in your spaCy pipeline allowing the inclusion of existing models within [SpaCy](https://spacy.io) workflows. 

As for as possible it follows a similar API as [spacy-transformers](https://github.com/explosion/spacy-transformers).

**NOTE**: Since the release of spaCy-wrap, Explosion released the [spacy-huggingface-pipelines](https://github.com/explosion/spacy-huggingface-pipelines) it takes the approach of wrapping the Huggingface pipeline as opposed to the transformer. That means token aggregation and conversion into spans happens at
the Huggingface pipeline, while in spaCy-wrap it happens at the logits of the model which can sometimes lead to unfortunate differences in results.
I generally recommend using the spacy-huggingface-pipelines for most use cases, but if you need to use the transformer output more directly 
spaCy-wrap can have its uses.

## Installation

Installing spacy-wrap is simple using pip:

```
pip install spacy_wrap
```

## Examples
The following shows a simple example of how you can quickly add a fine-tuned transformer model from the Huggingface model hub for either [text classification](https://huggingface.co/models?pipeline_tag=text-classification&sort=downloads), [named entity](https://huggingface.co/models?pipeline_tag=token-classification&sort=downloads) or [token classification](https://huggingface.co/models?pipeline_tag=token-classification&sort=downloads). 

### Sequence Classification
In this example, we will use a [model](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english) fine-tuned for sentiment classification on SST2. This model classifies whether a text is positive or negative. We will add this model to a blank English pipeline:


```python
import spacy
import spacy_wrap

nlp = spacy.blank("en")

config = {
    "doc_extension_trf_data": "clf_trf_data",  # document extention for the forward pass
    "doc_extension_prediction": "sentiment",  # document extention for the prediction
    "model": {
        # the model name or path of huggingface model
        "name": "distilbert-base-uncased-finetuned-sst-2-english",  
    },
}

transformer = nlp.add_pipe("sequence_classification_transformer", config=config)

doc = nlp("spaCy is a wonderful tool")

print(doc.cats)
# {'NEGATIVE': 0.001, 'POSITIVE': 0.999}
print(doc._.sentiment)
# 'POSITIVE'
print(doc._.clf_trf_data)
# TransformerData(wordpieces=...
```
These pipelines can also easily be applied to multiple documents using the `nlp.pipe` as one would expect from a spaCy component:

```python
docs = nlp.pipe(
    [
        "I hate wrapping my own models",
        "Isn't there a tool for this?!",
        "spacy-wrap is great for wrapping models",
    ]
)

for doc in docs:
    print(doc._.sentiment)
# 'NEGATIVE'
# 'NEGATIVE'
# 'POSITIVE'
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
    # choose custom labels
    "labels": ["Not hate Speech", "Hate speech"],
    "model": {
        "name": "DaNLP/da-bert-hatespeech-detection",  # the model name or path of huggingface model
    },
}

transformer = nlp.add_pipe("classification_transformer", config=config)

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


### Token Classification
We can also use the model for token classification: 

```python
import spacy
import spacy_wrap
nlp = spacy.blank("en")

config = {"model": {"name": "vblagoje/bert-english-uncased-finetuned-pos"}, 
          # "predictions_to": ["pos"]  # optional, can be "pos", "tag" or "ents"
}

snlp.add_pipe("token_classification_transformer", config=config)

text = "My name is Wolfgang and I live in Berlin"

doc = nlp(text)
print(doc._.tok_clf_predictions)
# ['PRON', 'NOUN', 'AUX', 'PROPN', 'CCONJ', 'PRON', 'VERB', 'ADP', 'PROPN']
```

By default, spacy-wrap will automatically detect it the labels follow the universal POS tags as well. If so it will also assign it to the `token.pos`, similar regular spacy pipelines:

```python
print(doc[0].pos_)
# 'PRON'
```

### Named Entity Recognition
In this example, we use a model fine-tuned for named entity recognition. spacy-wrap will in this case infer from the IOB tags that the model is intended for named entity recognition and assign it to `doc.ents`.

```python
import spacy
import spacy_wrap
nlp = spacy.blank("en")

# specify model from the hub
config = {"model": {"name": "dslim/bert-base-NER"}, 
          "predictions_to": ["ents"]} # forced to be named entity recognition, if left out it will be estimated from the labels

# add it to the pipe
nlp.add_pipe("token_classification_transformer", config=config)

doc = nlp("My name is Wolfgang and I live in Berlin.")

print(doc.ents)
# (Wolfgang, Berlin)
```

# 📖 Documentation

| Documentation              |                                             |
| -------------------------- | ------------------------------------------- |
| 🔧 **[Installation]**       | Installation instructions for spacy-wrap.   |
| 📰 **[News and changelog]** | New additions, changes and version history. |
| 🎛 **[Documentation]**      | The reference for spacy-wrap's API.         |

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

