"""Test for the sequence classification transformer pipeline component."""
import shutil

import pytest
import spacy

EXAMPLES = []
EXAMPLES.append(
    (
        {
            "doc_extension_trf_data": "clf_trf_data",
            "doc_extension_prediction": "hate_speech",
            "labels": ["Not hate Speech", "Hate speech"],
            "model": {
                "name": "DaNLP/da-bert-hatespeech-detection",
            },
        },
        [("Senile gamle idiot", "Hate speech"), ("Jeg er glad", "Not hate Speech")],
    ),
)
EXAMPLES.append(
    (
        {
            "model": {
                "name": "DaNLP/da-bert-hatespeech-detection",
            },
        },
        [("Senile gamle idiot", "Hate speech"), ("Jeg er glad", "Not hate Speech")],
    ),
)
EXAMPLES.append(
    (
        {
            "model": {
                "name": "distilbert-base-uncased-finetuned-sst-2-english",
            },
        },
        [
            ("I like you. I love you", "POSITIVE"),
            ("I hate you. I dislike you", "NEGATIVE"),
        ],
    ),
)


class TestClassificationTransformer:
    @pytest.mark.parametrize("config, examples", EXAMPLES)
    def test_forward(self, config: dict, examples: list):
        """tests if that the forward pass work as intended."""

        nlp = spacy.blank("en")
        nlp.add_pipe("sequence_classification_transformer", config=config)
        nlp.initialize()

        for text, label in examples:
            doc = nlp(text)

            assert doc._.clf_trf_data
            prediction = getattr(doc._, config["doc_extension_prediction"])
            assert prediction == label
            prob = getattr(doc._, config["doc_extension_prediction"] + "_prob")
            assert isinstance(prob, dict)

    def test_to_and_from_disk(self):
        """tests if the pipeline can be serialized to disk."""
        nlp = spacy.blank("da")
        config = {
            "doc_extension_trf_data": "clf_trf_data",
            "doc_extension_prediction": "hate_speech",
            "labels": ["Not hate Speech", "Hate speech"],
            "model": {
                "name": "DaNLP/da-bert-hatespeech-detection",
            },
        }
        nlp.add_pipe("sequence_classification_transformer", config=config)
        nlp.initialize()

        transformer = nlp.get_pipe("classification_transformer")

        transformer.to_disk("spacy_wrap_serialization_test")
        transformer.from_disk("spacy_wrap_serialization_test")

        # remove test directory
        shutil.rmtree("spacy_wrap_serialization_test")
