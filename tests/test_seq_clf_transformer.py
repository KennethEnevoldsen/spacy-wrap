"""Test for the sequence classification transformer pipeline component."""
import shutil

import pytest
import spacy

import spacy_wrap  # noqa F401

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
        [("Senile gamle idiot", "offensive"), ("Jeg er glad", "not offensive")],
    ),
)
EXAMPLES.append(
    (
        {
            "assign_to_cats": False,
            "model": {
                "name": "distilbert-base-uncased-finetuned-sst-2-english",
            },
        },
        [
            ("I like you. I love you", "POSITIVE"),
            ("I hate you. I dislike you", "NEGATIVE"),
            ("", None),
        ],
    ),
)


class TestSequenceClassificationTransformer:
    @pytest.mark.parametrize("config, examples", EXAMPLES)
    def test_forward(self, config: dict, examples: list):
        """tests if that the forward pass work as intended."""

        nlp = spacy.blank("en")
        nlp.add_pipe("sequence_classification_transformer", config=config)

        for text, label in examples:
            doc = nlp(text)

            trf_data_ext = (
                config["doc_extension_trf_data"]
                if "doc_extension_trf_data" in config
                else "seq_clf_trf_data"
            )
            clf_trf_data = getattr(doc._, trf_data_ext)
            clf_pred_ext = (
                config["doc_extension_prediction"]
                if "doc_extension_prediction" in config
                else "seq_clf_prediction"
            )
            assert clf_trf_data
            prediction = getattr(doc._, clf_pred_ext)
            assert prediction == label
            prob = getattr(doc._, clf_pred_ext + "_prob")
            assert isinstance(prob, dict)

            if "assign_to_cats" not in config or config["assign_to_cats"]:
                assert max(doc.cats, key=doc.cats.get) == label
            else:
                assert doc.cats == {}

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

        transformer = nlp.get_pipe("sequence_classification_transformer")

        transformer.to_disk("spacy_wrap_serialization_test")
        transformer.from_disk("spacy_wrap_serialization_test")

        # remove test directory
        shutil.rmtree("spacy_wrap_serialization_test")
