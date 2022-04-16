import pytest
import shutil

import spacy

import spacy_wrap


class TestClassificationTransformer:
    @pytest.fixture(scope="class")
    def nlp(self):
        nlp = spacy.blank("da")
        config = {
            "doc_extension_trf_data": "clf_trf_data",
            "doc_extension_prediction": "hate_speech",
            "labels": ["Not hate Speech", "Hate speech"],
            "model": {
                "name": "DaNLP/da-bert-hatespeech-detection",
            },
        }
        nlp.add_pipe("classification_transformer", config=config)
        return nlp

    def test_forward(self, nlp):
        """tests if that the forward pass work as intended"""

        doc = nlp("Senile gamle idiot")

        doc._.clf_trf_data
        assert doc._.hate_speech == "Hate speech"
        assert isinstance(doc._.hate_speech_prob, dict)

    def test_to_and_from_disk(self, nlp):
        """tests if the pipeline can be serialized to disk"""

        transformer = nlp.get_pipe("classification_transformer")

        transformer.to_disk("spacy_wrap_serialization_test")
        transformer.from_disk("spacy_wrap_serialization_test")

        # remove test directory
        shutil.rmtree("spacy_wrap_serialization_test")
