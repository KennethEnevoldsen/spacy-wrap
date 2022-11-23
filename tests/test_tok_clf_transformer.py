"""Test for the token classification transformer pipeline component."""
import shutil

import pytest
import spacy

import spacy_wrap  # noqa F401

EXAMPLES_NER = []
EXAMPLES_NER.append(
    (
        {
            "doc_extension_trf_data": "tok_clf_trf_data",
            "doc_extension_prediction": "token_clf_iob_tags",
            "labels": None,  # infer from model
            "model": {
                "name": "dslim/bert-base-NER",  # model from the hub
                "@architectures": "spacy-wrap.TokenClassificationTransformerModel.v1",
            },
        },
        (
            "My name is Wolfgang and I live in Berlin. I love you.",
            [("Wolfgang", (3, 4), "PER"), ("Berlin", (8, 9), "LOC")],
        ),
    ),
)
EXAMPLES_NER.append(
    (
        {
            "doc_extension_trf_data": "tok_clf_trf_data",
            "doc_extension_prediction": "token_clf_iob_tags",
            "labels": None,  # infer from model
            "model": {
                "name": "dslim/bert-base-NER",  # model from the hub
                "@architectures": "spacy-wrap.TokenClassificationTransformerModel.v1",
            },
        },
        ("", []),
    ),
)
EXAMPLES_NER.append(
    (
        {
            "model": {
                "name": "PlanTL-GOB-ES/bsc-bio-ehr-es-pharmaconer",
            },
        },
        (
            "Se realizó estudio analítico destacando incremento de niveles de PTH y vitamina D (103,7 pg/ml y 272 ng/ml, respectivamente), atribuidos al exceso de suplementación de vitamina D .",
            [
                ("PTH", (9, 10), "PROTEINAS"),
                ("vitamina D", (11, 13), "NORMALIZABLES"),
                ("vitamina D", (33, 35), "NORMALIZABLES"),
            ],
        ),
    ),
)

EXAMPLES_POS = []
EXAMPLES_POS.append(
    (
        {"model": {"name": "vblagoje/bert-english-uncased-finetuned-pos"}},
        (
            "My name is Wolfgang and I live in Berlin",
            ["PRON", "NOUN", "AUX", "PROPN", "CCONJ", "PRON", "VERB", "ADP", "PROPN"],
        ),
    ),
)


class TestTokenClassificationTransformer:
    @pytest.mark.parametrize("config, example", EXAMPLES_NER)
    def test_forward_ner(self, config: dict, example: tuple):
        """tests if that the forward pass work as intended for NER models."""

        nlp = spacy.blank("es")
        nlp.add_pipe("token_classification_transformer", config=config)

        text, expected = example
        doc = nlp(text)

        trf_data_ext = (
            config["doc_extension_trf_data"]
            if "doc_extension_trf_data" in config
            else "tok_clf_trf_data"
        )

        assert getattr(doc._, trf_data_ext)
        assert len(doc.ents) == len(expected)

        for ent, (text, (start, end), label) in zip(doc.ents, expected):
            assert ent.text == text
            assert ent.start == start
            assert ent.end == end
            assert ent.label_ == label

    @pytest.mark.parametrize("config, example", EXAMPLES_POS)
    def test_forward_pos(self, config: dict, example: tuple):
        """tests if that the forward pass work as intended for POS models."""

        nlp = spacy.blank("en")
        nlp.add_pipe("token_classification_transformer", config=config)
        text, expected = example
        doc = nlp(text)

        for token, label in zip(doc, expected):
            assert token.pos_ == label

    def test_to_and_from_disk(self):
        """tests if the pipeline can be serialized to disk."""

        nlp = spacy.blank("en")
        nlp.add_pipe("token_classification_transformer", config=EXAMPLES_NER[0][0])

        transformer = nlp.get_pipe("token_classification_transformer")

        transformer.to_disk("spacy_wrap_serialization_test")
        transformer.from_disk("spacy_wrap_serialization_test")

        # remove test directory
        shutil.rmtree("spacy_wrap_serialization_test")
