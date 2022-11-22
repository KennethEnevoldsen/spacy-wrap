"""Test for the token classification transformer pipeline component."""
import shutil

import pytest
import spacy

EXAMPLES = []
EXAMPLES.append(
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
EXAMPLES.append(
    (
        {
            "model": {
                "name": "PlanTL-GOB-ES/bsc-bio-ehr-es-pharmaconer",
            },
        },
        (
            "Se realizó estudio analítico destacando incremento de niveles de PTH y vitamina D (103,7 pg/ml y 272 ng/ml, respectivamente), atribuidos al exceso de suplementación de vitamina D.",
            [
                ("PTH", (3, 4), "PROTEINAS"),
                ("vitamina D", (5, 6), "NORMALIZABLES"),
                ("vitamina D", (7, 8), "NORMALIZABLES"),
            ],
        ),
    ),
)


class TestTokenClassificationTransformer:
    @pytest.mark.parametrize("config, example", EXAMPLES)
    def test_forward(self, config: dict, example: tuple):
        """tests if that the forward pass work as intended."""

        nlp = spacy.blank("en")
        nlp.add_pipe("token_classification_transformer", config=config)
        nlp.initialize()

        text, expected = example
        doc = nlp(text)

        assert doc._.tok_clf_trf_data
        assert len(doc.ents) == len(expected)

        for ent, (text, (start, end), label) in zip(doc.ents, expected):
            assert ent.text == text
            assert ent.start == start
            assert ent.end == end
            assert ent.label_ == label

    def test_to_and_from_disk(self):
        """tests if the pipeline can be serialized to disk."""

        nlp = spacy.blank("en")
        nlp.add_pipe("token_classification_transformer", config=EXAMPLES[0][0])
        nlp.initialize()

        transformer = nlp.get_pipe("token_classification_transformer")

        transformer.to_disk("spacy_wrap_serialization_test")
        transformer.from_disk("spacy_wrap_serialization_test")

        # remove test directory
        shutil.rmtree("spacy_wrap_serialization_test")
