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
        {"model": {"name": "dslim/bert-base-NER"}},
        (
            "My name is Wolfgang 🚀🚀 🚀 and I live in Berlin.",
            [("Wolfgang 🚀🚀 🚀", (3, 7), "PER"), ("Berlin", (11, 12), "LOC")],
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
            "Se realizó estudio analítico destacando incremento de niveles de PTH y vitamina D (103,7 pg/ml y 272 ng/ml, respectivamente), atribuidos al exceso de suplementación de vitamina D .",  # noqa: E501
            [
                ("PTH", (9, 10), "PROTEINAS"),
                ("vitamina D", (11, 13), "NORMALIZABLES"),
                ("vitamina D", (33, 35), "NORMALIZABLES"),
            ],
        ),
    ),
)

# Added to test samples longer than window size
EXAMPLES_NER.append(
    (
        {
            "model": {"name": "saattrupdan/nbailab-base-ner-scandi"},
            "aggregation_strategy": "first",
        },
        (
            """Se mor, den ligner en hund." Han skulle blive 72 år, inden han fik sit gennembrud hos det brede publikum med albummet "The Healer". Hans navn er John Lee Hooker, født 22. august 1917 i bluesmusikkens hjemstat Mississippi. Og nok fik han først det afgørende gennembrud i 1989, hvor han fik en Grammy for "The Healer", men elskere af eksempelvis Johnny Winter, Status Quo, Animals, Georgia Satellites, ZZ Top, The Black Crowes og Rolling Stones'guitarist Keith Richards kan roligt lige nu kaste sig i støvet og takke inderligt og intenst for, at John Lee Hooker kom til verden dengang i 1917. John Lee Hookers indflydelse på andre musikere som sanger og især som guitarist har været intet mindre end monumental. John Lee Hookers indflydelse på andre musikere som sanger og især som guitarist har været intet mindre end monumental. John Lee Hookers indflydelse på andre musikere som sanger og især som guitarist har været intet mindre end monumental. John Lee Hookers indflydelse på andre musikere som sanger og især som guitarist har været intet mindre end monumental. John Lee Hookers indflydelse på andre musikere som sanger og især som guitarist har været intet mindre end monumental. """,  # noqa: E501
            [
                ("The Healer", (27, 29), "MISC"),
                ("John Lee Hooker", (34, 37), "PER"),
                ("Mississippi", (46, 47), "LOC"),
                ("Grammy", (63, 64), "MISC"),
                ("The Healer", (66, 68), "MISC"),
                ("Johnny Winter", (74, 76), "PER"),
                ("Status Quo", (77, 79), "ORG"),
                ("Animals", (80, 81), "ORG"),
                ("Georgia Satellites", (82, 84), "ORG"),
                ("ZZ Top", (85, 87), "ORG"),
                ("The Black Crowes", (88, 91), "ORG"),
                ("Rolling Stones'guitarist", (92, 94), "ORG"),
                ("Keith Richards", (94, 96), "PER"),
                ("John Lee Hooker", (112, 115), "PER"),
                ("John Lee Hookers", (122, 125), "PER"),
                ("John Lee Hookers", (142, 145), "PER"),
                ("John Lee Hookers", (162, 165), "PER"),
                ("John Lee Hookers", (182, 185), "PER"),
                ("John Lee Hookers", (202, 205), "PER"),
            ],
        ),
    ),
)

# previously caused an indexError due to the leading space.
EXAMPLES_NER.append(
    (
        {"model": {"name": "saattrupdan/nbailab-base-ner-scandi"}},
        (
            " Hans navn er John Lee Hooker, født 22",
            [("John Lee Hooker", (4, 7), "PER")],
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
