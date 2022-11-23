"""
Copyright (C) 2022 Explosion AI and Kenneth Enevoldsen - All Rights Reserved
You may use, distribute and modify this code under the
terms of the MIT license.

Original code from:
https://github.com/explosion/spacy-transformers/blob/5a36943fccb66b5e7c7c2079b1b90ff9b2f9d020/spacy_transformers/data_classes.py


The following functions are copied/modified:
- split_by_doc. Changed to fetch logits instead of token embeddings
"""


from typing import List, Literal

import numpy as np
import torch
from spacy.tokens import Doc, Span
from spacy_transformers.align import get_token_positions
from spacy_transformers.data_classes import TransformerData
from thinc.api import torch2xp
from transformers.file_utils import ModelOutput


def split_by_doc(self) -> List[TransformerData]:
    """Split a TransformerData that represents a batch into a list with one
    TransformerData per Doc.

    Original code from:
    https://github.com/explosion/spacy-transformers/blob/5a36943fccb66b5e7c7c2079b1b90ff9b2f9d020/spacy_transformers/data_classes.py

    The following parts are modified:
    - split_by_doc. Changed to fetch logits instead of token embeddings
    """
    flat_spans = []
    for doc_spans in self.spans:
        flat_spans.extend(doc_spans)
    token_positions = get_token_positions(flat_spans)
    outputs = []
    start = 0
    prev_tokens = 0
    for doc_spans in self.spans:
        if len(doc_spans) == 0 or len(doc_spans[0]) == 0:
            outputs.append(TransformerData.empty())
            continue
        start_i = token_positions[doc_spans[0][0]]
        end_i = token_positions[doc_spans[-1][-1]] + 1
        end = start + len(doc_spans)
        doc_tokens = self.wordpieces[start:end]
        doc_align = self.align[start_i:end_i]
        doc_align.data = doc_align.data - prev_tokens
        model_output = ModelOutput()
        logits = self.model_output.logits  # changed to fetch logits
        for key, output in self.model_output.items():
            if isinstance(output, torch.Tensor):
                model_output[key] = torch2xp(output[start:end])
            elif (
                isinstance(output, tuple)
                and all(isinstance(t, torch.Tensor) for t in output)
                and all(t.shape[0] == logits.shape[0] for t in output)
            ):
                model_output[key] = [torch2xp(t[start:end]) for t in output]
        outputs.append(
            TransformerData(
                wordpieces=doc_tokens,
                model_output=model_output,
                align=doc_align,
            ),
        )
        prev_tokens += doc_tokens.input_ids.size
        start += len(doc_spans)
    return outputs


def softmax(x):
    return np.exp(x) / sum(np.exp(x))


def add_iob_tags(doc: Doc, iob: List[str]) -> Doc:
    """Add iob tags to Doc.

    Args:
        doc (Doc): A SpaCy doc
        iob (List[str]): a list of tokens on the IOB format
    Returns:
        Doc: A doc with the spans to the new IOB
    """
    ent = []
    for i, label in enumerate(iob):

        # turn IOB labels into spans
        if label == "O":
            continue
        iob_, ent_type = label.split("-")
        if (i - 1 >= 0 and iob_ == "I" and iob[i - 1] == "O") or (
            i == 0 and iob_ == "I"
        ):
            iob_ = "B"
        if iob_ == "B":
            start = i
        if i + 1 >= len(iob) or iob[i + 1].split("-")[0] != "I":
            ent.append(Span(doc, start, i + 1, label=ent_type))
    doc.set_ents(ent)
    return doc


def add_pos_tags(
    doc: Doc,
    pos: List[str],
    extension: Literal["pos", "tag"] = "tag",
) -> Doc:
    """Add pos tags to Doc.

    Args:
        doc (Doc): A SpaCy doc
        pos (List[str]): A list of pos tags
        extension (Literal["pos", "tag"], optional): The extension to add the tags to.
            Defaults to "tag". If "pos" is used note that the tags have to be in the
            UD format.

    Returns:
        Doc: A doc with new pos tags
    """
    extension = extension + "_"
    for token, tag in zip(doc, pos):
        setattr(token, extension, tag)
    return doc


def install_extensions(doc_ext_attr) -> None:
    if not Doc.has_extension(doc_ext_attr):
        Doc.set_extension(doc_ext_attr, default=None)


UPOS_TAGS = {
    # POS tags
    # Universal POS Tags
    # http://universaldependencies.org/u/pos/
    # list from:
    # https://github.com/explosion/spaCy/blob/master/spacy/glossary.py
    "ADJ": "adjective",
    "ADP": "adposition",
    "ADV": "adverb",
    "AUX": "auxiliary",
    "CONJ": "conjunction",
    "CCONJ": "coordinating conjunction",
    "DET": "determiner",
    "INTJ": "interjection",
    "NOUN": "noun",
    "NUM": "numeral",
    "PART": "particle",
    "PRON": "pronoun",
    "PROPN": "proper noun",
    "PUNCT": "punctuation",
    "SCONJ": "subordinating conjunction",
    "SYM": "symbol",
    "VERB": "verb",
    "X": "other",
    "EOL": "end of line",
    "SPACE": "space",
}
