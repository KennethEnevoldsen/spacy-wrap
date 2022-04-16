"""
Copyright (C) 2022 Explosion AI - All Rights Reserved
You may use, distribute and modify this code under the
terms of the MIT license.

Original code from:
https://github.com/explosion/spacy-transformers/blob/master/spacy_transformers/layers/transformer_model.py

The following functions are copied/modified:
- create_ClassificationTransformerModel_v1. Changed to call
ClassificationTransformerModel instead of TransformerModel

TODO:

- [ ] Check if last part of init() is relevant? 
    - Will have to run the code
- [x] check replace_listener and replace_listener_cfg - are they relevant?
    - I don't think so
- [x] Do i need to wrap the Transformer model?
    - you don't
- [ ] is has transformer ever set to true?
    - check after init
"""

import copy
from pathlib import Path
from typing import Callable, Dict, Union

from spacy_transformers.align import get_alignment
from spacy_transformers.data_classes import HFObjects, WordpieceBatch
from spacy_transformers.layers.transformer_model import (
    _convert_transformer_inputs,
    _convert_transformer_outputs,
    forward,
    huggingface_tokenize,
    set_pytorch_transformer,
)
from spacy_transformers.truncate import truncate_oversize_splits

from thinc.api import CupyOps, Model, get_current_ops

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from spacy_wrap.layers.hf_classification_wrapper import HFWrapper


class ClassificationTransformerModel(Model):
    """
    This is a variation of the TransformerModel from spacy-transformers with some utility regarding listeners removed
    """

    def __init__(
        self,
        name: str,
        get_spans: Callable,
        tokenizer_config: dict = {},
        transformer_config: dict = {},
        mixed_precision: bool = False,
        grad_scaler_config: dict = {},
    ):
        """
        get_spans (Callable[[List[Doc]], List[Span]]):
            A function to extract spans from the batch of Doc objects.
            This is used to manage long documents, by cutting them into smaller
            sequences before running the transformer. The spans are allowed to
            overlap, and you can also omit sections of the Doc if they are not
            relevant.
        tokenizer_config (dict): Settings to pass to the transformers tokenizer.
        transformer_config (dict): Settings to pass to the transformers forward pass.
        """
        hf_model = HFObjects(None, None, None, tokenizer_config, transformer_config)
        wrapper = HFWrapper(
            hf_model,
            convert_inputs=_convert_transformer_inputs,
            convert_outputs=_convert_transformer_outputs,
            mixed_precision=mixed_precision,
            grad_scaler_config=grad_scaler_config,
            load_model_from_config_fn=AutoModelForSequenceClassification.from_config,
        )
        super().__init__(
            "clf_transformer",
            forward,
            init=init,
            layers=[wrapper],
            dims={"nO": None},
            attrs={
                "get_spans": get_spans,
                "name": name,
                "set_transformer": set_pytorch_transformer,
                "has_transformer": False,
                "flush_cache_chance": 0.0,
            },
        )

    @property
    def tokenizer(self):
        return self.layers[0].shims[0]._hfmodel.tokenizer

    @property
    def transformer(self):
        return self.layers[0].shims[0]._hfmodel.transformer

    @property
    def _init_tokenizer_config(self):
        return self.layers[0].shims[0]._hfmodel._init_tokenizer_config

    @property
    def _init_transformer_config(self):
        return self.layers[0].shims[0]._hfmodel._init_transformer_config

    def copy(self):
        """
        Create a copy of the model, its attributes, and its parameters. Any child
        layers will also be deep-copied. The copy will receive a distinct `model.id`
        value.
        """
        copied = ClassificationTransformerModel(self.name, self.attrs["get_spans"])
        params = {}
        for name in self.param_names:
            params[name] = self.get_param(name) if self.has_param(name) else None
        copied.params = copy.deepcopy(params)
        copied.dims = copy.deepcopy(self._dims)
        copied.layers[0] = copy.deepcopy(self.layers[0])
        for name in self.grad_names:
            copied.set_grad(name, self.get_grad(name).copy())
        return copied


def init(model: Model, X=None, Y=None):
    if model.attrs["has_transformer"]:
        return
    name = model.attrs["name"]
    tok_cfg = model._init_tokenizer_config
    trf_cfg = model._init_transformer_config
    hf_model = huggingface_from_pretrained(name, tok_cfg, trf_cfg)
    model.attrs["set_transformer"](model, hf_model)
    tokenizer = model.tokenizer
    # Call the model with a batch of inputs to infer the width
    if X:
        # If we're dealing with actual texts, do the work to setup the wordpieces
        # batch properly
        docs = X
        get_spans = model.attrs["get_spans"]
        nested_spans = get_spans(docs)
        flat_spans = []
        for doc_spans in nested_spans:
            flat_spans.extend(doc_spans)
        token_data = huggingface_tokenize(tokenizer, [span.text for span in flat_spans])
        wordpieces = WordpieceBatch.from_batch_encoding(token_data)
        align = get_alignment(
            flat_spans, wordpieces.strings, tokenizer.all_special_tokens
        )
        wordpieces, align = truncate_oversize_splits(
            wordpieces, align, tokenizer.model_max_length
        )
    else:
        texts = ["hello world", "foo bar"]
        token_data = huggingface_tokenize(tokenizer, texts)
        wordpieces = WordpieceBatch.from_batch_encoding(token_data)
    model.layers[0].initialize(X=wordpieces)
    model_output = model.layers[0].predict(wordpieces)


def huggingface_from_pretrained(
    source: Union[Path, str], tok_config: Dict, trf_config: Dict
) -> HFObjects:
    """
    Create a Huggingface transformer model from pretrained/finetuned weights. Will
    download the model if it is not already downloaded.

    Note this is the same af in spacy-transformers with only AutoModel replaced with
    AutoModelForSequenceClassification

    Args:
        source (Union[str, Path]): The name of the model or a path to it, such as
            'bert-base-cased'.
        tok_config (dict): Settings to pass to the tokenizer.
        trf_config (dict): Settings to pass to the transformer.
    """
    if hasattr(source, "absolute"):
        str_path = str(source.absolute())
    else:
        str_path = source
    tokenizer = AutoTokenizer.from_pretrained(str_path, **tok_config)
    vocab_file_contents = None
    if hasattr(tokenizer, "vocab_file"):
        with open(tokenizer.vocab_file, "rb") as fileh:
            vocab_file_contents = fileh.read()
    trf_config["return_dict"] = True
    config = AutoConfig.from_pretrained(str_path, **trf_config)
    transformer = AutoModelForSequenceClassification.from_pretrained(
        str_path, config=config
    )
    ops = get_current_ops()
    if isinstance(ops, CupyOps):
        transformer.cuda()
    return HFObjects(tokenizer, transformer, vocab_file_contents)
