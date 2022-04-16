"""
Copyright (C) 2022 Explosion AI and K. Enevoldsen - All Rights Reserved
You may use, distribute and modify this code under the
terms of the MIT license.

Original code from:
https://github.com/explosion/spacy-transformers/blob/master/spacy_transformers/layers/hf_wrapper.py

The following functions are copied/modified:
- HFWrapper.v1 renamed to spacy_wrap_HFWrapper.v1. Changed to use a generalized HFShim
to be able to forward the three new arguments:
    load_config_fn
    load_tokenizer_fn
    load_model_from_config_fn

"""

from typing import Callable, Optional, Any

from thinc.layers.pytorchwrapper import forward as pt_forward
from thinc.layers.pytorchwrapper import (
    convert_pytorch_default_inputs,
    convert_pytorch_default_outputs,
)
from thinc.api import registry, Model

from spacy_transformers.data_classes import HFObjects

from transformers import AutoModel, AutoConfig, AutoTokenizer

from .hf_shim import HFShim


@registry.layers("spacy_wrap_HFWrapper.v1")
def HFWrapper(
    hf_model: "HFObjects",
    convert_inputs: Optional[Callable] = None,
    convert_outputs: Optional[Callable] = None,
    mixed_precision: bool = False,
    grad_scaler_config: dict = {},
    load_config_fn: Callable = AutoConfig.from_pretrained,
    load_tokenizer_fn: Callable = AutoTokenizer.from_pretrained,
    load_model_from_config_fn: Callable = AutoModel.from_config,
) -> Model[Any, Any]:
    """Wrap a PyTorch HF model, so that it has the same API as Thinc models.
    To optimize the model, you'll need to create a PyTorch optimizer and call
    optimizer.step() after each batch. See examples/wrap_pytorch.py

    Your PyTorch model's forward method can take arbitrary args and kwargs,
    but must return either a single tensor as output or a tuple. You may find the
    PyTorch register_forward_hook helpful if you need to adapt the output.

    The convert functions are used to map inputs and outputs to and from your
    PyTorch model. Each function should return the converted output, and a callback
    to use during the backward pass. So:

        Xtorch, get_dX = convert_inputs(X)
        Ytorch, torch_backprop = model.shims[0](Xtorch, is_train)
        Y, get_dYtorch = convert_outputs(Ytorch)

    To allow maximum flexibility, the PyTorchShim expects ArgsKwargs objects
    on the way into the forward and backward passed. The ArgsKwargs objects
    will be passed straight into the model in the forward pass, and straight
    into `torch.autograd.backward` during the backward pass.
    """
    if convert_inputs is None:
        convert_inputs = convert_pytorch_default_inputs
    if convert_outputs is None:
        convert_outputs = convert_pytorch_default_outputs

    return Model(
        "hf-pytorch",
        pt_forward,
        attrs={"convert_inputs": convert_inputs, "convert_outputs": convert_outputs},
        shims=[
            HFShim(
                hf_model,
                mixed_precision=mixed_precision,
                grad_scaler_config=grad_scaler_config,
                load_config_fn=load_config_fn,
                load_tokenizer_fn=load_tokenizer_fn,
                load_model_from_config_fn=load_model_from_config_fn,
            )
        ],
        dims={"nI": None, "nO": None},
    )
