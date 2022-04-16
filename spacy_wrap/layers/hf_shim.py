"""
Copyright (C) 2022 Explosion AI and K. Enevoldsen - All Rights Reserved
You may use, distribute and modify this code under the
terms of the MIT license.

Original code from:
https://github.com/explosion/spacy-transformers/blob/master/spacy_transformers/layers/hf_shim.py

The following functions are copied/modified:
- HFWrapper renamed to HFClassificationWrapper. Changed to use a generalized HFShim
"""
from typing import Any, Callable

from io import BytesIO
from pathlib import Path
import srsly
import torch
from thinc.api import get_current_ops
from spacy.util import SimpleFrozenDict

from spacy_transformers.data_classes import HFObjects
from spacy_transformers.util import make_tempdir

from thinc.api import PyTorchGradScaler, PyTorchShim

from transformers import AutoModel, AutoConfig, AutoTokenizer


class HFShim(PyTorchShim):
    """Interface between a HF Pytorch model and a Thinc Model."""

    def __init__(
        self,
        model: HFObjects,
        config=None,
        optimizer: Any = None,
        mixed_precision: bool = False,
        grad_scaler_config: dict = {},
        load_config_fn: Callable = AutoConfig.from_pretrained,
        load_tokenizer_fn: Callable = AutoTokenizer.from_pretrained,
        load_model_from_config_fn: Callable = AutoModel.from_config,
    ):
        self._hfmodel = model
        self.load_config_fn = load_config_fn
        self.load_tokenizer_fn = load_tokenizer_fn
        self.load_model_from_config_fn = load_model_from_config_fn

        # Enable gradient scaling when mixed precision is enabled and gradient
        # scaling is not explicitly disabled in the configuration.
        if "enabled" not in grad_scaler_config:
            grad_scaler_config["enabled"] = mixed_precision

        super().__init__(
            model.transformer,
            config,
            optimizer,
            mixed_precision,
            grad_scaler=PyTorchGradScaler(**grad_scaler_config),
        )

    def to_bytes(self):
        config = {}
        tok_dict = {}
        weights_bytes = {}
        tok_cfg = {}
        trf_cfg = {}
        hf_model = self._hfmodel
        if hf_model.transformer is not None:
            tok_dict = {}
            config = hf_model.transformer.config.to_dict()
            tokenizer = hf_model.tokenizer
            with make_tempdir() as temp_dir:
                if hasattr(tokenizer, "vocab_file"):
                    vocab_file_name = tokenizer.vocab_files_names["vocab_file"]
                    vocab_file_path = str((temp_dir / vocab_file_name).absolute())
                    with open(vocab_file_path, "wb") as fileh:
                        fileh.write(hf_model.vocab_file_contents)
                    tokenizer.vocab_file = vocab_file_path
                tokenizer.save_pretrained(str(temp_dir.absolute()))
                for x in temp_dir.glob("**/*"):
                    if x.is_file():
                        tok_dict[x.name] = x.read_bytes()
            filelike = BytesIO()
            torch.save(self._model.state_dict(), filelike)
            filelike.seek(0)
            weights_bytes = filelike.getvalue()
        else:
            tok_cfg = hf_model._init_tokenizer_config
            trf_cfg = hf_model._init_transformer_config
        msg = {
            "config": config,
            "state": weights_bytes,
            "tokenizer": tok_dict,
            "_init_tokenizer_config": tok_cfg,
            "_init_transformer_config": trf_cfg,
        }
        return srsly.msgpack_dumps(msg)

    def from_bytes(self, bytes_data):
        msg = srsly.msgpack_loads(bytes_data)
        config_dict = msg["config"]
        tok_dict = msg["tokenizer"]
        if config_dict:
            with make_tempdir() as temp_dir:
                config_file = temp_dir / "config.json"
                srsly.write_json(config_file, config_dict)
                config = self.load_config_fn(config_file)
                for x, x_bytes in tok_dict.items():
                    Path(temp_dir / x).write_bytes(x_bytes)
                tokenizer = self.load_tokenizer_fn(str(temp_dir.absolute()))
                vocab_file_contents = None
                if hasattr(tokenizer, "vocab_file"):
                    vocab_file_name = tokenizer.vocab_files_names["vocab_file"]
                    vocab_file_path = str((temp_dir / vocab_file_name).absolute())
                    with open(vocab_file_path, "rb") as fileh:
                        vocab_file_contents = fileh.read()

            transformer = self.load_model_from_config_fn(config)
            self._hfmodel = HFObjects(
                tokenizer,
                transformer,
                vocab_file_contents,
                SimpleFrozenDict(),
                SimpleFrozenDict(),
            )
            self._model = transformer
            filelike = BytesIO(msg["state"])
            filelike.seek(0)
            ops = get_current_ops()
            if ops.device_type == "cpu":
                map_location = "cpu"
            else:  # pragma: no cover
                device_id = torch.cuda.current_device()
                map_location = f"cuda:{device_id}"
            self._model.load_state_dict(torch.load(filelike, map_location=map_location))
            self._model.to(map_location)
        else:
            self._hfmodel = HFObjects(
                None,
                None,
                None,
                msg["_init_tokenizer_config"],
                msg["_init_transformer_config"],
            )
        return self
