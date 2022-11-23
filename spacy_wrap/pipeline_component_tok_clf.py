"""
Copyright (C) 2022 Explosion AI and Kenneth Enevoldsen - All Rights Reserved
You may use, distribute and modify this code under the
terms of the MIT license.

Original code from:
https://github.com/explosion/spacy-transformers/blob/master/spacy_transformers/pipeline_component.py

The following functions are copied/modified from their code:
- make_classification_transformer. Now created an instance of
ClassificationTransformer instead of Transformer, which require the three new arguments:
doc_extension_trf_data, doc_extension_prediction, labels
- ClassificationTransformer. A varation of the Transformer. Includes changes to the init
adding additional extensions, changed to load methods using AutoModelForClassification
instead of Automodel. Code related to listeners has also been removed to avoid potential
collision with the existing transformer model. There has also been a rework of the
get_loss and and update functions.
- install_extensions. Added argument, which is no longer predefined.
- install_extensions. Added argument.
"""

from pathlib import Path
from typing import Callable, Iterable, Iterator, List, Literal, Optional, Tuple, Union

import numpy as np
import srsly
from spacy import Errors, util
from spacy.language import Language
from spacy.pipeline.pipe import deserialize_config
from spacy.pipeline.trainable_pipe import TrainablePipe
from spacy.tokens import Doc
from spacy.training import Example
from spacy.util import minibatch
from spacy.vocab import Vocab  # pylint: disable=no-name-in-module
from spacy_transformers.annotation_setters import null_annotation_setter
from spacy_transformers.data_classes import FullTransformerBatch, TransformerData
from spacy_transformers.util import batch_by_length
from thinc.api import Config, Model

from .util import UPOS_TAGS, add_iob_tags, add_pos_tags, softmax, split_by_doc

DEFAULT_CONFIG_STR = """
[token_classification_transformer]
max_batch_items = 4096
doc_extension_trf_data = "tok_clf_trf_data"
doc_extension_prediction = "tok_clf_predictions"
predictions_to = null
labels = null
aggregation_strategy = "average"

[token_classification_transformer.set_extra_annotations]
@annotation_setters = "spacy-transformers.null_annotation_setter.v1"

[token_classification_transformer.model]
@architectures = "spacy-wrap.TokenClassificationTransformerModel.v1"
tokenizer_config = {"use_fast": true}
transformer_config = {}
mixed_precision = false
grad_scaler_config = {}

[token_classification_transformer.model.get_spans]
@span_getters = "spacy-transformers.strided_spans.v1"
window = 128
stride = 96
"""

DEFAULT_CONFIG = Config().from_str(DEFAULT_CONFIG_STR)


@Language.factory(
    "token_classification_transformer",
    default_config=DEFAULT_CONFIG["token_classification_transformer"],
)
def make_token_classification_transformer(
    nlp: Language,
    name: str,
    model: Model[List[Doc], FullTransformerBatch],
    set_extra_annotations: Callable[[List[Doc], FullTransformerBatch], None],
    max_batch_items: int,
    doc_extension_trf_data: str,
    doc_extension_prediction: str,
    aggregation_strategy: Literal["first", "average", "max"],
    labels: Optional[List[str]] = None,
    predictions_to: Optional[List[Literal["pos", "tag", "ents"]]] = None,
) -> "TokenClassificationTransformer":
    """Construct a ClassificationTransformer component, which lets you plug a
    model from the Huggingface transformers library into spaCy so you can use
    it in your pipeline. One or more subsequent spaCy components can use the
    transformer outputs as features in its model, with gradients backpropagated
    to the single shared weights.

    Args:
        nlp (Language): The current nlp object.
        name (str): The name of the component instance.
        model (Model[List[Doc], FullTransformerBatch]): A thinc Model object wrapping
            the transformer. Usually you will want to use the ClassificationTransformer
            layer for this.
        set_extra_annotations (Callable[[List[Doc], FullTransformerBatch], None]): A
            callback to set additional information onto the batch of `Doc` objects.
            The doc._.{doc_extension_trf_data} attribute is set prior to calling the callback
            as well as doc._.{doc_extension_prediction} and doc._.{doc_extension_prediction}_prob.
            By default, no additional annotations are set.
        max_batch_items (int): The maximum number of items to process in a batch.
        doc_extension_trf_data (str): The name of the doc extension to store the
            transformer data in.
        doc_extension_prediction (str): The name of the doc extension to store the
            predictions in.
        aggregation_strategy (Literal["first", "average", "max"]): The aggregation
            strategy to use. Chosen to correspond to the aggregation strategies
            used in the `TokenClassificationPipeline` in Huggingface:
            https://huggingface.co/docs/transformers/v4.24.0/en/main_classes/pipelines#transformers.TokenClassificationPipeline.aggregation_strategy
            “first”: Words will simply use the tag of the first token of the word
            when there is ambiguity. “average”: Scores will be averaged first across
            tokens, and then the maximum label is applied. “max”: Word entity will
            simply be the token with the maximum score.
        labels (List[str]): A list of labels which the transformer model outputs, should
            be ordered.
        predictions_to (Optional[List[Literal["pos", "tag", "ents"]]]): A list of
            attributes the predictions should be written to. Default to None. In which
            case it is inferred from the labels. If the labels are UPOS tags, the
            predictions will be written to the "pos" attribute. If the labels are
            IOB tags, the predictions will be written to the "ents" attribute. "tag" is
            not inferred from the labels, but can be added explicitly.
            Note that if the "pos" attribute is set the labels must be UPOS tags and if
            the "ents" attribute is set the labels must be IOB tags.

    Returns:
        TokenClassificationTransformer: The constructed component.

    Example:
        >>> import spacy
        >>> import spacy_wrap
        >>>
        >>> nlp = spacy.blank("en")
        >>> nlp.add_pipe("token_classification_transformer", config={
        ...     "model": {"name": "vblagoje/bert-english-uncased-finetuned-pos"}}
        ... )
        >>> doc = nlp("My name is Wolfgang and I live in Berlin")
        >>> print([tok.pos_ for tok in doc])
        ["PRON", "NOUN", "AUX", "PROPN", "CCONJ", "PRON", "VERB", "ADP", "PROPN"]
    """
    clf_trf = TokenClassificationTransformer(
        vocab=nlp.vocab,
        model=model,
        set_extra_annotations=set_extra_annotations,
        max_batch_items=max_batch_items,
        aggregation_strategy=aggregation_strategy,
        name=name,
        labels=labels,
        doc_extension_trf_data=doc_extension_trf_data,
        doc_extension_prediction=doc_extension_prediction,
        predictions_to=predictions_to,
    )
    return clf_trf


class TokenClassificationTransformer(TrainablePipe):
    """spaCy pipeline component that provides access to a transformer model
    from the Huggingface transformers library. Usually you will connect
    subsequent components to the shared transformer using the
    TransformerListener layer. This works similarly to spaCy's Tok2Vec
    component and Tok2VecListener sublayer. The activations from the
    transformer are saved in the doc._.trf_data extension attribute. You can
    also provide a callback to set additional annotations.

    Args:
        vocab (Vocab): The Vocab object for the pipeline.
        model (Model[List[Doc], FullTransformerBatch]): A thinc Model object wrapping
            the transformer. Usually you will want to use the TransformerModel
            layer for this.
        set_extra_annotations (Callable[[List[Doc], FullTransformerBatch], None]): A
            callback to set additional information onto the batch of `Doc` objects.
            The doc._.{doc_extension_trf_data} attribute is set prior to calling the
            callback as well as doc._.{doc_extension_prediction} and
            doc._.{doc_extension_prediction}_prob. By default, no additional annotations
            are set.
        labels (List[str]): A list of labels which the transformer model outputs, should
            be ordered.
    """

    def __init__(
        self,
        vocab: Vocab,
        model: Model[List[Doc], FullTransformerBatch],
        labels: List[str],
        doc_extension_trf_data: str,
        doc_extension_prediction: str,
        set_extra_annotations: Callable = null_annotation_setter,
        *,
        name: str = "token_classification_transformer",
        max_batch_items: int = 128 * 32,  # Max size of padded batch
        predictions_to: Optional[List[Literal["pos", "tag", "ents"]]] = None,
        aggregation_strategy: Literal["first", "average", "max"],
    ):
        """Initialize the transformer component."""
        self.name = name
        self.vocab = vocab
        self.model = model
        if not isinstance(self.model, Model):
            raise ValueError(f"Expected Thinc Model, got: {type(self.model)}")
        self.set_extra_annotations = set_extra_annotations
        self.cfg = {"max_batch_items": max_batch_items}
        self.doc_extension_trf_data = doc_extension_trf_data
        self.doc_extension_prediction = doc_extension_prediction
        self.model_labels = labels
        self.predictions_to = predictions_to
        self.aggregation_strategy = aggregation_strategy
        self.is_initialized = False

        install_extensions(self.doc_extension_trf_data)
        install_extensions(self.doc_extension_prediction)
        install_extensions(self.doc_extension_prediction + "_prob")

        # is there any argument for not doing this here?
        if not self.is_initialized:
            self.__initialize_component()

    @property
    def is_trainable(self) -> bool:
        return False

    def set_annotations(
        self,
        docs: Iterable[Doc],
        predictions: FullTransformerBatch,
    ) -> None:
        """Assign the extracted features to the Doc objects. By default, the
        TransformerData object is written to the doc._.{doc_extension_trf_data}
        attribute. Your set_extra_annotations callback is then called, if
        provided.

        Args:
            docs (Iterable[Doc]): The documents to modify.
            predictions: (FullTransformerBatch): A batch of activations.
        """
        doc_data = split_by_doc(predictions)
        for doc, data in zip(docs, doc_data):
            # check if doc is data is empty
            setattr(doc._, self.doc_extension_trf_data, data)

            iob_tags, iob_prob = self.convert_to_token_predictions(
                data,
                self.aggregation_strategy,
                self.model_labels,
            )
            setattr(doc._, self.doc_extension_prediction, iob_tags)
            setattr(doc._, f"{self.doc_extension_prediction}_prob", iob_prob)
            if "ents" in self.predictions_to:
                doc = add_iob_tags(doc, iob_tags)
            if "pos" in self.predictions_to:
                doc = add_pos_tags(doc, iob_tags, extension="pos")
            if "tag" in self.predictions_to:
                doc = add_pos_tags(doc, iob_tags, extension="tag")

        self.set_extra_annotations(docs, predictions)

    @staticmethod
    def convert_to_token_predictions(
        data: TransformerData,
        aggregation_strategy: Literal["first", "average", "max"],
        labels: List[str],
    ) -> Tuple[List[str], List[dict]]:
        """Convert the transformer data to token predictions.

        Args:
            data (TransformerData): The transformer data.
            aggregation_strategy (Literal["first", "average", "max"]): The aggregation
                strategy to use. Chosen to correspond to the aggregation strategies
                used in the `TokenClassificationPipeline` in Huggingface:
                https://huggingface.co/docs/transformers/v4.24.0/en/main_classes/pipelines#transformers.TokenClassificationPipeline.aggregation_strategy
                “first”: Words will simply use the tag of the first token of the word
                when there is ambiguity. “average”: Scores will be averaged first across
                tokens, and then the maximum label is applied. “max”: Word entity will
                simply be the token with the maximum score.

        Returns:
            Tuple[List[str], List[dict]]: The token tags and a dict containing the
                probabilites and corresponding labels.
        """
        if not data.model_output:  # if data is empty e.g. due to empty string
            return [], []

        if aggregation_strategy == "first":

            def agg(x):
                return x[0]

        elif aggregation_strategy == "average":

            def agg(x):
                return x.mean(axis=0)

        elif aggregation_strategy == "max":

            def agg(x):
                return x.max(axis=0)

        else:
            raise ValueError(
                f"Aggregation strategy {aggregation_strategy} is not supported.",
            )

        token_probabilities = []
        token_tags = []
        logits = data.model_output.logits[0]
        for align in data.align:
            # aggregate the logits for each token
            agg_token_logits = agg(logits[align.data[:, 0]])
            token_probabilities_ = {
                "prob": softmax(agg_token_logits).round(decimals=3),
                "label": labels,
            }
            token_probabilities.append(token_probabilities_)
            token_tags.append(labels[np.argmax(agg_token_logits)])

        return token_tags, token_probabilities

    def __call__(self, doc: Doc) -> Doc:
        """Apply the pipe to one document. The document is modified in place,
        and returned. This usually happens under the hood when the nlp object
        is called on a text and all components are applied to the Doc.

        Args:
            docs (Doc): The Doc to process.

        Returns:
            (Doc): The processed Doc.
        """
        install_extensions(self.doc_extension_trf_data)
        outputs = self.predict([doc])
        self.set_annotations([doc], outputs)
        return doc

    def pipe(self, stream: Iterable[Doc], *, batch_size: int = 128) -> Iterator[Doc]:
        """Apply the pipe to a stream of documents. This usually happens under
        the hood when the nlp object is called on a text and all components are
        applied to the Doc.

        Args:
            stream (Iterable[Doc]): A stream of documents.
            batch_size (int): The number of documents to buffer.

        Yield:
            (Doc): Processed documents in order.
        """
        install_extensions(self.doc_extension_trf_data)
        for outer_batch in minibatch(stream, batch_size):
            outer_batch = list(outer_batch)
            for indices in batch_by_length(outer_batch, self.cfg["max_batch_items"]):
                subbatch = [outer_batch[i] for i in indices]
                self.set_annotations(subbatch, self.predict(subbatch))
            yield from outer_batch

    def predict(self, docs: Iterable[Doc]) -> FullTransformerBatch:
        """Apply the pipeline's model to a batch of docs, without modifying
        them. Returns the extracted features as the FullTransformerBatch
        dataclass.

        Args:
            docs (Iterable[Doc]): The documents to predict.
        Returns:
            (FullTransformerBatch): The extracted features.
        """
        if not any(len(doc) for doc in docs):
            # Handle cases where there are no tokens in any docs.
            activations = FullTransformerBatch.empty(len(docs))
        else:
            activations = self.model.predict(docs)
        return activations

    def __initialize_component(self):
        """Initialize the component. This avoid the need to call
        nlp.initialize().

        For related issue see:
        https://github.com/explosion/spaCy/issues/7027
        """
        if self.is_initialized:
            return
        self.model.initialize()

        # extract the labels from the model config
        hf_model = self.model.layers[0].shims[0]._model
        if self.model_labels is None:
            # extract hf_model.config.label2id.items()
            # convert to sorted list
            self.model_labels = [
                tag[0]
                for tag in sorted(hf_model.config.label2id.items(), key=lambda x: x[1])
            ]

        # infer the predictions_to attribute
        if self.predictions_to is None:
            self.predictions_to = []
            # check if labels are IOB tags
            if all(is_iob_tag(label) for label in self.model_labels):
                self.predictions_to.append("ents")
            # check if labels are POS tags
            if all(label in UPOS_TAGS for label in self.model_labels):
                self.predictions_to.append("pos")

        self.is_initialized = True

    def initialize(
        self,
        get_examples: Callable[[], Iterable[Example]],
        *,
        nlp: Optional[Language] = None,
    ):
        """Initialize the pipe for training, using data examples if available.

        Args:
            nlp (Language): The current nlp object.
        """
        self.__initialize_component()

    def to_disk(
        self, path: Union[str, Path], *, exclude: Iterable[str] = tuple()
    ) -> None:
        """Serialize the pipe to disk.

        Args:
            path (str / Path): Path to a directory.
            exclude (Iterable[str]): String names of serialization fields to exclude.
        """
        serialize = {}
        serialize["cfg"] = lambda p: srsly.write_json(p, self.cfg)
        serialize["vocab"] = self.vocab.to_disk
        serialize["model"] = self.model.to_disk
        util.to_disk(path, serialize, exclude)

    def from_disk(
        self, path: Union[str, Path], *, exclude: Iterable[str] = tuple()
    ) -> "TokenClassificationTransformer":
        """Load the pipe from disk.

        Args:
            path (str / Path): Path to a directory.
            exclude (Iterable[str]): String names of serialization fields to exclude.

        Returns:
            (Transformer): The loaded object.
        """

        def load_model(p):
            try:
                with open(p, "rb") as mfile:
                    self.model.from_bytes(mfile.read())
            except AttributeError:
                raise ValueError(Errors.E149) from None

        deserialize = {
            "vocab": self.vocab.from_disk,
            "cfg": lambda p: self.cfg.update(deserialize_config(p)),
            "model": load_model,
        }
        util.from_disk(path, deserialize, exclude)
        return self


def install_extensions(doc_ext_attr) -> None:
    if not Doc.has_extension(doc_ext_attr):
        Doc.set_extension(doc_ext_attr, default=None)


def is_iob_tag(label: str) -> bool:
    """Check if a label is an IOB tag.

    Args:
        label (str): The label to check.

    Returns:
        (bool): True if the label is an IOB tag.
    """
    label_ = label.lower()
    return label_.startswith("i-") or label_.startswith("b-") or label_ == "o"
