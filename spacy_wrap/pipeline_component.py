from typing import Callable, List, Iterable, Dict, Optional
import warnings

from spacy.language import Language
from spacy.tokens import Doc
from spacy.vocab import Vocab

from spacy_transformers.data_classes import FullTransformerBatch
from spacy_transformers.pipeline_component import Transformer
from spacy_transformers.layers import TransformerListener
from spacy_transformers.annotation_setters import null_annotation_setter
from thinc.api import Config, Model

from .util import split_by_doc, softmax


DEFAULT_CONFIG_STR = """
[classification_transformer]
max_batch_items = 4096
doc_extension_trf_data = "clf_trf_data"
doc_extension_prediction = "classification"
labels = ["positive", negative"]

[classification_transformer.set_extra_annotations]
@annotation_setters = "spacy-transformers.null_annotation_setter.v1"

[classification_transformer.model]
@architectures = "spacy-wrap.ClassificationTransformerModel.v1"
name = "roberta-base"
tokenizer_config = {"use_fast": true}
transformer_config = {}
mixed_precision = false
grad_scaler_config = {}

[classification_transformer.model.get_spans]
@span_getters = "spacy-transformers.strided_spans.v1"
window = 128
stride = 96
"""

DEFAULT_CONFIG = Config().from_str(DEFAULT_CONFIG_STR)


@Language.factory(
    "classification_transformer",
    default_config=DEFAULT_CONFIG["classification_transformer"],
)
def make_classification_transformer(
    nlp: Language,
    name: str,
    model: Model[List[Doc], FullTransformerBatch],
    set_extra_annotations: Callable[[List[Doc], FullTransformerBatch], None],
    max_batch_items: int,
    doc_extension_trf_data: str,
    doc_extension_prediction: str,
    labels: List[str],
):
    """Construct a ClassificationTransformer component, which lets you plug a model from the
    Huggingface transformers library into spaCy so you can use it in your
    pipeline. One or more subsequent spaCy components can use the transformer
    outputs as features in its model, with gradients backpropagated to the single
    shared weights.

    Args:
        model (Model[List[Doc], FullTransformerBatch]): A thinc Model object wrapping
            the transformer. Usually you will want to use the ClassificationTransformer
            layer for this.
        set_extra_annotations (Callable[[List[Doc], FullTransformerBatch], None]): A
            callback to set additional information onto the batch of `Doc` objects.
            The doc._.{doc_extension_trf_data} attribute is set prior to calling the callback
            as well as doc._.{doc_extension_prediction} and doc._.{doc_extension_prediction}_prob.
            By default, no additional annotations are set.
        labels (List[str]): A list of labels which the transformer model outputs, should be ordered.
    """
    return ClassificationTransformer(
        nlp.vocab,
        model,
        set_extra_annotations,
        max_batch_items=max_batch_items,
        name=name,
        labels=labels,
        doc_extension_trf_data=doc_extension_trf_data,
        doc_extension_prediction=doc_extension_prediction,
    )


class ClassificationTransformer(Transformer):
    """spaCy pipeline component that provides access to a transformer model from
    the Huggingface transformers library. Usually you will connect subsequent
    components to the shared transformer using the TransformerListener layer.
    This works similarly to spaCy's Tok2Vec component and Tok2VecListener
    sublayer.

    The activations from the transformer are saved in the doc._.trf_data extension
    attribute. You can also provide a callback to set additional annotations.

    Args:
        vocab (Vocab): The Vocab object for the pipeline.
        model (Model[List[Doc], FullTransformerBatch]): A thinc Model object wrapping
            the transformer. Usually you will want to use the TransformerModel
            layer for this.
        set_extra_annotations (Callable[[List[Doc], FullTransformerBatch], None]): A
            callback to set additional information onto the batch of `Doc` objects.
            The doc._.{doc_extension_trf_data} attribute is set prior to calling the callback
            as well as doc._.{doc_extension_prediction} and doc._.{doc_extension_prediction}_prob.
            By default, no additional annotations are set.
        labels (List[str]): A list of labels which the transformer model outputs, should be ordered.
    """

    def __init__(
        self,
        vocab: Vocab,
        model: Model[List[Doc], FullTransformerBatch],
        set_extra_annotations: Callable = null_annotation_setter,
        *,
        name: str = "classification_transformer",
        max_batch_items: int = 128 * 32,  # Max size of padded batch
        labels: List[str],
        doc_extension_trf_data: str,
        doc_extension_prediction: str,
    ):
        """Initialize the transformer component."""
        self.name = name
        self.vocab = vocab
        self.model = model
        if not isinstance(self.model, Model):
            raise ValueError(f"Expected Thinc Model, got: {type(self.model)}")
        self.set_extra_annotations = set_extra_annotations
        self.cfg = {"max_batch_items": max_batch_items}
        self.listener_map: Dict[str, List[TransformerListener]] = {}

        self.doc_extension_trf_data = doc_extension_trf_data

        if not Doc.has_extension(doc_extension_trf_data):
            Doc.set_extension(doc_extension_trf_data, default=None)

        install_classification_extensions(
            doc_extension_prediction, labels, doc_extension_trf_data
        )

    def set_annotations(
        self, docs: Iterable[Doc], predictions: FullTransformerBatch
    ) -> None:
        """Assign the extracted features to the Doc objects. By default, the
        TransformerData object is written to the doc._.trf_data attribute. Your
        set_extra_annotations callback is then called, if provided.

        Args:
            docs (Iterable[Doc]): The documents to modify.
            predictions: (FullTransformerBatch): A batch of activations.
        """
        doc_data = split_by_doc(predictions)
        for doc, data in zip(docs, doc_data):
            setattr(doc._, self.doc_extension_trf_data, data)
        self.set_extra_annotations(docs, predictions)


def install_classification_extensions(
    doc_extension_prediction: str,
    labels: list,
    doc_extension: str,
):
    prob_getter, label_getter = make_classification_getter(
        doc_extension_prediction, labels, doc_extension
    )
    if not Doc.has_extension(f"{doc_extension_prediction}_prob"):
        Doc.set_extension(f"{doc_extension_prediction}_prob", getter=prob_getter)
    if not Doc.has_extension(doc_extension_prediction):
        Doc.set_extension(doc_extension_prediction, getter=label_getter)


def make_classification_getter(
    doc_extension_prediction, labels, doc_extension_trf_data
):
    def prob_getter(doc) -> dict:
        trf_data = getattr(doc._, doc_extension_trf_data)
        if trf_data.tensors:
            return {
                "prob": softmax(trf_data.tensors[0][0]).round(decimals=3),
                "labels": labels,
            }
        else:
            warnings.warn(
                "The tensors from the transformer forward pass is empty this is likely caused by an empty input string. Thus the model will return None"
            )
            return {
                "prob": None,
                "labels": labels,
            }

    def label_getter(doc) -> Optional[str]:
        prob = getattr(doc._, f"{doc_extension_prediction}_prob")
        if prob["prob"] is not None:
            return labels[int(prob["prob"].argmax())]
        else:
            return None

    return prob_getter, label_getter
