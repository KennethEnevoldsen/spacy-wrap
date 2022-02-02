import spacy
from spacy_wrap import ClassificationTransformer
from thinc.api import Config
from spacy.language import Language

from typing import Callable, List

from spacy.language import Language
from spacy.tokens import Doc

from spacy_transformers.data_classes import FullTransformerBatch
from thinc.api import Config, Model

DEFAULT_CONFIG_STR = """
[subjectivity]
max_batch_items = 4096
doc_extension_trf_data = "subjectivity_trf_data"
doc_extension_prediction = "subjectivity"
labels = ["objective", "subjective"]

[subjectivity.set_extra_annotations]
@annotation_setters = "spacy-transformers.null_annotation_setter.v1"

[subjectivity.model]
@architectures = "spacy-wrap.ClassificationTransformerModel.v1"
name = "DaNLP/da-bert-tone-subjective-objective"
tokenizer_config = {"use_fast": true}
transformer_config = {}
mixed_precision = false
grad_scaler_config = {}

[subjectivity.model.get_spans]
@span_getters = "spacy-transformers.strided_spans.v1"
window = 128
stride = 96
"""


DEFAULT_CONFIG = Config().from_str(DEFAULT_CONFIG_STR)

from spacy_wrap import make_classification_transformer

Language.factory(
    "subjectivity",
    default_config=DEFAULT_CONFIG["subjectivity"],
)(make_classification_transformer)

# OR

# def make_berttone_subjectivity(
#     nlp: Language,
#     name: str,
#     model: Model[List[Doc], FullTransformerBatch],
#     set_extra_annotations: Callable[[List[Doc], FullTransformerBatch], None],
#     max_batch_items: int,
#     doc_extension_trf_data: str,
#     doc_extension_prediction: str,
#     labels: List[str],
# ) -> ClassificationTransformer:
#     """Adds the DaNLP BertTone model for detecting whether a statement is subjective to the pipeline.

#     Args:
#         nlp (Language): A spacy text-processing pipeline
#         name (str): Name of the component

#     Returns:
#         ClassificationTransformer: A transformer compenent
#     """
#     return ClassificationTransformer(
#         nlp.vocab,
#         model,
#         set_extra_annotations,
#         max_batch_items=max_batch_items,
#         name=name,
#         labels=labels,
#         doc_extension_trf_data=doc_extension_trf_data,
#         doc_extension_prediction=doc_extension_prediction,
#     )


nlp = spacy.blank("da")
transformer = nlp.add_pipe("subjectivity")
transformer.model.initialize()

texts = [
    "Analysen viser, at økonomien bliver forfærdelig dårlig",
    "Jeg tror alligevel, det bliver godt",
]

docs = nlp.pipe(texts)

for doc in docs:
    print(doc._.subjectivity)
    print(doc._.subjectivity_prob)
# objective
# {'prob': array([1., 0.], dtype=float32), 'labels': ['objective', 'subjective']}
# subjective
# {'prob': array([0., 1.], dtype=float32), 'labels': ['objective', 'subjective']}
