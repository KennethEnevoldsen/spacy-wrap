import spacy
import spacy_wrap

def test_clf_trf():
    nlp = spacy.blank("da")

    config = {
        "doc_extension_trf_data": "clf_trf_data",
        "doc_extension_prediction": "hate_speech",
        "labels": ["Not hate Speech", "Hate speech"],
        "model": {
            "name": "DaNLP/da-bert-hatespeech-detection",
        },
    }

    transformer = nlp.add_pipe("classification_transformer", config=config)
    transformer.model.initialize()

    doc = nlp("Senile gamle idiot")
    
    doc._.clf_trf_data
    assert doc._.hate_speech == "Hate speech"
    assert isinstance(doc._.hate_speech_prob, dict)
