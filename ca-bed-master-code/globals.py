from sentence_transformers import SentenceTransformer


SENTENCE_TRANSFORMER = SentenceTransformer(
    "quora-distilbert-multilingual",
    backend="onnx",
)
