import os
from transformers import pipeline

sentiment_fn = pipeline(
        "sentiment-analysis",
        "lvwerra/distilbert-imdb",
        top_k=2,
        truncation=True,
        batch_size=256,
        device=0 if int(os.environ.get("LOCAL_RANK", 0)) == 0 else -1,
)

def get_positive_score(scores):
    "Extract value associated with a positive sentiment from pipeline's output"
    return dict(map(lambda x: tuple(x.values()), scores))["POSITIVE"]

def metric_fn(samples: List[str], **kwargs) -> Dict[str, List[float]]:
    sentiments = list(map(get_positive_score, sentiment_fn(samples)))
    return dict(sentiments=sentiments)