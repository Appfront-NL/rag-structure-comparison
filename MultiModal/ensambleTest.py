from sentence_transformers import SentenceTransformer
import numpy as np
import mteb


def match_dims(embs):
    min_dim = min(emb.shape[1] for emb in embs)
    return [emb[:, :min_dim] for emb in embs]

class EnsembleModel:
    def __init__(self, models):
        self.models = models

    def encode(self, sentences, task_name=None, prompt_type=None, **kwargs):
        embs = [model.encode(sentences, **kwargs) for model in self.models]
        embs = [emb.cpu().numpy() if hasattr(emb, "cpu") else emb for emb in embs]
        embs = match_dims(embs)  # Truncate to smallest dimension
        return np.mean(embs, axis=0)



# 2 good french performaing models
models = [
    # SentenceTransformer("Lajavaness/bilingual-embedding-small", trust_remote_code=True),
    SentenceTransformer("thenlper/gte-small"),
    SentenceTransformer("intfloat/e5-small")
    # SentenceTransformer("Lajavaness/bilingual-embedding-base", trust_remote_code=True)
]

ensemble_model = EnsembleModel(models)

tasks = mteb.get_tasks(tasks=["AlloprofRetrieval"])

evaluation = mteb.MTEB(tasks=tasks)

results = evaluation.run(ensemble_model, 
                         output_folder="MultiModal/ensable-results"
)
