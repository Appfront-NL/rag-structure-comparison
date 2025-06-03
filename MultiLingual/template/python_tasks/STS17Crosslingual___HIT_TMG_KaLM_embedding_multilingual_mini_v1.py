import os
import mteb
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

MODEL_NAME = "HIT-TMG/KaLM-embedding-multilingual-mini-v1"
TASK_NAME = "STS17Crosslingual"

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

model = SentenceTransformer(MODEL_NAME)
benchmark = mteb.get_benchmark("MTEB(Europe, v1)")
task = next(t for t in benchmark.tasks if t.__class__.__name__ == TASK_NAME)

evaluation = mteb.MTEB(tasks=[task])
data = []

batch_size = 32
while batch_size >= 8:
    try:
        with torch.no_grad():
            result = evaluation.run(
                model,
                output_folder=f"results/{TASK_NAME}___{MODEL_NAME.replace('/', '_')}",
                return_all_scores=True,
                batch_size=batch_size
            )
        if isinstance(result, list):
            data.extend(result)
        else:
            for task_name, subsets in result.items():
                for subset_name, metrics in subsets.items():
                    data.append({
                        "model_name": MODEL_NAME,
                        "task_name": task_name,
                        "subset": subset_name,
                        "main_score": metrics.get("main_score", None),
                        **metrics
                    })
        torch.cuda.empty_cache()
        break
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            torch.cuda.empty_cache()
            batch_size //= 2
        else:
            raise

pd.DataFrame(data).to_csv(f"results/{TASK_NAME}___{MODEL_NAME.replace('/', '_')}_summary.csv", index=False)
