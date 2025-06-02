import os
import mteb
import pandas as pd
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import torch

# --- Memory Management Setup ---
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Choose your model
MODEL_NAME = "HIT-TMG/KaLM-embedding-multilingual-mini-v1"

# Load model
model = SentenceTransformer(MODEL_NAME)

# Load benchmark
benchmark = mteb.get_benchmark("MTEB(Europe, v1)")
tasks = benchmark.tasks

# Desired task names
selected_task_names = { # uncommented files done
    # "AlloprofRetrieval",
    # "StatcanDialogueDatasetRetrieval",
    # "WikipediaRetrievalMultilingual",
    # "BelebeleRetrieval",
    # "AlloprofReranking",
    # "WikipediaRerankingMultilingual",
    "WebLINXCandidatesReranking",
    # "DiaBLaBitextMining",
    # "BUCCBitextMiningFast",
    # "STS17Crosslingual",
    # "STSES",
    # "STS12"
}

# Filter tasks
selected_tasks = [task for task in tasks if task.__class__.__name__ in selected_task_names]

# Setup evaluation
evaluation = mteb.MTEB(tasks=selected_tasks)

# Store results
data = []

# Run one task at a time to manage memory
for task in selected_tasks:
    print(f"\n➡️ Running task: {task.__class__.__name__}")

    # Try a large batch size first, reduce if OOM happens
    batch_size = 64
    while batch_size >= 8:
        try:
            with torch.no_grad():  # inference-only mode
                result = evaluation.run(
                    model,
                    output_folder=f"ML-results-test/{task.__class__.__name__}",
                    return_all_scores=True,
                    batch_size=batch_size
                )

            # Collect results
            for task_name, subsets in result.items():
                for subset_name, metrics in subsets.items():
                    data.append({
                        "model_name": MODEL_NAME,
                        "task_name": task_name,
                        "subset": subset_name,
                        "main_score": metrics.get("main_score", None),
                        **metrics
                    })

            # Clear CUDA memory between tasks
            torch.cuda.empty_cache()
            break  # break the batch-size loop if successful

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"❗ CUDA OOM at batch size {batch_size}, reducing...")
                torch.cuda.empty_cache()
                batch_size = batch_size // 2
            else:
                raise e  # re-raise non-OOM errors

# Save results to CSV
df = pd.DataFrame(data)
df.to_csv("results/summary.csv", index=False)
print(df)
