import mteb
import pandas as pd
from collections import defaultdict

# Load the benchmark
benchmark = mteb.get_benchmark("MTEB(Europe, v1)")
tasks = benchmark.tasks

# Desired task names
selected_task_names = {
    "AlloprofRetrieval",
    "StatcanDialogueDatasetRetrieval",
    "WikipediaRetrievalMultilingual",
    "BelebeleRetrieval",
    "AlloprofReranking",
    "WikipediaRerankingMultilingual",
    "WebLINXCandidatesReranking",
    "DiaBLaBitextMining",
    "BUCCBitextMiningFast",
    "STS17Crosslingual",
    "STSES",
    "STS12"
}

# Filter for selected tasks
selected_tasks = [task for task in tasks if task.__class__.__name__ in selected_task_names]

# Load the model
# MODEL_NAME = "NovaSearch/jasper_en_vision_language_v1"
# MODEL_NAME = "ibm-granite/granite-embedding-107m-multilingual"
MODEL_NAME = "avsolatorio/NoInstruct-small-Embedding-v0"

model = mteb.get_model(MODEL_NAME)

# Run the evaluation
evaluation = mteb.MTEB(tasks=selected_tasks)
results = evaluation.run(model, output_folder="results", return_all_scores=True)

# Collect and save results
data = []

# If only one task, results is a list of TaskResult objects
if isinstance(results, list):
    task_result = results[0]  # Only one task result
    scores = task_result.scores  # This is a dict like {"test": [ {...}, ... ]}
    test_scores = scores.get("test", [])
    if test_scores:
        main_score = test_scores[0].get("main_score", None)
        data.append({
            "model_name": MODEL_NAME,
            "task_name": task_result.task,
            "subset": "test",
            "main_score": main_score,
            **test_scores[0]
        })
else:
    for task_name, subsets in results.items():
        for subset_name, metrics in subsets.items():
            data.append({
                "model_name": MODEL_NAME,
                "task_name": task_name,
                "subset": subset_name,
                "main_score": metrics.get("main_score", None),
                **metrics
            })

# Save results to CSV
df = pd.DataFrame(data)
df.to_csv("results/summary.csv", index=False)
print(df)
