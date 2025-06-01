import mteb
import pandas as pd
from collections import defaultdict
import os


# List of models to evaluate
model_names = [
    # "Lajavaness/bilingual-embedding-base",
    "aari1995/German_Semantic_STS_V2",
    "avsolatorio/GIST-large-Embedding-v0"
]

# Load the benchmark and filter tasks
benchmark = mteb.get_benchmark("MTEB(Europe, v1)")
tasks = benchmark.tasks
selected_task_names = {
    "AlloprofRetrieval",
    "StatcanDialogueDatasetRetrieval",
    "WikipediaRetrievalMultilingual",
    "BelebeleRetrieval",
    "AlloprofReranking",
    "WikipediaRerankingMultilingual",
    # "WebLINXCandidatesReranking",
    "DiaBLaBitextMining",
    "BUCCBitextMiningFast",
    "STS17Crosslingual",
    "STSES",
    "STS12"
}
selected_tasks = [task for task in tasks if task.__class__.__name__ in selected_task_names]

# Evaluate each model
for model_name in model_names:
    print(f"Running evaluation for: {model_name}")
    model = mteb.get_model(model_name)
    
    # Create a unique output folder per model
    safe_model_name = model_name.replace("/", "_")
    output_folder = f"ResultsMultiModal/{safe_model_name}"
    os.makedirs(output_folder, exist_ok=True)
    
    evaluation = mteb.MTEB(tasks=selected_tasks)
    # results = evaluation.run(model, output_folder=output_folder, return_all_scores=True)
    results = evaluation.run(
        model,
        output_folder=output_folder,
        return_all_scores=True,
        encode_kwargs={"batch_size": 8}  # or try even smaller like 4 or 2
    )
    # Collect results
    data = []
    if isinstance(results, list):
        task_result = results[0]
        scores = task_result.scores
        test_scores = scores.get("test", [])
        if test_scores:
            main_score = test_scores[0].get("main_score", None)
            data.append({
                "model_name": model_name,
                "task_name": task_result.task,
                "subset": "test",
                "main_score": main_score,
                **test_scores[0]
            })
    else:
        for task_name, subsets in results.items():
            for subset_name, metrics in subsets.items():
                data.append({
                    "model_name": model_name,
                    "task_name": task_name,
                    "subset": subset_name,
                    "main_score": metrics.get("main_score", None),
                    **metrics
                })

    # Save summary CSV
    df = pd.DataFrame(data)
    summary_csv = os.path.join(output_folder, "summary.csv")
    df.to_csv(summary_csv, index=False)
    print(f"Results saved to {summary_csv}")