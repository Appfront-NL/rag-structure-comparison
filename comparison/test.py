import pandas as pd
import mteb

# # Set pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)

# Use E5 model (requires prompt handling, so use MTEB's wrapper)
model_name = "intfloat/multilingual-e5-large-instruct"
model = mteb.get_model(model_name)

# Select task
tasks = mteb.get_tasks(tasks=["Banking77Classification"])

# Run evaluation
evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(model, output_folder="results", return_all_scores=True)

data = []

# If only one task, results is a list of TaskResult objects
if isinstance(results, list):
    task_result = results[0]  # Only one task result
    scores = task_result.scores  # This is a dict like {"test": [ {...}, ... ]}
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

df = pd.DataFrame(data)
print(df)