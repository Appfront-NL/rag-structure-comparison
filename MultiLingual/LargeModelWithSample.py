import pandas as pd
import mteb
import random
from collections import defaultdict

# Display settings
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 1000)
pd.set_option("display.max_colwidth", 100)

MODEL_NAME = "intfloat/multilingual-e5-large-instruct"

# Load benchmark and tasks
benchmark = mteb.get_benchmark("MTEB(Europe, v1)")
tasks = benchmark.tasks

# Group tasks by their 'type'
type_to_tasks = defaultdict(list)
for task in tasks:
    task_type = task.metadata.type  # this is the correct way to access type
    type_to_tasks[task_type].append(task)

# Sample one task per type
random.seed(43)
sampled_tasks = [random.choice(task_list) for task_list in type_to_tasks.values()]

# model_name = "intfloat/multilingual-e5-small"


model = mteb.get_model(MODEL_NAME)

# Run evaluation
evaluation = mteb.MTEB(tasks=sampled_tasks)
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

df = pd.DataFrame(data)
df.to_csv("results/summary.csv", index=False)
print(df)
