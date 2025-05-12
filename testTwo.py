import pandas as pd
import mteb
import random
random.seed(42)

# Display settings
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 1000)
pd.set_option("display.max_colwidth", 100)

# Benchmark and model
benchmark = mteb.get_benchmark("MTEB(Europe, v1)")
tasks = random.sample(benchmark.tasks, 3)  # ✅ Run only the first task to keep it quick and light
# model_name = "intfloat/multilingual-e5-small"
model_name = "intfloat/multilingual-e5-large-instruct"

model = mteb.get_model(model_name)

# Run evaluation
evaluation = mteb.MTEB(tasks=tasks)
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
df.to_csv("results/summary.csv", index=False)
print(df)
