import pandas as pd
import mteb

# set panda tables to max
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)

model_names = ["sentence-transformers/all-MiniLM-L6-v2"]
tasks = mteb.get_tasks(tasks=["Banking77Classification"])
results = mteb.load_results(models=model_names, tasks=tasks)

data = []

for model_result in results.model_results:
    model_name = model_result.model_name
    for task_result in model_result.task_results:
        task_name = task_result.task
        scores = task_result.scores  # this is a dict like {"test": [ {...}, {...}, ... ]}

        # Grab main score from the first test result if it exists
        test_scores = scores.get("test", [])
        if test_scores:
            main_score = test_scores[0].get("main_score", None)
            data.append({
                "model_name": model_name,
                "task_name": task_name,
                "main_score": main_score,
            })

df = pd.DataFrame(data)
print(df)
